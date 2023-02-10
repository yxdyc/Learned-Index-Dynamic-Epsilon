//
// Created by daoyuan on 2019/12/25.
//

#ifndef LEARNED_INDEX_RMI_INDEXER_LINEAR_HPP
#define LEARNED_INDEX_RMI_INDEXER_LINEAR_HPP


#include <vector>
#include <list>
#include <chrono>
#include <cstddef>
#include "IndexMechanism.hpp"
#include "cstddef"
#include "cstdio"
#include "cstdlib"
#include "cstring"
#include "PgmIndexer.hpp"
#include "ctime"
#include "Utilities.hpp"

#include <math.h>


template <typename KeyType, typename Iterator>
class RMILinearIndexer {
    /*
     * A two-layer RMI linear model
     */
public:

    struct RMIPos{
        double pos;
        size_t lower_error_, upper_error_;

        RMIPos(double predicted_pos, size_t lower_error, size_t upper_error){
            pos = predicted_pos;
            lower_error_ = lower_error;
            upper_error_ = upper_error;
        }
    };

    struct LinearModel{
        double alpha_, beta_;
        size_t lower_error_, upper_error_; // both are positive numbers
        // the "mean_x, mean_y, c, m2, n" are stored to support online model update (re-training)
        double mean_x_, mean_y_, c_, m2_;
        size_t n_;

        LinearModel(){
            alpha_ = 0;
            beta_ = 0;
            n_ = 0;
            mean_x_ = 0;
            mean_y_ = 0;
            c_= 0.0;
            m2_ = 0.0;
            lower_error_ = std::numeric_limits<unsigned long>::min();
            upper_error_ = std::numeric_limits<unsigned long>::min();
        }
        LinearModel(double alpha, double beta, size_t n):
            alpha_(alpha), beta_(beta), n_(n){
            n_ = 0;
            mean_x_ = 0;
            mean_y_ = 0;
            c_= 0.0;
            m2_ = 0.0;
        };

        inline double linear(double inp) {
            return alpha_ + beta_ * inp;
        }

        /*
         * train the linear model
         *
         * @params compress_key weather transform the keys using log()
         * @params second_model_size, all_data_size used to re-scale the model
         */
        void train(Iterator first, Iterator end, bool compress_key = false,
                size_t second_model_size=0, size_t all_data_size = 0){
            double dx , dx2;
            size_t data_size = 0;
            for (Iterator it = first; it != end; it ++){
                double x = (*it).first;
                if (compress_key){
                    //x = log(x - 20170117000000 + 1);
                    x = log(x);
                }
                size_t y = (*it).second;
                if (second_model_size > 0){
                    // if second_model_size > 0, indicating the non-last layer model, rescale the y
                    y = floor(double (y) * second_model_size / all_data_size);
                }
                n_ += 1;
                dx = x - mean_x_;
                mean_x_ += dx / double(n_);
                mean_y_ += (y - mean_y_) / double(n_);
                c_ += dx * (y - mean_y_);

                dx2 = x - mean_x_;
                m2_ += dx * dx2;
                data_size ++;
            }
            if (data_size == 0){
                alpha_ = 0;
                beta_ = 0;
                return;
            } else if (data_size == 1 and alpha_ == 0){
                alpha_ = mean_y_;
                beta_ = 0;
                return;
            }

            double cov = c_ / double(n_ - 1), var = m2_ / double (n_ - 1);
            assert(var >= 0.0);
            if (var == 0.0){
                alpha_ = mean_y_;
                beta_ = 0;
                return;
            }

            beta_ = cov / var;
            alpha_ = mean_y_ - beta_ * mean_x_;
            return;
        }

        /*
         * the standard RMI stores the min-error and max-error for every model on the last stage, such that:
         * each key can be searched by binary search with respective lower and upper bound for every model
         */
        void track_lower_upper_bound(Iterator begin, Iterator end){
            for (auto it = begin; it != end; it ++){
                long pred_error = long(round(linear((*it).first))) - long((*it).second);
                if (pred_error < 0) {
                    pred_error = abs(pred_error);
                    //lower_error_ = pred_error > lower_error_ ? pred_error : lower_error_;
                    upper_error_ = pred_error > upper_error_ ? pred_error : upper_error_;
                } else {
                    //upper_error_ = pred_error > upper_error_ ? pred_error : upper_error_;
                    lower_error_ = pred_error > lower_error_ ? pred_error : lower_error_;
                }
            }
        }

    };



    size_t rmi_size_;
    size_t second_model_size_;
    std::set<size_t> untrained_model_ids_;
    std::vector<int> model_trained_indicators_;
    bool complete_submodels_;
    bool compress_key_, find_near_seg_;
    double sample_rate_;
    std::string search_strategy_;


    Iterator first_key_iter_, last_key_iter_;
    size_t data_size_;

    std::vector<std::vector<std::pair<key_type_transformed, size_t>>> data_partitions_;
    std::vector<LinearModel> last_layer_models_;

    double L0_alpha_, L0_beta_;

    RMILinearIndexer(Iterator first_key_iter, Iterator last_key_iter, size_t data_size, size_t second_model_size):
            model_trained_indicators_(second_model_size, 1) {
        first_key_iter_ = first_key_iter;
        last_key_iter_ = last_key_iter;
        search_strategy_ = "binary_search";
        data_size_ = data_size;
        second_model_size_ = second_model_size;
        compress_key_ = false;
        for(int i = 0; i < second_model_size_; i++){
            LinearModel tmp_model;
            last_layer_models_.emplace_back(tmp_model);
            std::vector<std::pair<key_type_transformed, size_t>> tmp_data;
            data_partitions_.emplace_back(tmp_data);
        }
    }

    inline double linear(double alpha, double beta, double inp) {
        return alpha + beta * inp;
    }


    inline double cubic(double a, double b, double c, double d, double x) {
        return (((a * x + b) * x + c) * x) + d;
    }


    inline size_t FCLAMP(double inp, double bound) {
        if (inp < 0.0) return 0;
        return round(inp > bound ? bound : inp);
    }

/* learn RMI index based on a given sample rate
 * @return the construction time excluding the sampling time
 */
    std::chrono::system_clock::time_point learn_index(Iterator first_iter, Iterator last_iter, float sample_rate = 1.0) {
        std::vector<std::pair<KeyType, size_t>> sampled_data;
        Iterator sampled_first_iter, sampled_last_iter;
        sample_rate_ = sample_rate;
        if (sample_rate != 1.0) {
            size_t sample_size = round(data_size_ * sample_rate);
            std::experimental::sample(first_iter, last_iter, std::back_inserter(sampled_data), sample_size,
                                      std::mt19937{std::random_device{}()});
            std::sort(sampled_data.begin(), sampled_data.end());
            sampled_first_iter = sampled_data.begin();
            sampled_last_iter = sampled_data.end();
        } else{
            sampled_first_iter = first_iter;
            sampled_last_iter = last_iter;
        }
        std::chrono::system_clock::time_point t0 = std::chrono::high_resolution_clock::now();

        std::cout<<"Training first layer model"<<std::endl;
        LinearModel model_0;
        model_0.train(sampled_first_iter, sampled_last_iter, compress_key_, second_model_size_, data_size_);

        L0_alpha_ = model_0.alpha_;
        L0_beta_ = model_0.beta_;


        for (auto it = sampled_first_iter; it != sampled_last_iter; it++){
            double key = (double) (*it).first;
            if (compress_key_){
                //key = log(key - 20170117000000 + 1);
                key = log(key);
            }
            double fp = linear(L0_alpha_, L0_beta_, key);
            //model_idx_predict = floor(double (model_idx_predict) * second_model_size_ / data_size_);
            size_t model_idx_predict = FCLAMP(fp, second_model_size_ - 1);
            data_partitions_[model_idx_predict].emplace_back(*it);
        }

        int zero_data_partitions_number = 0;
        for (int i = 0; i < second_model_size_; i++){
            auto first = data_partitions_[i].begin();
            auto end = data_partitions_[i].end();
            auto partition_size = std::distance(first, end);
            if (partition_size == 0){
                zero_data_partitions_number ++;
                model_trained_indicators_[i] = 0;
            } else {
                last_layer_models_[i].train(first, end, compress_key_);
                last_layer_models_[i].track_lower_upper_bound(first, end);
            }
        }
        std::cout<<"The number of zero data partitions is: " << zero_data_partitions_number << std::endl;

        size_t re_trained_number = 0;
        if (complete_submodels_) {
            // re-training the blank last_layer_models
            for (auto it = first_iter; it != last_iter; it++) {
                double key = (double) (*it).first;
                if (compress_key_) {
                    // key = log(key - 20170117000000 + 1);
                    key = log(key);
                }
                double fp = linear(L0_alpha_, L0_beta_, key);
                size_t model_idx_predict = FCLAMP(fp, second_model_size_ - 1);
                if (last_layer_models_[model_idx_predict].alpha_ == 0){
                    data_partitions_[model_idx_predict].emplace_back(*it);
                    untrained_model_ids_.insert(model_idx_predict);
                    re_trained_number ++;
                }

                // if (last_layer_models_[model_idx_predict].alpha_ == 0 or
                //         (untrained_model_ids_.find(model_idx_predict) != untrained_model_ids_.end())){
                //     untrained_model_ids_.insert(model_idx_predict);
                //     last_layer_models_[model_idx_predict].train(it, it+1);
                //     re_trained_number ++;
                // }
            }
            for(auto it = untrained_model_ids_.begin(); it != untrained_model_ids_.end(); it ++){
                size_t i = *it;
                auto first = data_partitions_[i].begin();
                auto end = data_partitions_[i].end();
                last_layer_models_[i].train(first, end, compress_key_);
            }
            std::cout<<"The number of re-trained data is: " << re_trained_number << std::endl;
            std::cout<<"The number of re-trained sub model is: " << untrained_model_ids_.size() << std::endl;
        }

        return t0;
    }


    void print_segments_statistics(){
        std::vector<double_t > slopes;
        int model_idx = 0;

        for (auto x : last_layer_models_){
            if (model_trained_indicators_[model_idx] == 0){
                model_idx++;
                continue;
            } else{
                model_idx++;
                slopes.emplace_back(x.beta_);
                if (x.beta_ > 100000){
                    int debug = 0;
                }
            }
        }
        std::cout<< "The non-blank segment number is: " << slopes.size() << "; ";
        calculate_mean_std(slopes, true);
        return;
    }



/**
 * evaluate index on the init dataset
 * @write_to_file output the predictions and real positions to file
 * @shuffle evaluation with sequential query order or random query order
 */
    void evaluate_indexer(int payload_size, bool write_to_file, bool shuffle=true) {
        std::vector<std::pair<key_type_transformed, size_t>> tmp_data(first_key_iter_, last_key_iter_);
        if (shuffle == true){
            std::srand(1234);
            random_shuffle(tmp_data.begin(), tmp_data.end());
        }

        std::vector<RMIPos> all_predicted_pos;
        auto t0 = std::chrono::high_resolution_clock::now();
        for (Iterator it = tmp_data.begin(); it != tmp_data.end(); it++) {
            RMIPos pos = predict_position((*it).first);
            all_predicted_pos.emplace_back(pos);
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        int i = 0;
        size_t wrong_return_count = 0;
        for (Iterator it = tmp_data.begin(); it != tmp_data.end(); it++, i++) {
            size_t corrected_res = correct_position(all_predicted_pos[i], (*it).first);
            if (corrected_res != (*it).second) {
                wrong_return_count ++;
                RMIPos pos = predict_position((*it).first);
                auto wrong_res = correct_position(all_predicted_pos[i], (*it).first);
            }
        }
        auto t2 = std::chrono::high_resolution_clock::now();
        //lookup-time per query
        auto predict_time = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count() / data_size_;
        auto correct_time = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() / data_size_;
        //size
        size_t size_of_index = this->index_size_in_bytes();
        size_t size_of_index_payloads = this->index_size_in_bytes_with_payload(payload_size);

        // ML-oriented metric
        // regress_metric = ["RMSE", "MAE", "R2"]
        std::vector<size_t> all_true_pos, all_predicted_pos_without_metainfo;
        size_t zero_prediction_count = 0;
        for (size_t i = 0; i < data_size_; i++) {
            all_true_pos.push_back(tmp_data[i].second);
            size_t predicted_pos_without_metainfo = round(all_predicted_pos[i].pos);
            all_predicted_pos_without_metainfo.push_back(predicted_pos_without_metainfo);
            if (predicted_pos_without_metainfo == 0){
                zero_prediction_count ++;
            }
        }

        std::cout << "predict time: " << predict_time << std::endl;
        std::cout << "correct time: " << correct_time << std::endl;
        std::cout << "overall lookup time: " << predict_time + correct_time << std::endl;
        std::cout << "size of index: " << size_of_index << std::endl;
        std::cout << "size of index with payloads: " << size_of_index_payloads << std::endl;
        std::cout << "wrong return count: " << wrong_return_count << std::endl;
        dlib::matrix<double, 1, 4> results;
        results = evaluate_regression_ML(all_predicted_pos_without_metainfo, all_true_pos, 10000, write_to_file);
        std::cout << "ML oriented matricx: " << results << std::endl;

    }

    /*
     * evaluate index on given dataset with random query order
     */
    template <typename T_data>
    void evaluate_indexer(T_data data, int payload_size, bool write_to_file=false) {
        T_data random_data = data;
        auto tmp_first_key_iter = this->first_key_iter_;
        auto tmp_last_key_iter = this->last_key_iter_;

        this->first_key_iter_ = data.begin(); // we need search on the whole dataset
        this->last_key_iter_ = data.end(); // we need search on the whole datase
        std::srand(1234);
        random_shuffle(random_data.begin(), random_data.end());

        std::vector<RMIPos> all_predicted_pos;
        auto t0 = std::chrono::high_resolution_clock::now();
        for (Iterator it = random_data.begin(); it != random_data.end(); it++) {
            RMIPos pos = predict_position((*it).first);
            all_predicted_pos.emplace_back(pos);
        }
        assert(all_predicted_pos.size() == data_size_);

        auto t1 = std::chrono::high_resolution_clock::now();
        int i = 0;
        size_t wrong_return_count = 0;
        for (Iterator it = random_data.begin(); it != random_data.end(); it++, i++) {
            KeyType corrected_res = correct_position(all_predicted_pos[i], (*it).first);
            assert(isEqual(corrected_res, (*it).first));
#ifdef Debug
            if (corrected_res != (*it).second) {
                wrong_return_count ++;
                RMIPos pos = predict_position((*it).first);
                auto wrong_res = correct_position(all_predicted_pos[i], (*it).first);
            }
#endif
        }
        auto t2 = std::chrono::high_resolution_clock::now();
        //lookup-time per query
        auto predict_time = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count() / data_size_;
        auto correct_time = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() / data_size_;
        //size
        size_t size_of_index = this->index_size_in_bytes();
        size_t size_of_index_payloads = this->index_size_in_bytes_with_payload(payload_size);

        // ML-oriented metric
        // regress_metric = ["RMSE", "MAE", "R2"]
        std::vector<size_t> all_true_pos, all_predicted_pos_without_metainfo;

#ifdef Debug
        size_t zero_prediction_count = 0;
        for (size_t i = 0; i < data_size_; i++) {
            all_true_pos.push_back(i);
            size_t predicted_pos_without_metainfo = round(all_predicted_pos[i].pos);
            all_predicted_pos_without_metainfo.push_back(predicted_pos_without_metainfo);
            if (predicted_pos_without_metainfo == 0){
                zero_prediction_count ++;
            }
        }
        std::cout << "zero prediction count: " << wrong_return_count << std::endl;
        std::cout << "wrong return count: " << wrong_return_count << std::endl;
#endif
        std::cout << "predict time: " << predict_time << std::endl;
        std::cout << "correct time: " << correct_time << std::endl;
        std::cout << "overall lookup time: " << predict_time + correct_time << std::endl;
        std::cout << "size of index: " << size_of_index << std::endl;
        std::cout << "size of index with payloads: " << size_of_index_payloads << std::endl;

        dlib::matrix<double, 1, 4> results;
        results = evaluate_regression_ML(all_predicted_pos_without_metainfo, all_true_pos, 10000, write_to_file);
        std::cout << "ML oriented matricx: " << results;

        this->first_key_iter_ = tmp_first_key_iter;
        this->last_key_iter_ = tmp_last_key_iter;

    }

    // void reset_gapped_linking_array(){
    //     gapped_array_.clear();
    //     linking_array_.clear();
    // }

    // void set_gapped_linking_array(
    //         std::vector<std::pair<KeyType, std::vector<KeyType>>> const & gapped_array_with_linking_array){
    //     reset_gapped_linking_array();
    //     size_t i = 0;
    //     gapped_array_.reserve(gapped_array_with_linking_array.size());
    //     linking_array_.reserve(gapped_array_with_linking_array.size());
    //     for (auto key_pair : gapped_array_with_linking_array){
    //         gapped_array_.emplace_back(key_pair.first);
    //         linking_array_.emplace_back(key_pair.second);
    //     }
    // }



    void evaluate_indexer_analysis(int payload_size, bool write_to_file) {
        std::vector<RMIPos> all_predicted_pos;
        auto t0 = std::chrono::high_resolution_clock::now();
        for (Iterator it = first_key_iter_; it != last_key_iter_; it++) {
            RMIPos pos = predict_position(*it);
            all_predicted_pos.emplace_back(pos);
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        int i = 0;
        size_t wrong_return_count = 0;
        for (Iterator it = first_key_iter_; it != last_key_iter_; it++, i++) {
            size_t corrected_res = correct_position(all_predicted_pos[i], *it);
            if (corrected_res != i) {
                wrong_return_count ++;
            }
        }
        auto t2 = std::chrono::high_resolution_clock::now();
        //lookup-time per query
        auto predict_time = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count() / data_size_;
        auto correct_time = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() / data_size_;
        //size
        size_t size_of_index = this->index_size_in_bytes();

        // ML-oriented metric
        // regress_metric = ["RMSE", "MAE", "R2"]
        std::vector<size_t> all_true_pos, all_predicted_pos_without_metainfo;
        size_t zero_prediction_count = 0;
        for (size_t i = 0; i < data_size_; i++) {
            all_true_pos.push_back(i);
            size_t predicted_pos_without_metainfo = round(all_predicted_pos[i]);
            size_t delta = predicted_pos_without_metainfo > i ?
                           predicted_pos_without_metainfo - i : i - predicted_pos_without_metainfo;
            all_predicted_pos_without_metainfo.push_back(predicted_pos_without_metainfo);
            if (predicted_pos_without_metainfo == 0){
                zero_prediction_count ++;
            } else{
                if  (delta > 200000 and predicted_pos_without_metainfo != 0){
                    double pos = predict_position(*(first_key_iter_ + i)).pos;
                    double numric_error = pos - i;
                }
            }
        }
        dlib::matrix<double, 1, 4> results;
        results = evaluate_regression_ML(all_predicted_pos_without_metainfo, all_true_pos, 200000, write_to_file);
        std::cout << "ML oriented matricx for non-zero keys: " << results << std::endl;
        results = evaluate_regression_ML(all_predicted_pos_without_metainfo, all_true_pos, 200000, write_to_file);
        std::cout << "ML oriented matricx for zero keys: " << results << std::endl;
    }

    void out_segments_spans(std::string out_path){
        std::vector<KeyType> delta_x_of_each_seg;

        for (auto it = data_partitions_.begin(); it != data_partitions_.end() - 1; it++) {
            if ((*it).size() == 0){
                // the blank data_partitions
                delta_x_of_each_seg.emplace_back(0);
                continue;
            } else{
                KeyType parti_begin = (*((*it).begin())).first;
                KeyType parti_end = (*((*it).end() - 1)).first;
                KeyType delta_x = parti_end - parti_begin;
                if (delta_x > 10000000){
                    int stop = 1;
                }
                delta_x_of_each_seg.emplace_back(delta_x);
            }
       }
        write_vector_to_f(delta_x_of_each_seg, out_path);
    }


    void evaluate_indexer_analysis_for_sampled_keys(int payload_size, bool write_to_file) {
        std::set<KeyType> sampled_keys;
        std::ifstream fin("/home/xxx/work/learned_index/ref_implements/RMI/sampled_keys_0.9_50k.txt");
        std::string line;
        if (!fin){
            std::cout << "open sampled file failure." << std::endl;
            exit(1);
        }
        while (getline(fin, line)) {
            sampled_keys.insert(std::stoll(line));
        }

        std::vector<RMIPos> all_predicted_pos;
        auto t0 = std::chrono::high_resolution_clock::now();
        for (Iterator it = first_key_iter_; it != last_key_iter_; it++) {
            RMIPos pos = predict_position(*it);
            all_predicted_pos.emplace_back(pos);
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        int i = 0;
        size_t wrong_return_count = 0;
        for (Iterator it = first_key_iter_; it != last_key_iter_; it++, i++) {
            size_t corrected_res = correct_position(all_predicted_pos[i], *it);
            if (corrected_res != i) {
                wrong_return_count ++;
            }
        }
        auto t2 = std::chrono::high_resolution_clock::now();
        //lookup-time per query
        auto predict_time = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count() / data_size_;
        auto correct_time = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() / data_size_;
        //size
        size_t size_of_index = this->index_size_in_bytes();

        // ML-oriented metric
        // regress_metric = ["RMSE", "MAE", "R2"]
        std::vector<size_t> all_true_pos_sampled, all_predicted_pos_without_metainfo_sampled;
        std::vector<size_t> all_true_pos_non_sampled, all_predicted_pos_without_metainfo_non_sampled;
        for (size_t i = 0; i < data_size_; i++) {
            size_t key = (*(first_key_iter_ + i));
            if (sampled_keys.find(key) != sampled_keys.end()){
                all_true_pos_sampled.push_back(i);
                size_t predicted_pos_without_metainfo = round(all_predicted_pos[i]);
                size_t delta = predicted_pos_without_metainfo > i ?
                               predicted_pos_without_metainfo - i : i - predicted_pos_without_metainfo;
                all_predicted_pos_without_metainfo_sampled.push_back(predicted_pos_without_metainfo);
            } else{
                all_true_pos_non_sampled.push_back(i);
                size_t predicted_pos_without_metainfo = round(all_predicted_pos[i]);
                size_t delta = predicted_pos_without_metainfo > i ?
                               predicted_pos_without_metainfo - i : i - predicted_pos_without_metainfo;
                all_predicted_pos_without_metainfo_non_sampled.push_back(predicted_pos_without_metainfo);
            }
        }
        std::cout << "The size of sampled and none-sampled are: " << all_true_pos_sampled.size() << ", " << all_true_pos_non_sampled.size() << std::endl;
        dlib::matrix<double, 1, 4> results;
        results = evaluate_regression_ML(all_predicted_pos_without_metainfo_sampled, all_true_pos_sampled, 200000, write_to_file);
        std::cout << "ML oriented matricx for sampled keys: " << results;
        results = evaluate_regression_ML(all_predicted_pos_without_metainfo_non_sampled, all_true_pos_non_sampled, 200000, write_to_file);
        std::cout << "ML oriented matricx for non-sampled keys: " << results << std::endl;
    }


    std::vector<double> get_segment_stats(int seg_size=0) {
        if (seg_size == 0){
            seg_size = second_model_size_;
        }
        std::vector<double> seg_stats;
        std::map<KeyType, double> seg_cover_counter;
        for (int j = 0; j < seg_size; ++j) {
            seg_stats.emplace_back(0.0);
        }
        int i = 0;
        for(Iterator it = first_key_iter_; it != last_key_iter_; it++, i++){
            KeyType model_idx = find_seg_key((*it));
            seg_stats[model_idx] += 1.0;
        }
        return seg_stats;
    }

    inline KeyType find_seg_key(KeyType key){
        size_t modelIndex;
        double fpred;
        auto res = double (key);
        fpred = linear(L0_alpha_, L0_beta_, (double) key);
        modelIndex = FCLAMP(fpred, second_model_size_ - 1.0);
        return modelIndex;
    }

    RMIPos predict_position(KeyType key) {
        size_t modelIndex;
        double fpred;
        double key_double = (double) key;
        if (compress_key_){
            //key_double = log(key_double - 20170117000000 + 1);
            key_double = log(key_double);
        }
        fpred = linear(L0_alpha_, L0_beta_, key_double);
        modelIndex = FCLAMP(fpred, second_model_size_ - 1.0);
        if (find_near_seg_ and model_trained_indicators_[modelIndex] == 0){
            int i = 1, idx_low(1), idx_high(0);
            if ((second_model_size_-1 - modelIndex) > (modelIndex-0)){
                modelIndex = 0;
            } else{
                modelIndex = second_model_size_-1;
            }
            while (idx_low != 0 and (idx_high != (second_model_size_-1))){
                idx_low = FCLAMP(double(fpred-i), second_model_size_ - 1.0);
                if(model_trained_indicators_[idx_low] != 0){
                    modelIndex = idx_low;
                    break;
                }
                idx_high = FCLAMP(double(fpred+i), second_model_size_ - 1.0);
                if(model_trained_indicators_[idx_high] != 0){
                    modelIndex = idx_high;
                    break;
                }
                i++;
            }
        }

        fpred = linear(last_layer_models_[modelIndex].alpha_, last_layer_models_[modelIndex].beta_,
                       key_double);
        fpred = FCLAMP(fpred, data_size_ - 1.0);
        // adjust the lower_error and upper_error to avoid "index out of bounds" [0, data_size_-1]
        auto lower_error = last_layer_models_[modelIndex].lower_error_;
        lower_error = lower_error >= fpred ? fpred : lower_error;
        auto upper_error = last_layer_models_[modelIndex].upper_error_;
        upper_error = (upper_error + fpred) >= (data_size_ - 1 - fpred) ? (data_size_ - 1 - fpred) : upper_error;
        return RMIPos(fpred, lower_error, upper_error);
    }

    KeyType correct_position(RMIPos predicted_pos, KeyType key) {
        Iterator true_pos_iter;
        if (search_strategy_ == "binary_search"){
            KeyType pos = KeyType(predicted_pos.pos);
            auto lower = first_key_iter_ + (pos - predicted_pos.lower_error_);
            auto upper = first_key_iter_ + (pos + predicted_pos.upper_error_);
            true_pos_iter = std::lower_bound(lower, upper, key, CompareForDataPair<KeyType>());
        } else if (search_strategy_ == "exp_search" or search_strategy_ == "exponential_search"){
            true_pos_iter = exponential_search(first_key_iter_, last_key_iter_, key,
                    KeyType(round(predicted_pos.pos)), CompareForDataPair<KeyType>());
        } else{
            throw "Wrong search setrategy";
        }
        //return std::distance(this->first_key_iter_, true_pos_iter);
        return (*true_pos_iter).second;

    };

    /**
    * @return the size in bytes of the index models.
    */
    size_t index_size_in_bytes() {
        size_t size_of_linear = sizeof(LinearModel);
        rmi_size_ = second_model_size_ * size_of_linear;
        //return size_in_bytes;
        return rmi_size_;
    };

    /**
    * @return the size in bytes of the index models including a dummy payloads
    */
    size_t index_size_in_bytes_with_payload(int payload) {
        return this->index_size_in_bytes() + data_size_ * payload;
    };

    std::vector<FittingSegmentModified<key_type_transformed, float_t>> linear_models_to_segments() {
        std::vector<FittingSegmentModified<key_type_transformed, float_t>> segments;
        for (int i = 0; i < second_model_size_; i++){
            auto first = data_partitions_[i].begin();
            auto end = data_partitions_[i].end();
            auto partition_size = std::distance(first, end);
            if (partition_size > 0){
                auto seg_begin = (*first).first;
                auto seg_end = (*(end-1)).first;
                auto seg_end_y = (*(end-1)).second;
                auto seg_slope = last_layer_models_[i].beta_;
                auto seg_intercept = last_layer_models_[i].alpha_;

                FittingSegmentModified<key_type_transformed, float_t> tmp_seg(seg_begin, seg_slope,
                        seg_intercept, seg_end, seg_end_y);
                segments.emplace_back(tmp_seg);
            }
        }

        return segments;

    }

    void clear_index() {
        data_partitions_.clear();
        untrained_model_ids_.clear();
        last_layer_models_.clear();
        for(int i = 0; i < second_model_size_; i++){
            model_trained_indicators_[i] = 1;
            LinearModel tmp_model;
            last_layer_models_.emplace_back(tmp_model);
            std::vector<std::pair<key_type_transformed, size_t>> tmp_data;
            data_partitions_.emplace_back(tmp_data);
        }
    }
};




#endif //LEARNED_INDEX_BTREE_INDEXER_HPP
