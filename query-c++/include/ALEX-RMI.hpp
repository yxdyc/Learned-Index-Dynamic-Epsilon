//
// Created by daoyuan on 2019/12/25.
//

#ifndef LEARNED_INDEX_ALEX_RMI_HPP
#define LEARNED_INDEX_ALEX_RMI_HPP


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
class ALEX_RMILinearIndexer {
public:

    inline static double linear(double alpha, double beta, double inp) {
        return alpha + beta * inp;
    }


    inline double cubic(double a, double b, double c, double d, double x) {
        return (((a * x + b) * x + c) * x) + d;
    }


    inline static size_t FCLAMP(double inp, double bound) {
        if (inp < 0.0) return 0;
        return (inp > bound ? bound : (size_t)inp);
    }

    struct LinearModel{
        double alpha_, beta_;
        double mean_x_, mean_y_, c_, m2_;
        size_t n_, second_model_size_;
        LinearModel(){ // track statistics to support online learning
            alpha_ = 0;
            beta_ = 0;
            n_ = 0;
            mean_x_ = 0;
            mean_y_ = 0;
            c_= 0.0;
            m2_ = 0.0;
            second_model_size_ = 0;
        }
        void clean(){
            alpha_ = 0;
            beta_ = 0;
            n_ = 0;
            mean_x_ = 0;
            mean_y_ = 0;
            c_= 0.0;
            m2_ = 0.0;
            second_model_size_ = 0;
        }
        LinearModel(double alpha, double beta, size_t n):
            alpha_(alpha), beta_(beta), n_(n){
            n_ = 0;
            mean_x_ = 0;
            mean_y_ = 0;
            c_= 0.0;
            m2_ = 0.0;
        };
        inline size_t predict_next_idx(KeyType key){
            size_t modelIndex;
            double fpred;
            auto res = double (key);
            fpred = linear(alpha_, beta_, (double) key);
            modelIndex = FCLAMP(fpred, second_model_size_ - 1.0);
            return modelIndex;
        }
        void train(Iterator first, Iterator end, bool compress_key = false, size_t next_keys_size=0, size_t all_data_size = 0,
                   bool sequential_y = false){
            if (second_model_size_ == 0){
                second_model_size_ = next_keys_size;
            }
            double dx , dx2;
            size_t data_size = 0;
            int i = 0;
            for (Iterator it = first; it != end; it ++, i++){
                double x = (*it).first;
                if (compress_key){
                    //x = log(x - 20170117000000 + 1);
                    x = log(x);
                }
                size_t y = i;
                if (not sequential_y){
                    y = (*it).second;
                }
                if (next_keys_size > 0){
                    // if second_model_size > 0, indicating the non-last layer model, rescale the y
                    y = floor(double (next_keys_size) / all_data_size * y);
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


    };


    struct Node{
        std::vector<std::pair<KeyType, size_t>> keys_;
        size_t num_non_null_keys_;
        double density_;
        LinearModel model_;
        // used to map the predi to the child node idx since ALEX merges multiple adjacent partitions
        std::vector<size_t> predictions_to_child_idx_;
        std::vector<Node> child_nodes_;

        Node(double density, size_t second_model_size): model_(){
            num_non_null_keys_ = 0;
            density_ = density;
            model_.second_model_size_ = second_model_size;
            for (int i = 0; i < model_.second_model_size_; i++){
                predictions_to_child_idx_.emplace_back(i); // init: model prediction i -> child i
            }
        }
        Node(std::vector<std::pair<KeyType, size_t>> keys, double density, size_t second_model_size): model_(){
            keys_ = keys;
            num_non_null_keys_ = keys.size();
            density_ = density;
            model_.second_model_size_ = second_model_size;
            for (int i = 0; i < model_.second_model_size_; i++){
                predictions_to_child_idx_.emplace_back(i); // init: model prediction i -> child i
            }
        }

        void expand(std::string node_layout){
            size_t expand_size;
            if (node_layout == "GappedArray"){
                expand_size = keys_.size() / density_;
            } else if (node_layout == "PackedMemoryArray"){
                expand_size = keys_.size() * 2;
            } else{
                throw std::invalid_argument( "Node layout must be GappedArray or PackedMemoryArray." );
            }
            std::vector<std::pair<KeyType, size_t>> expand_keys;
            for(int i = 0; i < expand_size; i++){
                // To simplify the gap checking, here we use key '0' to indicate the gap.
                // A more general way is using bitmap as mentioned in ALEX paper.
                expand_keys.emplace_back(std::make_pair(0, i));
            }
            // retrain the model based on the existing keys, rescale by the expand_size;
            bool sequential_y_when_init = true; // only leaf node will call the expand() function
            model_.clean();
            model_.train(keys_.begin(), keys_.end(), false, expand_size, keys_.size(), sequential_y_when_init);
            // model_based insert
            int i = 0;
            size_t insert_pos = 0;
            for(auto data : keys_){
                insert_pos = model_.predict_next_idx(data.first);
                i ++;

                // find the gap to right of the predicted_pos
                while (expand_keys[insert_pos].first != 0){
                    insert_pos++;
                    // the alex paper has not mentioned that how they deal with the situation when the tailed partition
                        // are fully-packed, here we simply allocate the remained gaps after the last position
                    if (insert_pos == expand_keys.size()) {
                        size_t append_expand_size = (keys_.size() - i ) / density_;
                        size_t last_idx = expand_keys.size();
                        for(int j = 0; j < append_expand_size; j++){
                            expand_keys.emplace_back(std::make_pair(0, last_idx + j));
                        }
                    }
                }
                // std::cout<< i << ", " << data.first << ", " << insert_pos<< std::endl;

                expand_keys[insert_pos].first = data.first;
            }
            model_.second_model_size_ = expand_keys.size();
            keys_ = expand_keys;
        }
    };


    Node root_;

    size_t rmi_size_;
    size_t second_model_size_, leaf_split_size_;
    std::set<size_t> untrained_model_ids_;
    std::vector<int> model_trained_indicators_;
    bool complete_submodels_;
    bool compress_key_, find_near_seg_;
    double sample_rate_;
    double density_;
    size_t max_keys_;
    bool sequential_y_when_init_;


    Iterator first_key_iter_, last_key_iter_;
    size_t data_size_;

    // std::vector<std::vector<std::pair<KeyType, size_t>>> data_partitions_;
    // std::vector<LinearModel> last_layer_models_;
    // double L0_alpha_, L0_beta_;

    ALEX_RMILinearIndexer(Iterator first_key_iter, Iterator last_key_iter,
            size_t data_size, size_t second_model_size, size_t leaf_split_size, double density, size_t max_keys):
            root_(density, second_model_size){
        first_key_iter_ = first_key_iter;
        last_key_iter_ = last_key_iter;
        data_size_ = data_size;
        second_model_size_ = second_model_size;
        leaf_split_size_ = leaf_split_size;
        compress_key_ = false;
        density_ = density;
        max_keys_ = max_keys;
        sequential_y_when_init_ = true;

        // for(int i = 0; i < data_size; i++){
        //     std::vector<std::pair<KeyType, size_t>> tmp_data;
        //     data_partitions_.emplace_back(tmp_data);
        // }
        // for(int i = 0; i < second_model_size_; i++){
        //     LinearModel tmp_model;
        //     last_layer_models_.emplace_back(tmp_model);
        // }
    }


    // initialize a node
    void initialize(Node & node, size_t max_keys) {
       std::vector<std::vector<std::pair<KeyType, size_t>>> partitions;
       Iterator sampled_first_iter = node.keys_.begin();
       Iterator sampled_last_iter = node.keys_.end();
       size_t whole_data_size = node.keys_.size();

       std::cout<<"Training first layer model of node: "<< & node << std::endl;
       LinearModel model_0;
       size_t cur_node_second_model_size = node.model_.second_model_size_;

       model_0.train(sampled_first_iter, sampled_last_iter, false, cur_node_second_model_size, 
               whole_data_size, sequential_y_when_init_);
       node.model_ = model_0;

       for (auto i = 0; i < cur_node_second_model_size; i ++){
           std::vector<std::pair<KeyType, size_t>> tmp_parti;
           partitions.emplace_back(tmp_parti);
       }

       // get_partitions
       for (auto it = sampled_first_iter; it != sampled_last_iter; it++){
           double key = (double) (*it).first;
           double fp = linear(model_0.alpha_, model_0.beta_, key);
           size_t model_idx_predict = FCLAMP(fp, cur_node_second_model_size - 1);
           partitions[model_idx_predict].emplace_back(*it);
       }

       // iter the partitions
       for (int i = 0; i < partitions.size(); i++){
           auto partition = partitions[i];
           if (partition.size() > max_keys){  // inner node
               size_t child_num = node.child_nodes_.size();
               Node new_node(partition, node.density_, second_model_size_);
               node.child_nodes_.emplace_back(new_node);
               node.predictions_to_child_idx_[i] = child_num;
               initialize(node.child_nodes_[child_num], max_keys);
           }
           else{ // leaf node
               //merge multiple adjacent partitions
               std::vector<std::pair<KeyType, size_t>> merged_partition = partition;
               size_t child_num = node.child_nodes_.size();
               node.predictions_to_child_idx_[i] = child_num;
               size_t accumulated_size = partition.size();
               while (i < (partitions.size() - 1)){
                   i++;
                   auto next_partition = partitions[i];
                   if (accumulated_size + next_partition.size() < max_keys){
                       accumulated_size += next_partition.size();
                       node.predictions_to_child_idx_[i] = child_num;
                       merged_partition.insert(merged_partition.end(), next_partition.begin(), next_partition.end());
                   }
                   else{
                       i--;
                       break;
                   }
               }
               size_t merged_data_size = merged_partition.size();
               // second_model_size = merged_data_size, indicates the leaf node
               Node new_node(merged_partition, node.density_, merged_data_size);
               new_node.model_.train(merged_partition.begin(), merged_partition.end(), false, 0,
                       merged_data_size, sequential_y_when_init_);
               node.child_nodes_.emplace_back(new_node);
           }
       }
    }



    std::chrono::system_clock::time_point learn_index(Iterator first_iter, Iterator last_iter, float sample_rate = 1.0) {
        std::vector<std::pair<KeyType, size_t>> sampled_data;
        Iterator sampled_first_iter, sampled_last_iter;
        sample_rate_ = sample_rate;
        if (sample_rate != 1.0) {
            size_t sample_size = round(data_size_ * sample_rate);
            //std::sample(first_iter, last_iter, std::back_inserter(sampled_data), sample_size, std::mt19937{std::random_device{}()});
            std::experimental::sample(first_iter, last_iter, std::back_inserter(sampled_data), sample_size,
                                      std::mt19937{std::random_device{}()});
            std::sort(sampled_data.begin(), sampled_data.end());
        }else{
            std::vector<std::pair<KeyType, size_t>> data(first_iter, last_iter);
            sampled_data = data;
        }

        std::chrono::system_clock::time_point t0 = std::chrono::high_resolution_clock::now();

        root_.keys_ = sampled_data;
        root_.num_non_null_keys_ = sampled_data.size();
        root_.model_.second_model_size_ = ceil(double (sampled_data.size()) / max_keys_);

        initialize(root_, max_keys_);

        return t0;
    }

    inline KeyType  query(KeyType key) {
        // find leaf node
        size_t next_layer_index;
        Node & tmp_cur_node = root_;
        while (tmp_cur_node.child_nodes_.size() != 0){
            next_layer_index = tmp_cur_node.model_.predict_next_idx(key);
            auto child_idx = tmp_cur_node.predictions_to_child_idx_[next_layer_index];
            tmp_cur_node = tmp_cur_node.child_nodes_[child_idx];
        }
        next_layer_index = tmp_cur_node.model_.predict_next_idx(key);

        // search target key starting from predict_pos
        auto actual_pos_iter = exponential_search_check_gap(tmp_cur_node.keys_.begin(), tmp_cur_node.keys_.end(), key,
                                                            next_layer_index);
        // assert(key = (*actual_pos_iter).first);

        return key; // assume we do not access the real data bind of the key.
    }

    void insert(KeyType key){

        // find leaf node
        size_t next_layer_index;
        Node & tmp_cur_node = root_;
        while (tmp_cur_node.child_nodes_.size() != 0){
            next_layer_index = tmp_cur_node.model_.predict_next_idx(key);
            auto child_idx = tmp_cur_node.predictions_to_child_idx_[next_layer_index];
            tmp_cur_node = tmp_cur_node.child_nodes_[child_idx];
        }

        // split node for adaptive RMI, since the bound will be reached after inserting the current key
        if (tmp_cur_node.num_non_null_keys_ == max_keys_){
            std::vector<std::vector<std::pair<KeyType, size_t>>> partitions;
            for(int i = 0; i < leaf_split_size_; i++){
                std::vector<std::pair<KeyType, size_t>> tmp_parti;
                partitions.emplace_back(tmp_parti);
            }
            // get partitions uniformlly, according to the number of splitting leaf node
            size_t number_of_each_partition = 0;
            size_t cur_leaf = 0;
            size_t total_numbers_of_each_partition = ceil(double (tmp_cur_node.keys_.size() + 1) / leaf_split_size_);

            for (auto it = tmp_cur_node.keys_.begin(); it != tmp_cur_node.keys_.end(); it++){
                double tmp_key;
                if((*it).first < key and (*(it+1)).first > key) {
                    // the inserted key
                    tmp_key = (double) key;
                    double fp = linear(tmp_cur_node.model_.alpha_, tmp_cur_node.model_.beta_, tmp_key);
                    size_t model_idx_predict = FCLAMP(fp, second_model_size_ - 1);
                    tmp_cur_node.predictions_to_child_idx_[model_idx_predict] = cur_leaf;
                    partitions[cur_leaf].emplace_back(*it);
                    number_of_each_partition++;
                    if (number_of_each_partition == total_numbers_of_each_partition) {
                        number_of_each_partition = 0;
                        cur_leaf++;
                    }
                }
                tmp_key = (double) (*it).first;
                double fp = linear(tmp_cur_node.model_.alpha_, tmp_cur_node.model_.beta_, tmp_key);
                size_t model_idx_predict = FCLAMP(fp, second_model_size_ - 1);
                tmp_cur_node.predictions_to_child_idx_[model_idx_predict] = cur_leaf;
                partitions[cur_leaf].emplace_back(*it);
                number_of_each_partition ++;
                if (number_of_each_partition == total_numbers_of_each_partition){
                    number_of_each_partition = 0;
                    cur_leaf++;
                }
            }
            for(int i = 0; i < leaf_split_size_; i++){
                Node new_node(partitions[i], density_, partitions[i].size()); // second_model_size = keys.size, indicates the leaf node
                new_node.model_.train(partitions[i].begin(), partitions[i].end(), false, 0, partitions[i].size(), sequential_y_when_init_);
                tmp_cur_node.child_nodes_.emplace_back(new_node);
            }
            return;
        }

        // Gapped Array insert
        if (double(tmp_cur_node.num_non_null_keys_) / tmp_cur_node.keys_.size() >= tmp_cur_node.density_){
            tmp_cur_node.expand("GappedArray");
        }
        next_layer_index = tmp_cur_node.model_.predict_next_idx(key);
        // find non-zero low_bound and high_bound of the inserted pos
        size_t low_bound = (next_layer_index == 0) ? 0 : next_layer_index;
        size_t high_bound = (next_layer_index == (tmp_cur_node.keys_.size() - 1)) ? next_layer_index : next_layer_index + 1;
        while(tmp_cur_node.keys_[low_bound].first == 0 and low_bound != 0){
            low_bound --;
        }
        while(tmp_cur_node.keys_[high_bound].first == 0 and high_bound != (tmp_cur_node.keys_.size() - 1)){
            high_bound ++;
        }
        // correct the insert position, to maintain the order
        if (tmp_cur_node.keys_[low_bound].first > key or key > tmp_cur_node.keys_[high_bound].first){
            auto correct_pos_iter = exponential_search_check_gap(tmp_cur_node.keys_.begin(), tmp_cur_node.keys_.end(),
                                                                 key,
                                                                 next_layer_index);
            assert(std::distance(tmp_cur_node.keys_.begin(), correct_pos_iter) == (*correct_pos_iter).second);
            next_layer_index = (*correct_pos_iter).second; // the exponential_search_check_gap returns the lower_bound
        }
        // if the corrected_pos is occupied, make a gap, by shifting the elements by one position in the direction of the closest gap.
        if (tmp_cur_node.keys_[next_layer_index].first != 0){
            size_t shifted_pos = 1;
            while(1){
                // find the nearest gap, shifting
                size_t low_bound = (next_layer_index - shifted_pos <= 0) ? 0 : next_layer_index - shifted_pos;
                size_t high_bound = ((next_layer_index + shifted_pos) >= (tmp_cur_node.keys_.size() - 1)) ?
                        tmp_cur_node.keys_.size() - 1 : next_layer_index + shifted_pos;
                if (tmp_cur_node.keys_[high_bound].first == 0){
                    for(int j = high_bound; j > next_layer_index; j --){
                        tmp_cur_node.keys_[j].first = tmp_cur_node.keys_[j - 1].first;
                    }
                    break;
                } else if (tmp_cur_node.keys_[low_bound].first == 0){
                    for(int j = low_bound; j < next_layer_index; j ++){
                        tmp_cur_node.keys_[j].first = tmp_cur_node.keys_[j + 1].first;
                    }
                    break;
                } else{
                    shifted_pos ++;
                }
            }
        }
        tmp_cur_node.keys_[next_layer_index].first = key;
        tmp_cur_node.num_non_null_keys_++;
    }

    void evaluate_indexer_dynamic(int payload_size, bool write_to_file,
            double init_ratio, size_t reads_of_a_cycle, size_t writes_of_a_cycle){

        std::random_shuffle(first_key_iter_, last_key_iter_);
        size_t init_size = floor(init_ratio * data_size_);

        std::sort(first_key_iter_, first_key_iter_ + init_size);

        learn_index(first_key_iter_, first_key_iter_ + init_size);

        std::chrono::system_clock::time_point t0, t1;

        std::random_device dev;
        std::mt19937 rng(dev());
        size_t transaction = 0;
        size_t sampled_idx = 0;
        t0 = std::chrono::high_resolution_clock::now();
        //while (init_size < data_size_){
        while (init_size < init_size + 10000){
            //std::uniform_int_distribution<std::mt19937::result_type> dist(0,init_size);
            for(int i = 0; i < reads_of_a_cycle; i++, sampled_idx++){
                //sampled_idx = dist(rng);
                auto key = (*(first_key_iter_ + sampled_idx)).first;
                auto res = query(key);
            }
            size_t unwrited_keys_num = (data_size_ - init_size);
            size_t write_count = writes_of_a_cycle < unwrited_keys_num ? writes_of_a_cycle : unwrited_keys_num;
            for(int i = 0; i < write_count; i++){
                auto key = (*(first_key_iter_ + init_size + i)).first;
                insert(key);
            }
            init_size += write_count;
            transaction += (reads_of_a_cycle + write_count);
        }
        t1 = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t1 - t0);

        std::cout << "The throughput is " << (double (transaction) / time_span.count()) << " transactions per second."<< std::endl;

    }


    void evaluate_indexer(int payload_size, bool write_to_file) {
        std::vector<double> all_predicted_pos;
        auto t0 = std::chrono::high_resolution_clock::now();
        for (Iterator it = first_key_iter_; it != last_key_iter_; it++) {
            double pos = predict_position((*it).first);
            all_predicted_pos.emplace_back(pos);
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        int i = 0;
        size_t wrong_return_count = 0;
        for (Iterator it = first_key_iter_; it != last_key_iter_; it++, i++) {
            size_t corrected_res = correct_position(all_predicted_pos[i], (*it).first);
            if (corrected_res != i) {
                wrong_return_count ++;
            }
        }
        auto t2 = std::chrono::high_resolution_clock::now();
        //lookup-time per query
        auto predict_time = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count() / data_size_;
        auto correct_time = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() / data_size_;
        //size
        size_t size_of_payload = this->size_payloads_bytes(payload_size);

        // ML-oriented metric
        // regress_metric = ["RMSE", "MAE", "R2"]
        std::vector<size_t> all_true_pos, all_predicted_pos_without_metainfo;
        size_t zero_prediction_count = 0;
        for (size_t i = 0; i < data_size_; i++) {
            all_true_pos.push_back(i);
            size_t predicted_pos_without_metainfo = round(all_predicted_pos[i]);
            all_predicted_pos_without_metainfo.push_back(predicted_pos_without_metainfo);
            if (predicted_pos_without_metainfo == 0){
                zero_prediction_count ++;
            }
            //size_t delta = predicted_pos_without_metainfo > i ?
            //        predicted_pos_without_metainfo - i : i - predicted_pos_without_metainfo;
            //if  (delta > 200000 and predicted_pos_without_metainfo != 0){
            //    double pos = predict_position((*(first_key_iter_ + i))).first;
            //    double numric_error = pos - i;
            //}
        }

        std::cout << "predict time: " << predict_time << std::endl;
        std::cout << "correct time: " << correct_time << std::endl;
        std::cout << "overall lookup time: " << predict_time + correct_time << std::endl;
        std::cout << "size of payloads: " << size_of_payload << std::endl;
        std::cout << "wrong return count: " << wrong_return_count << std::endl;

        dlib::matrix<double, 1, 4> results;
        results = evaluate_regression_ML(all_predicted_pos_without_metainfo, all_true_pos, 10000, write_to_file);
        std::cout << "ML oriented matricx: " << results << std::endl;

    }


    void evaluate_indexer_analysis(int payload_size, bool write_to_file) {
        std::vector<double> all_predicted_pos;
        auto t0 = std::chrono::high_resolution_clock::now();
        for (Iterator it = first_key_iter_; it != last_key_iter_; it++) {
            double pos = predict_position(*it);
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
        size_t size_of_payload = this->size_payloads_bytes(payload_size);

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
                    double pos = predict_position(*(first_key_iter_ + i));
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

        std::vector<double> all_predicted_pos;
        auto t0 = std::chrono::high_resolution_clock::now();
        for (Iterator it = first_key_iter_; it != last_key_iter_; it++) {
            double pos = predict_position(*it);
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
        size_t size_of_payload = this->size_payloads_bytes(payload_size);

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



    double predict_position(KeyType key) {
        // find leaf node
        size_t next_layer_index;
        Node & tmp_cur_node = root_;
        while (tmp_cur_node.child_nodes_.size() != 0){
            next_layer_index = tmp_cur_node.model_.predict_next_idx(key);
            auto child_idx = tmp_cur_node.predictions_to_child_idx_[next_layer_index];
            tmp_cur_node = tmp_cur_node.child_nodes_[child_idx];
        }
        next_layer_index = tmp_cur_node.model_.predict_next_idx(key);

        return next_layer_index;
    }

    auto correct_position(float predicted_pos, KeyType key) {
        Iterator true_pos_iter = exponential_search_check_gap(first_key_iter_, last_key_iter_, key,
                                                              KeyType(round(predicted_pos)));
        return std::distance(this->first_key_iter_, true_pos_iter);

    };

/**
* @return the size in bytes of the data payloads.
*/
    size_t size_payloads_bytes(int payload_size) {
        size_t size_of_linear = sizeof(LinearModel);
        rmi_size_ = second_model_size_ * size_of_linear;
        //return size_in_bytes;
        return rmi_size_;
    };

};




#endif //LEARNED_INDEX_BTREE_INDEXER_HPP
