//
// Implementation of FITing-Tree.
//
// Galakatos A, Markovitch M, Binnig C, et al. Fiting-tree: A data-aware index structure[C]//Proceedings of the 2019
// International Conference on Management of Data. 2019: 1189-1206.
//


#ifndef LEARNED_INDEX_FITTINGTREEINDEXER_HPP
#define LEARNED_INDEX_FITTINGTREEINDEXER_HPP



#include <vector>
#include <list>
#include <chrono>
#include <cstddef>
#include <iterator>
#include <dlib/matrix.h>
#include <dlib/statistics.h>
#include <random>
#include <experimental/algorithm>
#include <cnpy.h>
#include "BtreeIndexer.hpp"
#include "Utilities.hpp"
#include "epsilon_meta_learner.hpp"



//using size_t = unsigned long;

/**
 * A struct that stores a segment.
 * @tparam KeyType the type of the elements that the segment indexes
 * @tparam Floating the floating-point type of the segment's parameters
 */
template<typename KeyType, typename Floating>
struct FittingSegment {
    //static_assert(std::is_floating_point<Floating>());
    KeyType seg_start;              ///< The first key that the segment indexes.
    Floating seg_slope;     ///< The slope of the segment.
    Floating seg_intercept;              ///< The intercept of the segment index, used for predict pos.

    FittingSegment() = default;

    /**
     * Constructs a new segment.
     * @param slope the slope of the segment
     * @param KeyType the start key of the segment
     * @param intercept the intercept of the lineal segment
     */
    FittingSegment(KeyType start, Floating slope, Floating intercept):
            seg_start(start), seg_slope(slope), seg_intercept(intercept){};

    friend inline bool operator<(const FittingSegment &s, const KeyType k) {
        return s.seg_start < k;
    }

    friend inline bool operator<(const FittingSegment &s1, const FittingSegment &s2) {
        return s1.seg_start < s2.seg_start;
    }

    /**
     * Returns the approximate position of the specified key.
     * @param k the key whose position must be approximated
     * @return the approximate position of the specified key
     */
    inline Floating operator()(KeyType k) const {
        Floating pos = seg_slope * double(k - seg_start) + seg_intercept;
        return pos > Floating(0) ? pos : 0.0;
    }
};


template<typename KeyType, typename Pos, typename Floating, typename Iterator>
class FittingTreeIndexer {
    using floating_type = Floating;
    using segment_type = FittingSegment<KeyType, Floating>;
    //using btree_const_iterator = typename stx::btree<KeyType, std::pair<float, size_t>, btreeFriend::btree_traits>::const_iterator;
    //using btree_const_iterator = stx::btree<size_t, std::pair<float, size_t>, std::pair<size_t, std::pair<float, size_t>>,std::less<KeyType>, btreeFriend::btree_traits>::const_iterator;

public:
    std::vector<segment_type> learned_segments_;
    //btreeFriend<stx::btree_default_map_traits<size_t, std::pair<Floating, size_t>>> btree_friend_;
    btreeFriend<stx::btree_map_traits<KeyType, std::pair<Floating, size_t>>> btree_friend_;
    std::map<KeyType, KeyType> segments_head_tail_;


    // used for meta-learner that changes epsilons for differenrt data
    double epsilon_; // the error bound used in the piece-wise segmentation
    int lookahead_n_; // the look forward number at most n steps
    std::vector<double> epsilons_; // the error bound used in the piece-wise segmentation
    int oracle_cur_epsilon_idx_; // test for oracle epsilon
    std::vector<double> mae_over_batches_; // the error bound used in the piece-wise segmentation

    // meta-learner
    EpsilonMetaLearner<KeyType, Iterator> meta_learner;

    FittingTreeIndexer(Iterator first_key_iter, Iterator last_key_iter, size_t data_size) :
            first_key_iter_(first_key_iter), last_key_iter_(last_key_iter), data_size_(data_size){
        data_indexed_ = std::vector<std::pair<key_type_transformed, size_t>> (0);
        ablate_meta_data_selector_ = false;
        ablate_meta_learner_ = false;
        batch_wise_meta_learner_ = false;
        lookahead_n_ = 10;
    };

    std::chrono::system_clock::time_point learn_index(const std::vector<std::pair<key_type_transformed, size_t>> & data,
            double error, double sample_rate = 1.0) {
        std::cout<< "[DEBUG for release mode] init tmp data, data size is " << data.size() << std::endl;
        data_indexed_ = data;
        data_size_ = data_indexed_.size();
        epsilon_ = error;
        epsilons_.emplace_back(epsilon_);
        if (data_size_ == 0){
            return std::chrono::high_resolution_clock::now();
        }
        std::cout<< "[DEBUG for release mode] memory address of passed_data, member data are " << (&data) << ", " <<
                (&data_indexed_) << std::endl;
        auto first_iter = data_indexed_.begin();
        auto last_iter = data_indexed_.end();
        std::cout<< "[DEBUG for release mode] init index member variables" <<std::endl;
        first_key_iter_ = data_indexed_.begin();
        last_key_iter_ = data_indexed_.end();
        std::vector<std::pair<KeyType, size_t>> sampled_data;
        std::chrono::system_clock::time_point t0;
        std::cout<< "[DEBUG for release mode] learn segs" <<std::endl;
        if (sample_rate != 1.0){
            size_t sample_size = round(data_indexed_.size() * sample_rate);
            std::experimental::sample(first_iter, last_iter, std::back_inserter(sampled_data), sample_size, std::mt19937{std::random_device{}()});
            std::sort(sampled_data.begin(), sampled_data.end());
            t0 = std::chrono::high_resolution_clock::now();
            learned_segments_ = learn_segments(sampled_data.begin(), sampled_data.end(), error);
        } else{
            t0 = std::chrono::high_resolution_clock::now();
            learned_segments_ = learn_segments(first_iter, last_iter, error);
        }
        std::cout<< "[DEBUG for release mode] organize" <<std::endl;
        organize_segments();
        std::cout<< "[DEBUG for release mode] organize end" <<std::endl;
        std::cout<< "[DEBUG for release mode] before return, input data size is "  << data.size() << std::endl;
        return t0;
    }

    std::chrono::system_clock::time_point learn_index_varied(const std::vector<std::pair<key_type_transformed, size_t>> & data,
                                                      double error, double sample_rate = 1.0) {
        std::cout<< "[DEBUG for release mode] init tmp data, data size is " << data.size() << std::endl;
        data_indexed_ = data;
        data_size_ = data_indexed_.size();
        // epsilon_ = error;
        // epsilons_.emplace_back(epsilon_);
        error = epsilons_[0];
        if (data_size_ == 0){
            return std::chrono::high_resolution_clock::now();
        }
        std::cout<< "[DEBUG for release mode] memory address of passed_data, member data are " << (&data) << ", " <<
                 (&data_indexed_) << std::endl;
        auto first_iter = data_indexed_.begin();
        auto last_iter = data_indexed_.end();
        std::cout<< "[DEBUG for release mode] init index member variables" <<std::endl;
        first_key_iter_ = data_indexed_.begin();
        last_key_iter_ = data_indexed_.end();
        std::vector<std::pair<KeyType, size_t>> sampled_data;
        std::chrono::system_clock::time_point t0;
        std::cout<< "[DEBUG for release mode] learn segs" <<std::endl;
        if (sample_rate != 1.0){
            size_t sample_size = round(data_indexed_.size() * sample_rate);
            std::experimental::sample(first_iter, last_iter, std::back_inserter(sampled_data), sample_size, std::mt19937{std::random_device{}()});
            std::sort(sampled_data.begin(), sampled_data.end());
            t0 = std::chrono::high_resolution_clock::now();
            learned_segments_ = learn_segments_varied(sampled_data.begin(), sampled_data.end(), error);
        } else{
            t0 = std::chrono::high_resolution_clock::now();
            learned_segments_ = learn_segments_varied(first_iter, last_iter, error);
        }
        std::cout<< "[DEBUG for release mode] organize" <<std::endl;
        organize_segments();
        std::cout<< "[DEBUG for release mode] organize end" <<std::endl;
        std::cout<< "[DEBUG for release mode] before return, input data size is "  << data.size() << std::endl;
        return t0;
    }


    /***
     * insert batched data in the end of index, inserted keys are all larger than current keys indexed.
     * @param first_iter
     * @param last_iter
     * @return
     */
    std::chrono::system_clock::time_point append_batch(const Iterator first_iter, const Iterator last_iter) {
        assert(std::is_sorted(first_iter, last_iter));
        assert((data_indexed_.size() == 0) or ((*first_iter).first > data_indexed_.back().first));
        std::chrono::system_clock::time_point t0;
        t0 = std::chrono::high_resolution_clock::now();
        for (auto it = first_iter; it != last_iter; it++) {
            data_indexed_.emplace_back((*it));
        }

        auto mae = learn_segments_with_MAE(first_iter, last_iter, epsilon_);
        mae_over_batches_.emplace_back(mae);
        epsilons_.emplace_back(epsilon_);

        auto temp_learned_segments = learn_segments(first_iter, last_iter, epsilon_);
        for (auto it = temp_learned_segments.begin(); it != temp_learned_segments.end(); it++) {
            btree_friend_.btree_map_storing_segs_of_fitting.insert(
                    (*it).seg_start, std::make_pair((*it).seg_slope, (*it).seg_intercept));
            learned_segments_.emplace_back(*it);
        }
        return t0;
    }

    void set_varied_epsilon(double min_epsilon, double max_epsilon, double add_rate, double mul_rate,
            std::string adjust_manner, double decay_threshold, double increase_threshold, int lookahead_n){
        min_epsilon_ = min_epsilon;
        max_epsilon_ = max_epsilon;
        add_rate_ = add_rate;
        mul_rate_ = mul_rate;
        adjust_manner_ = adjust_manner;
        decay_threshold_ = decay_threshold;
        increase_threshold_ = increase_threshold;
        lookahead_n_ = lookahead_n;
    }

    void print_varied_epsilon_setting(){
        std::cout << std::endl << "Init, min, max epsilon are: " << epsilon_ << ", " <<
                  min_epsilon_<<", "<<max_epsilon_<< std::endl;
        std::cout << "Adjust manner, add rate, mul rate are: " << adjust_manner_ << ", " <<
                  add_rate_ << ", " <<mul_rate_ << std::endl;
        std::cout << "Decay, increase threshold are: " << decay_threshold_ << ", " <<
                  increase_threshold_ << std::endl;
        std::cout <<"Ablation flags for meta-data selector and meta-learner are: " << ablate_meta_data_selector_ <<
            ", " << ablate_meta_learner_ << std::endl;
        std::cout <<"Batch-wise meta-learner flag is: " << batch_wise_meta_learner_ <<
            ", '0' indicates segment-wise meta-learner"<<std::endl;
        std::cout <<"Look-ahead number is: " << lookahead_n_ << std::endl;
    }

    /***
 * insert batched data in the end of index, inserted keys are all larger than current keys indexed.
 * the new segments will be learned with varied epsilon
 * @param first_iter
 * @param last_iter
 * @return
 */
    std::chrono::system_clock::time_point append_batch_varied_epsilon(
            const Iterator first_iter, const Iterator last_iter) {
        assert(std::is_sorted(first_iter, last_iter));
        assert((data_indexed_.size() == 0) or ((*first_iter).first > data_indexed_.back().first));
        std::chrono::system_clock::time_point t0;
        t0 = std::chrono::high_resolution_clock::now();
        for (auto it = first_iter; it != last_iter; it++) {
            data_indexed_.emplace_back((*it));
        }

        Iterator filtered_first_iter;
        if (ablate_meta_data_selector_){
            filtered_first_iter = first_iter;
        } else{
            filtered_first_iter = select_data_to_be_retain(first_iter, last_iter, epsilon_);
            if (filtered_first_iter == last_iter){
                // indicate that the data D_t (first_iter, last_iter) can be all covered by M_{t_1}
                // then we adopt M_t = M_{t_1}
                return t0;
            }
        }


        double adjusted_epsilon;
        std::vector<FittingSegment<KeyType, Floating>> temp_learned_segments;
        if (ablate_meta_learner_){
            // ablation study for the g function
            adjusted_epsilon = epsilon_;
            temp_learned_segments = learn_segments(filtered_first_iter, last_iter, epsilon_);
        } else {
            if (batch_wise_meta_learner_){
                auto estimated_mae_re_train = learn_segments_with_MAE(filtered_first_iter, last_iter, epsilon_);
                // estimate MAE, then adjust epsilon
                adjusted_epsilon = adjust_epsilon_by_mae(epsilon_, estimated_mae_re_train, min_epsilon_, max_epsilon_,
                                                         add_rate_, mul_rate_, adjust_manner_, decay_threshold_, increase_threshold_);
                epsilon_ = adjusted_epsilon;
                std::cout << "Change epsilon to : " << epsilon_ << std::endl;
                epsilons_.emplace_back(epsilon_);
                temp_learned_segments = learn_segments(filtered_first_iter, last_iter, epsilon_);
            } else{
                //Fine-grained segment-wise meta-learner
                temp_learned_segments = learn_segments_lookahead(filtered_first_iter, last_iter, epsilon_);
            }
        }


        for (auto it = temp_learned_segments.begin(); it != temp_learned_segments.end(); it++) {
            btree_friend_.btree_map_storing_segs_of_fitting.insert(
                    (*it).seg_start, std::make_pair((*it).seg_slope, (*it).seg_intercept));
            learned_segments_.emplace_back(*it);
        }

        auto mae = learn_segments_with_MAE(filtered_first_iter, last_iter, epsilon_);
        mae_over_batches_.emplace_back(mae);
        return t0;
    }


    /***
* insert batched data in the end of index, inserted keys are all larger than current keys indexed.
* the new segments will be learned with merged epsilon
* @param first_iter
* @param last_iter
* @return
*/
    std::chrono::system_clock::time_point append_batch_merged_epsilon(
            const Iterator first_iter, const Iterator last_iter,
            int batch_i, const std::vector<double> & last_epsilons, const std::vector<double> &last_maes) {
        assert(std::is_sorted(first_iter, last_iter));
        assert((data_indexed_.size() == 0) or ((*first_iter).first > data_indexed_.back().first));
        std::chrono::system_clock::time_point t0;
        t0 = std::chrono::high_resolution_clock::now();
        for (auto it = first_iter; it != last_iter; it++) {
            data_indexed_.emplace_back((*it));
        }

        // estimate MAE, then adjust epsilon
        double adjusted_epsilon = last_epsilons[0];
        // if (batch_i > 0){
        //     adjusted_epsilon = adjust_epsilon_by_mae_pair(
        //             last_epsilons[batch_i-1], last_epsilons[batch_i],
        //             last_maes[batch_i-1], last_maes[batch_i]);
        // }
        epsilon_ = adjusted_epsilon;
        std::cout << "Change epsilon to : " << epsilon_ << std::endl;
        epsilons_.emplace_back(epsilon_);

        auto temp_learned_segments = learn_segments(first_iter, last_iter, epsilon_);
        for (auto it = temp_learned_segments.begin(); it != temp_learned_segments.end(); it++) {
            btree_friend_.btree_map_storing_segs_of_fitting.insert(
                    (*it).seg_start, std::make_pair((*it).seg_slope, (*it).seg_intercept));
            learned_segments_.emplace_back(*it);
        }

        auto mae = learn_segments_with_MAE(first_iter, last_iter, epsilon_);
        mae_over_batches_.emplace_back(mae);
        return t0;
    }

    KeyType query(KeyType key) {
        auto predicted_pos = predict_position(key);
        KeyType res = correct_position(predicted_pos, key);
        return res;
    }

    void save_segments(std::string file_name) {
        unsigned int seg_count = learned_segments_.size();
        std::vector<KeyType> segs_begin_and_end_x;
        std::vector<size_t > segs_begin_and_end_y;
        segs_begin_and_end_x.reserve(seg_count * 2);
        segs_begin_and_end_y.reserve(seg_count * 2);
        std::vector<double > segs_slopes_and_intercept;
        segs_slopes_and_intercept.reserve(seg_count * 2);

        int i(0), j(1);
        std::vector<KeyType> seg_last_x;
        std::cout << "the number of segments is: "<< seg_count << std::endl;
        // found last key for each learned segment
        while (j < seg_count){
            auto next_seg_start = learned_segments_[j].seg_start;
            auto data_i = (*(data_indexed_.begin()+i)).first;
            if (next_seg_start != data_i){
                i ++;
            }
            else{
                auto data_last_i = (*(data_indexed_.begin()+i-1)).first;
                seg_last_x.emplace_back(data_last_i);
                j ++;
            }
        }
        seg_last_x.emplace_back(data_indexed_.back().first);


        j = 0;
        for (auto it = learned_segments_.begin(); it != learned_segments_.end(); it++) {
            segs_begin_and_end_x.emplace_back((*it).seg_start);
            segs_begin_and_end_x.emplace_back(seg_last_x[j]);
            segs_slopes_and_intercept.emplace_back((*it).seg_slope);
            segs_slopes_and_intercept.emplace_back((*it).seg_intercept);
            j++;
        }

        i = 0, j = 0;
        while (i < segs_begin_and_end_x.size() and j < data_indexed_.size()){
            if (isEqual(segs_begin_and_end_x[i], data_indexed_[j].first)){
                segs_begin_and_end_y.emplace_back(data_indexed_[j].second);
                i++;
            }
            j++;
        }
        if ((segs_begin_and_end_x.size() - segs_begin_and_end_y.size()) == 1) {
            std::cout<< "append last y" << std::endl;
            segs_begin_and_end_y.emplace_back(data_indexed_.back().second);
        }

        std::cout<< "write segments to file:" << file_name << std::endl;
        cnpy::npy_save(file_name + "_begin_end_x.npy", &segs_begin_and_end_x[0], {seg_count, 2}, "w");
        cnpy::npy_save(file_name + "_begin_end_y.npy", &segs_begin_and_end_y[0], {seg_count, 2}, "w");
        cnpy::npy_save(file_name + "_slope_inter.npy", &segs_slopes_and_intercept[0],{seg_count, 2},"w");

        // std::ofstream outFile;
        // std::cout<< "write segments to file:" << file_name << std::endl;
        // outFile.open(file_name, std::ios::out);
        // i = 0;
        // for (auto it = learned_segments_.begin(); it != learned_segments_.end(); it++) {
        //     std::stringstream stream;
        //     stream << std::fixed << std::setw(20) << std::setprecision(6) << (*it).seg_start
        //         << "," << (*it).seg_slope << "," << (*it).seg_intercept << "," <<  seg_last_x[i] << std::endl;
        //     i++ ;
        //     outFile << stream.str();
        // }
    }

    void evaluate_indexer(int payload_size, std::string pred_file_name="", bool shuffle=true) {
        std::vector<std::pair<key_type_transformed, size_t>> tmp_data(data_indexed_.begin(), data_indexed_.end());
        if (shuffle == true){
            std::srand(1234);
            random_shuffle(tmp_data.begin(), tmp_data.end());
        }

        std::vector<Pos> all_predicted_pos;
        auto t0 = std::chrono::high_resolution_clock::now();
        int i = 0;
        for(Iterator it = tmp_data.begin(); it != tmp_data.end(); it++, i++){
            Pos predicted_pos = predict_position((*it).first);
            all_predicted_pos.emplace_back(predicted_pos);
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        i = 0;
        size_t wrong_return_count = 0;
        for (Iterator it = tmp_data.begin(); it != tmp_data.end(); it++, i++) {
            size_t corrected_res = correct_position(all_predicted_pos[i], (*it).first);
            if (corrected_res != (*it).second) {
                wrong_return_count ++;
#ifdef Debug
                Pos predicted_pos = predict_position((*it).first);
                size_t corrected_res = correct_position(all_predicted_pos[i], (*it).first);
#endif
            }
        }

        auto t2 = std::chrono::high_resolution_clock::now();
        //lookup-time per query
        auto predict_time = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count() / data_indexed_.size();
        auto correct_time = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() / data_indexed_.size();
        //size
        size_t segments_count = learned_segments_.size();
        size_t size_of_payload = this->size_payloads_bytes(payload_size);

        // ML-oriented metric
        // regress_metric = ["RMSE", "MAE", "R2"]
        std::vector<size_t> all_true_pos, all_predicted_pos_without_metainfo;
        for (size_t i=0; i < data_indexed_.size(); i++){
            //all_true_pos.push_back(i);
            all_true_pos.push_back((*(tmp_data.begin() + i)).second);
            //KeyType key = *(data_indexed_.begin()() + i);
            KeyType key = (*(tmp_data.begin() + i)).first;
            //size_t predi_pos = round(all_predicted_pos[i](key));
            size_t predi_pos = all_predicted_pos[i].pos;
            size_t pos_delta = predi_pos >= i ? (predi_pos - i) : (i - predi_pos);
            if (pos_delta > epsilon_){
                Pos err_predicted_pos = predict_position((*(tmp_data.begin()+i)).first);
            }
            //assert(pos_delta <= epsilon_); // on down-sampling situation, this assert will fail.
            all_predicted_pos_without_metainfo.push_back(predi_pos);
        }
        std::cout<< "predict time: " << predict_time << std::endl;
        std::cout<< "correct time: " << correct_time << std::endl;
        std::cout<< "overall query time: " << predict_time + correct_time << std::endl;
        std::cout << "the number of all segments: " << segments_count << std::endl;
        std::cout<< "size of payloads: " << size_of_payload<< std::endl;
        std::cout << "wrong return count: " << wrong_return_count << std::endl;
        dlib::matrix<double, 1, 4> results;
        bool write_to_file = (pred_file_name == "") ? false : true;
        results = evaluate_regression_ML(all_predicted_pos_without_metainfo, all_true_pos,
                0, write_to_file, pred_file_name);
        std::cout<< "ML oriented matricx: " << results;
    }



    void evaluate_indexer_split() {
        std::vector<Pos> all_predicted_pos;
        auto t0 = std::chrono::high_resolution_clock::now();
        int i = 0;
        for(Iterator it = data_indexed_.begin(); it != data_indexed_.end(); it++, i++){
            //Pos predicted_pos = predict_position(*it);
            Pos predicted_pos = predict_position((*it).first);
            all_predicted_pos.emplace_back(predicted_pos);
            //size_t delta = predicted_pos.pos > i ? predicted_pos.pos - i : i - predicted_pos.pos;
            //if (delta > 1000000){
            //    Pos err_predicted_pos = predict_position((*it).first);
            //}
        }
        auto t1 = std::chrono::high_resolution_clock::now();

        // ML-oriented metric
        // regress_metric = ["RMSE", "MAE", "R2"]
        std::vector<size_t> all_true_pos_within, all_true_pos_outof_seg, all_predicted_pos_without_metainfo_within_seg,
            all_predicted_pos_without_metainfo_outof_seg;
        dlib::matrix<double, 1, 4> results;
        for (size_t i=0; i < data_indexed_.size(); i++){
            KeyType key = (*(data_indexed_.begin() + i)).first;
            if (check_within_segment(key)){
                all_true_pos_within.push_back((*(data_indexed_.begin() + i)).second);
                size_t predi_pos = all_predicted_pos[i].pos;
                size_t pos_delta = predi_pos >= i ? (predi_pos - i) : (i - predi_pos);
                if (pos_delta > epsilon_){
                    Pos err_predicted_pos = predict_position((*(data_indexed_.begin()+i)).first);
                }
                all_predicted_pos_without_metainfo_within_seg.push_back(predi_pos);
            }else{
                all_true_pos_outof_seg.push_back((*(data_indexed_.begin() + i)).second);
                size_t predi_pos = all_predicted_pos[i].pos;
                size_t pos_delta = predi_pos >= i ? (predi_pos - i) : (i - predi_pos);
                if (pos_delta > epsilon_){
                    Pos err_predicted_pos = predict_position((*(data_indexed_.begin()+i)).first);
                }
                all_predicted_pos_without_metainfo_outof_seg.push_back(predi_pos);
            }
        }
        std::cout<< "The size of within and outof are: " << all_true_pos_within.size() << ", " << all_true_pos_outof_seg.size() << std::endl;
        results = evaluate_regression_ML(all_predicted_pos_without_metainfo_within_seg, all_true_pos_within, epsilon_);
        std::cout<< "ML oriented matricx for keys within segments: " << results<< std::endl;
        results = evaluate_regression_ML(all_predicted_pos_without_metainfo_outof_seg, all_true_pos_outof_seg, epsilon_);
        std::cout<< "ML oriented matricx for keys out of segments:: " << results<< std::endl;

        std::map<size_t, size_t> counter_for_predictions;
        all_predicted_pos_without_metainfo_within_seg.insert(all_predicted_pos_without_metainfo_within_seg.end(),
                all_predicted_pos_without_metainfo_outof_seg.begin(), all_predicted_pos_without_metainfo_outof_seg.end());
        for (int i = 0; i < all_predicted_pos_without_metainfo_within_seg.size(); ++i) {
            auto pred = all_predicted_pos_without_metainfo_within_seg[i];
            if (counter_for_predictions.count(pred) == 0) {
                counter_for_predictions[pred] = 1;
            }else{
                counter_for_predictions[pred] += 1;
            }
        }
        size_t duplicate_count = 0;
        auto counter_it = counter_for_predictions.begin();
        while(counter_it != counter_for_predictions.end()){
            size_t count_num = counter_it -> second;
            if (count_num > 1){
                duplicate_count ++;
            }
            counter_it++;
        }
        std::cout<<duplicate_count<<std::endl;
    }

    std::vector<double> get_segment_stats() {
        std::vector<double> seg_stats;
        std::map<KeyType, double> seg_cover_counter;
        std::vector<Pos> all_predicted_pos;
        int i = 0;
        for(Iterator it = data_indexed_.begin(); it != data_indexed_.end(); it++, i++){
            KeyType seg_key = find_seg_key((*it).first);
            if (seg_cover_counter.count(seg_key) == 0) {
                seg_cover_counter[seg_key] = 1.0;
            }else{
                seg_cover_counter[seg_key] += 1.0;
            }
        }
        for (auto i = seg_cover_counter.begin(); i != seg_cover_counter.end(); i++) {
            seg_stats.emplace_back(i->second);
        }
        return seg_stats;
    }




    /**
    * Returns the size in bytes of the payload w.r.t. the data size restored in the  segments.
    * @return the size in bytes of the data payloads.
    */
    size_t size_payloads_bytes(int payload_size) const {
       auto btree_stats = btree_friend_.btree_map_storing_segs_of_fitting.get_stats();
       size_t size_in_bytes = sizeof(KeyType) * btree_stats.innernodes * btree_stats.innerslots
                              + sizeof(FittingSegment<size_t, float>)  * btree_stats.leaves * btree_stats.leafslots;
       //size_t size_in_bytes = sizeof(KeyType) * btree_stats.innernodes * btree_stats.innerslots
               //+ (sizeof(FittingSegment<size_t, float>) + payload_size) * btree_stats.leaves * btree_stats.leafslots;
       return size_in_bytes;
    }


    inline Pos predict_position(KeyType key) const;

    bool ablate_meta_data_selector_; // ablation study for the selection f() function
    bool ablate_meta_learner_;  // ablation study for the update g() function
    bool batch_wise_meta_learner_; //batch-wise or segment-wise meta-learner

protected:
    size_t data_size_;    ///< The number of elements in the data.
    Iterator& first_key_iter_;    ///< The iterator of the smallest element in the data.
    Iterator& last_key_iter_;    ///< The iterator of the largest element in the data.
    std::vector<std::pair<key_type_transformed, size_t>> data_indexed_;   ///< The stored real data to be indexed.

    double min_epsilon_; ///< lower bound in the dynamic epsilon adjustment
    double max_epsilon_; ///< upper bound in the dynamic epsilon adjustment
    double add_rate_; ///< decay/increase rate in the dynamic epsilon adjustment, ADD manner
    double mul_rate_; ///< decay/increase rate in the dynamic epsilon adjustment, MUL manner
    std::string adjust_manner_; ///< decay/increase manner in the dynamic epsilon adjustment, linear/exp
    double decay_threshold_; ///< decay threshold in the dynamic epsilon adjustment
    double increase_threshold_; ///< increase threshold in the dynamic epsilon adjustment




    std::vector<segment_type> learn_segments(Iterator first_iter, Iterator last_iter, double error);
    std::vector<segment_type> learn_segments_lookahead(Iterator first_iter, Iterator last_iter, double epsilon);
    std::vector<segment_type> learn_segments_varied(Iterator first_iter, Iterator last_iter, double epsilon,
                                                    bool hard_encoded_epsilon=true);

    double learn_segments_with_MAE(Iterator first_iter, Iterator last_iter, double epsilon);
    void organize_segments();

    inline bool check_within_segment(KeyType key);
    KeyType correct_position(Pos predicted_pos, KeyType key);

    KeyType find_seg_key(KeyType key) const;

    Iterator select_data_to_be_retain(Iterator first_iter, Iterator last_iter, double epsilon);
};




/**
 * Learn the index of query. In FittingTree indexer, the index are segments including starting keys and slopes
 * @tparam KeyType
 * @tparam Pos
 * @tparam Floating
 * @tparam Iterator
 * @param first_iter
 * @param last_iter
 * @param err
 * @return
 */
template<typename KeyType, typename Pos, typename Floating, typename Iterator>
std::vector<FittingSegment<KeyType, Floating>>
        FittingTreeIndexer<KeyType, Pos, Floating, Iterator>::learn_segments(
                Iterator first_iter, Iterator last_iter, double err) {
    std::vector<segment_type> segments;
    segments.reserve(8192);

    size_t input_data_size = std::distance(first_iter, last_iter);
    std::cout<< "Input data size: " << input_data_size << std::endl;
    if (input_data_size == 0)
        return segments;
    assert(std::is_sorted(first_iter, last_iter));
    this->epsilon_ = err;

    double_t slope_high = std::numeric_limits<double>::max();
    double_t slope_low = 0.0;
    KeyType segment_start = (*first_iter).first;
    size_t base_intercept = learned_segments_.size() == 0 ? 0 : data_indexed_.size()-input_data_size; // online learning version
    size_t intercept = base_intercept;
    Iterator segment_start_iter = first_iter;
    Iterator segment_end_iter = first_iter;
    for (Iterator it = first_iter; it != last_iter; it++){
        // size_t debug_i = std::distance(first_iter, it);
        //double_t delta_y = std::distance(segment_start_iter, it);
        double_t delta_y = (*it).second - (*segment_start_iter).second;
        double_t delta_x = (*it).first - (*segment_start_iter).first;
        //double_t delta_x = *it - segment_start;
        double_t slope = (delta_x == 0) ? 0.0 : (delta_y / delta_x);
        //std::cout<< "[DEBUG for release mode]" <<std::endl;
        if (slope <= slope_high and slope >= slope_low){
            if (delta_x == 0) continue; // the first point in the new segment
            double_t max_slope = (delta_y + err) / delta_x;
            slope_high = std::min(slope_high, max_slope);
            double_t min_slope =  delta_y >= err ? (delta_y - err) / delta_x : 0;
            slope_low = std::max(slope_low, min_slope);
            //size_t predi_pos = round((0.5 * (slope_high + slope_low))* float(*it - segment_start) + float(intercept));
            //size_t pos_delta = predi_pos >= debug_i ? (predi_pos - debug_i) : (debug_i - predi_pos);
            //assert(pos_delta < 128);
        }
        else{
            assert(slope_high >= slope_low);
            //size_t predi_pos = round((0.5 * (slope_high + slope_low))* float(*it - segment_start) + float(intercept));
            //size_t pos_delta = predi_pos >= debug_i ? (predi_pos - debug_i) : (debug_i - predi_pos);
            //assert(pos_delta < 128);
            segment_end_iter = it == first_iter ? first_iter:(it - 1);
            KeyType seg_last = (*segment_end_iter).first;
            if (segments_head_tail_.count(segment_start) == 0){
                segments_head_tail_[segment_start] = seg_last;
            }
            assert(intercept <= input_data_size + base_intercept);
#ifdef Debug
            if (intercept == 4628678){
                int i = 0;
            }
#endif
            segments.emplace_back(segment_start, 0.5 * (slope_high + slope_low), intercept);
            //segment_start = *it;
            segment_start = (*it).first;
            //intercept = std::distance(first_iter, it);
            intercept = (*it).second - (*first_iter).second + base_intercept;
            segment_start_iter = it;
            slope_high = std::numeric_limits<double>::max();
            slope_low = 0.0;
            if (it == (last_iter - 1)){
                // the corner case: the last point is a single segment
                slope_high = 0.0;
            }
        }
    }
    // the last segment
    assert(intercept <= input_data_size + base_intercept);
#ifdef Debug
    if (intercept == 4628678){
        int i = 0;
    }
#endif

    //std::cout<< "[DEBUG for release mode] last segment" <<std::endl;
    segments.emplace_back(segment_start, 0.5 * (slope_high + slope_low), intercept);


    return segments;
}




/**
 * Learn the index of query, epsilon is varied
 * @tparam KeyType
 * @tparam Pos
 * @tparam Floating
 * @tparam Iterator
 * @param first_iter
 * @param last_iter
 * @param epsilon
 * @return
 */
template<typename KeyType, typename Pos, typename Floating, typename Iterator>
std::vector<FittingSegment<KeyType, Floating>>
FittingTreeIndexer<KeyType, Pos, Floating, Iterator>::learn_segments_varied(
        Iterator first_iter, Iterator last_iter, double epsilon, bool hard_encoded_epsilon) {
    std::vector<segment_type> segments;
    segments.reserve(8192);

    size_t input_data_size = std::distance(first_iter, last_iter);
    std::cout<< "Input data size: " << input_data_size << std::endl;
    if (input_data_size == 0)
        return segments;
    assert(std::is_sorted(first_iter, last_iter));
    //this->epsilon_ = epsilon;
    epsilon = epsilons_[oracle_cur_epsilon_idx_];
    double_t slope_high = std::numeric_limits<double>::max();
    double_t slope_low = 0.0;
    KeyType segment_start = (*first_iter).first;
    size_t base_intercept = learned_segments_.size() == 0 ? 0 : data_indexed_.size()-input_data_size; // online learning version
    size_t intercept = base_intercept;
    Iterator segment_start_iter = first_iter;
    Iterator segment_end_iter = first_iter;
    for (Iterator it = first_iter; it != last_iter; it++){
        // size_t debug_i = std::distance(first_iter, it);
        //double_t delta_y = std::distance(segment_start_iter, it);
        double_t delta_y = (*it).second - (*segment_start_iter).second;
        double_t delta_x = (*it).first - (*segment_start_iter).first;
        //double_t delta_x = *it - segment_start;
        double_t slope = (delta_x == 0) ? 0.0 : (delta_y / delta_x);
        //std::cout<< "[DEBUG for release mode]" <<std::endl;
        if (slope <= slope_high and slope >= slope_low){
            if (delta_x == 0) continue; // the first point in the new segment
            double_t max_slope = (delta_y + epsilon) / delta_x;
            slope_high = std::min(slope_high, max_slope);
            double_t min_slope = delta_y >= epsilon ? (delta_y - epsilon) / delta_x : 0;
            slope_low = std::max(slope_low, min_slope);
            //size_t predi_pos = round((0.5 * (slope_high + slope_low))* float(*it - segment_start) + float(intercept));
            //size_t pos_delta = predi_pos >= debug_i ? (predi_pos - debug_i) : (debug_i - predi_pos);
            //assert(pos_delta < 128);
        }
        else{
            assert(slope_high >= slope_low);
            //size_t predi_pos = round((0.5 * (slope_high + slope_low))* float(*it - segment_start) + float(intercept));
            //size_t pos_delta = predi_pos >= debug_i ? (predi_pos - debug_i) : (debug_i - predi_pos);
            //assert(pos_delta < 128);
            segment_end_iter = it == first_iter ? first_iter:(it - 1);
            KeyType seg_last = (*segment_end_iter).first;
            if (segments_head_tail_.count(segment_start) == 0){
                segments_head_tail_[segment_start] = seg_last;
            }
            assert(intercept <= input_data_size + base_intercept);
#ifdef Debug
            if (intercept == 4628678){
                int i = 0;
            }
#endif
            segments.emplace_back(segment_start, 0.5 * (slope_high + slope_low), intercept);
            //segment_start = *it;
            segment_start = (*it).first;
            //intercept = std::distance(first_iter, it);
            intercept = (*it).second - (*first_iter).second + base_intercept;
            segment_start_iter = it;
            slope_high = std::numeric_limits<double>::max();
            slope_low = 0.0;
            if (it == (last_iter - 1)){
                // the corner case: the last point is a single segment
                slope_high = 0.0;
            }
            std::cout<<"oracle_cur_idx and epsilons_size are:" << oracle_cur_epsilon_idx_ << ", " << epsilons_.size()-1
                <<std::endl;
            if (oracle_cur_epsilon_idx_ < epsilons_.size()-1){
                oracle_cur_epsilon_idx_ ++;
                epsilon = epsilons_[oracle_cur_epsilon_idx_];
                std::cout<<"Change epsilon to:" << epsilon <<std::endl;
            }
        }
    }
    // the last segment
    assert(intercept <= input_data_size + base_intercept);
#ifdef Debug
    if (intercept == 4628678){
        int i = 0;
    }
#endif

    //std::cout<< "[DEBUG for release mode] last segment" <<std::endl;
    segments.emplace_back(segment_start, 0.5 * (slope_high + slope_low), intercept);


    return segments;
}




/**
 * Learn index with segment-level epsilon varing. In FittingTree indexer, the index are segments including starting keys and slopes
 * @tparam KeyType
 * @tparam Pos
 * @tparam Floating
 * @tparam Iterator
 * @param first_iter
 * @param last_iter
 * @param epsilon
 * @return
 */
template<typename KeyType, typename Pos, typename Floating, typename Iterator>
std::vector<FittingSegment<KeyType, Floating>>
FittingTreeIndexer<KeyType, Pos, Floating, Iterator>::learn_segments_lookahead(
        Iterator first_iter, Iterator last_iter, double epsilon) {
    std::vector<segment_type> segments;
    segments.reserve(8192);

    size_t input_data_size = std::distance(first_iter, last_iter);
    std::cout<< "Input data size: " << input_data_size << std::endl;
    if (input_data_size == 0)
        return segments;
    assert(std::is_sorted(first_iter, last_iter));
    epsilons_.emplace_back(epsilon);

    double_t slope_high = std::numeric_limits<double>::max();
    double_t slope_low = 0.0;
    KeyType segment_start = (*first_iter).first;
    size_t base_intercept = learned_segments_.size() == 0 ? 0 : data_indexed_.size()-input_data_size; // online learning version
    size_t intercept = base_intercept;
    Iterator segment_start_iter = first_iter;
    Iterator segment_end_iter = first_iter;
    for (Iterator it = first_iter; it != last_iter; it++){
        double_t delta_y = (*it).second - (*segment_start_iter).second;
        double_t delta_x = (*it).first - (*segment_start_iter).first;
        double_t slope = (delta_x == 0) ? 0.0 : (delta_y / delta_x);
        if (slope <= slope_high and slope >= slope_low){
            if (delta_x == 0) continue; // the first point in the new segment
            double_t max_slope = (delta_y + epsilon) / delta_x;
            slope_high = std::min(slope_high, max_slope);
            double_t min_slope = delta_y >= epsilon ? (delta_y - epsilon) / delta_x : 0;
            slope_low = std::max(slope_low, min_slope);
        }
        else {
            segment_type cur_seg = segment_type (segment_start, 0.5 * (slope_high + slope_low), intercept);

            auto adjust_epsilon = epsilon;
            if ((it+1) != last_iter and lookahead_n_ > 1){
                // Segment-level epsilon adjustment using a lookahead algorithm
                // when face the breaking point, lookahead at most n points to vary epsilon according to the slope trend
                Iterator lookahead_begin_iter = it + 1, lookahead_end_iter = it + 1;
                int i = 0;
                while (i < lookahead_n_ and lookahead_end_iter != last_iter){
                    i++;
                    lookahead_end_iter ++;
                }
                // double breaking_error = std::abs(cur_seg((*it).first) - (*it).second);
                // int i = 0;
                // double ave_lookahead_error = 0.0; // the average error of next lookahead_n_ keys
                // while (i < lookahead_n_ and lookahead_begin_iter != last_iter){
                //     auto lookahead_key = (*lookahead_begin_iter).first;
                //     auto lookahead_pos = (*lookahead_begin_iter).second;
                //     ave_lookahead_error += std::abs(cur_seg(lookahead_key) - lookahead_pos);
                //     i++;
                //     lookahead_begin_iter ++;
                // }
                //ave_lookahead_error = ave_lookahead_error / i;
                auto lookhead_mae = learn_segments_with_MAE(lookahead_begin_iter, lookahead_end_iter, epsilon);
                adjust_epsilon = adjust_epsilon_by_mae(epsilon, lookhead_mae, min_epsilon_, max_epsilon_,
                                      add_rate_, mul_rate_, adjust_manner_, decay_threshold_, increase_threshold_);
            }
            if (not isEqual(adjust_epsilon, epsilon)){
                epsilon = adjust_epsilon;
                epsilon_ = epsilon;
                epsilons_.emplace_back(epsilon);
            }

            assert(slope_high >= slope_low);
            segment_end_iter = it == first_iter ? first_iter:(it - 1);
            KeyType seg_last = (*segment_end_iter).first;
            if (segments_head_tail_.count(segment_start) == 0){
                segments_head_tail_[segment_start] = seg_last;
            }
            assert(intercept <= input_data_size + base_intercept);
#ifdef Debug
            if (intercept == 4628678){
                int i = 0;
            }
#endif
            segments.emplace_back(cur_seg);
            segment_start = (*it).first;
            intercept = (*it).second - (*first_iter).second + base_intercept;
            segment_start_iter = it;
            slope_high = std::numeric_limits<double>::max();
            slope_low = 0.0;
            if (it == (last_iter - 1)){
                // the corner case: the last point is a single segment
                slope_high = 0.0;
            }
        }
    }
    // the last segment
    assert(intercept <= input_data_size + base_intercept);
#ifdef Debug
    if (intercept == 4628678){
        int i = 0;
    }
#endif

    //std::cout<< "[DEBUG for release mode] last segment" <<std::endl;
    segments.emplace_back(segment_start, 0.5 * (slope_high + slope_low), intercept);


    return segments;
}



/**
 * Filter out data that no need to retain
 */
template<typename KeyType, typename Pos, typename Floating, typename Iterator>
Iterator FittingTreeIndexer<KeyType, Pos, Floating, Iterator>::select_data_to_be_retain
        (Iterator first_iter, Iterator last_iter, double epsilon) {
    if (learned_segments_.size() == 0){
        return first_iter;
    }
    Iterator cur_iter = first_iter;
    while (cur_iter != last_iter){
        auto tmp_key = (*cur_iter).first;
        auto tmp_pos = (*cur_iter).second;
        auto tmp_segment = learned_segments_.back();
        if (std::abs(tmp_segment(tmp_key) - tmp_pos) > epsilon){
            break;
        }
        cur_iter ++;
    }
    return cur_iter;
}


/**
 * Learn segments for a given data, meanwhile evaluate and return MAE
 * This function is used to estimate the gap between MAE and predefined error bound,
 * to provide hints for adaptively adjusting the error bound
 */
template<typename KeyType, typename Pos, typename Floating, typename Iterator>
double FittingTreeIndexer<KeyType, Pos, Floating, Iterator>::learn_segments_with_MAE
    (Iterator first_iter, Iterator last_iter, double epsilon) {
    segment_type tmp_segment;
    double total_absolute_error = 0.0;

    size_t input_data_size = std::distance(first_iter, last_iter);
    // std::cout<< "Estimated data size: " << input_data_size << std::endl;
    if (input_data_size == 0)
        return 0.0;
    if (input_data_size == 1)
        return 1.0;
    assert(std::is_sorted(first_iter, last_iter));

    double_t slope_high = std::numeric_limits<double>::max();
    double_t slope_low = 0.0;
    KeyType segment_start = (*first_iter).first;
    size_t base_intercept = learned_segments_.size() == 0 ? 0 : data_indexed_.size()-input_data_size; // online learning version
    size_t intercept = base_intercept;
    Iterator segment_start_iter = first_iter;
    Iterator segment_end_iter = first_iter;
    for (Iterator it = first_iter; it != last_iter; it++){
        double_t delta_y = (*it).second - (*segment_start_iter).second;
        double_t delta_x = (*it).first - (*segment_start_iter).first;
        double_t slope = (delta_x == 0) ? 0.0 : (delta_y / delta_x);
        if (slope <= slope_high and slope >= slope_low){
            if (delta_x == 0) continue; // the first point in the new segment
            double_t max_slope = (delta_y + epsilon) / delta_x;
            slope_high = std::min(slope_high, max_slope);
            double_t min_slope = delta_y >= epsilon ? (delta_y - epsilon) / delta_x : 0;
            slope_low = std::max(slope_low, min_slope);
        }
        else {
            assert(slope_high >= slope_low);
            segment_end_iter = it == first_iter ? first_iter:(it - 1);
            KeyType seg_last = (*segment_end_iter).first;
            if (segments_head_tail_.count(segment_start) == 0){
                segments_head_tail_[segment_start] = seg_last;
            }
            assert(intercept <= input_data_size + base_intercept);
#ifdef Debug
            if (intercept == 4628678){
                int i = 0;
            }
#endif
            tmp_segment = segment_type(segment_start, 0.5 * (slope_high + slope_low), intercept);

            // calculate the total absolute error for existing segment
            for (Iterator sub_it = segment_start_iter; sub_it != it; sub_it++){
                auto tmp_key = (*sub_it).first;
                auto tmp_pos = (*sub_it).second;
                total_absolute_error += abs(tmp_segment(tmp_key) - tmp_pos);
            }

            segment_start = (*it).first;
            intercept = (*it).second - (*first_iter).second + base_intercept;
            segment_start_iter = it;
            slope_high = std::numeric_limits<double>::max();
            slope_low = 0.0;
            if (it == (last_iter - 1)){
                slope_high = 0.0;
            }
        }
    }
    // the last segment
    assert(intercept <= input_data_size + base_intercept);
#ifdef Debug
    if (intercept == 4628678){
        int i = 0;
    }
#endif
    tmp_segment = segment_type(segment_start, 0.5 * (slope_high + slope_low), intercept);
    for (Iterator sub_it = segment_start_iter; sub_it != last_iter; sub_it++){
        auto tmp_key = (*sub_it).first;
        auto tmp_pos = (*sub_it).second;
        total_absolute_error += abs(tmp_segment(tmp_key) - tmp_pos);
    }

    return total_absolute_error / input_data_size;
}


template<typename KeyType, typename Pos, typename Floating, typename Iterator>
void FittingTreeIndexer<KeyType, Pos, Floating, Iterator>::organize_segments() {
    std::vector<std::pair<KeyType, std::pair<Floating, size_t>>> pairs;
    size_t i = 0;
    for (auto it = learned_segments_.begin(); it != learned_segments_.end(); it++, i++) {
        pairs.emplace_back((*it).seg_start, std::make_pair((*it).seg_slope, (*it).seg_intercept));
    }
    //stx::btree_map<KeyType, std::pair<Floating, size_t>, std::less<KeyType>, stx::btree_default_map_traits<size_t, std::pair<Floating, size_t>>>
    stx::btree_map<KeyType, std::pair<Floating, Floating>, std::less<KeyType>, stx::btree_map_traits<KeyType, std::pair<Floating, Floating>>>
            constructed_btree(pairs.begin(), pairs.end());
    btree_friend_.btree_map_storing_segs_of_fitting = constructed_btree;

}



template<typename KeyType, typename Pos, typename Floating, typename Iterator>
inline KeyType FittingTreeIndexer<KeyType, Pos, Floating, Iterator>::find_seg_key(KeyType key) const {
    auto seg_leaf_iter = btree_friend_.btree_map_storing_segs_of_fitting.tree.lower_bound_smallerthan(key);
    return seg_leaf_iter.key();

}



template<typename KeyType, typename Pos, typename Floating, typename Iterator>
inline Pos FittingTreeIndexer<KeyType, Pos, Floating, Iterator>::predict_position(KeyType key) const {
    /**
      * Returns the approximate position of a key.
      * @param key the value to search for
      * @return a struct with the approximate position
      * @see approx_pos_t
      */
    //btree_const_iterator seg_leaf_iter = btree_friend_.btree_map_slot_float.tree.lower_bound(key);
    auto seg_leaf_iter = btree_friend_.btree_map_storing_segs_of_fitting.tree.lower_bound_smallerthan(key);
    KeyType start, intercept;
    Floating slope;
    start = seg_leaf_iter.key();
    slope = seg_leaf_iter.data().first;
    intercept = seg_leaf_iter.data().second;
    segment_type predicted_pos(start, slope, intercept);

    size_t predi_pos = round(predicted_pos(key));
    size_t predi_lo = predi_pos >= epsilon_ ? predi_pos - epsilon_ : 0;
    size_t predi_hi = predi_pos + epsilon_ <= data_indexed_.size() ? predi_pos + epsilon_ : data_indexed_.size();

    // corner case: key is less than the first segment
    if (seg_leaf_iter.key() > key){
        predi_pos = 0;
        predi_lo = 0;
        int i(0);
        while (((*(data_indexed_.begin() + i)).first) < seg_leaf_iter.key()){
            i++;
        }
        predi_hi = i;
    }

    return {predi_pos, predi_lo, predi_hi};

    /*
    if (seg_leaf_iter.currslot == 0 and seg_leaf_iter.currnode->prevleaf){
        btree_const_iterator seg_prev_leaf_iter = seg_leaf_iter.currnode->prevleaf;
        short last_slot = seg_prev_leaf_iter.currnode->slotuse - 1 ;
        seg_start = seg_prev_leaf_iter.currnode->slotkey[last_slot];
        seg_slope = seg_prev_leaf_iter.currnode->slotdata[last_slot];
    } else{
        short slot = seg_leaf_iter.currslot - 1;  // the last key less than the query ''key''
        seg_start = seg_leaf_iter.currnode->slotkey[slot];
        seg_slope = seg_leaf_iter.currnode->slotdata[slot];
    }
     */
}



template<typename KeyType, typename Pos, typename Floating, typename Iterator>
inline bool FittingTreeIndexer<KeyType, Pos, Floating, Iterator>::check_within_segment(KeyType key) {
    /**
      * Returns the approximate position of a key.
      * @param key the value to search for
      * @return a struct with the approximate position
      * @see approx_pos_t
      */
    //btree_const_iterator seg_leaf_iter = btree_friend_.btree_map_slot_float.tree.lower_bound(key);
    auto seg_leaf_iter = btree_friend_.btree_map_storing_segs_of_fitting.tree.lower_bound_smallerthan(key);
    KeyType seg_head = seg_leaf_iter.key();
    KeyType seg_tail = segments_head_tail_[seg_head];

    if (key >= seg_head and key <= seg_tail){
        return true;
    } else{
        return false;
    }
}

/*
struct CompareForDataPair{
        static size_t as_key(std::pair<size_t, size_t> data_pair){
            return data_pair.first;  // (key, index_pos) pair
        }
        static size_t as_key(KeyType key){
            return key;
        }
    template< typename T1, typename T2 >
    bool operator()( T1 const& t1, T2 const& t2 ) const
    {
        return as_key(t1) < as_key(t2);
    }
};
 */

template<typename KeyType, typename Pos, typename Floating, typename Iterator>
inline KeyType FittingTreeIndexer<KeyType, Pos, Floating, Iterator>::correct_position(Pos predicted_pos, KeyType key) {
    // assert(key >= predicted_pos.seg_start);
    // auto predi_pos = round(double(key - predicted_pos.seg_start) * predicted_pos.seg_slope);
    //auto predi_pos = round(predicted_pos(key));
    //size_t delta_lo = predi_pos >= epsilon_ ? predi_pos - epsilon_ : 0;
    //size_t delta_hi = predi_pos + epsilon_ <= data_indexed_.size() ? predi_pos + epsilon_ : predi_pos + epsilon_ - data_indexed_.size();
    auto lo = this->data_indexed_.begin() + predicted_pos.lo;
    auto hi = this->data_indexed_.begin() + predicted_pos.hi;
    auto res_iter = std::lower_bound(lo, hi, key, CompareForDataPair<KeyType>());
    return std::distance(this->data_indexed_.begin(), res_iter); // the true indexed position of query ``key''
}


#endif //LEARNED_INDEX_FITTINGTREEINDEXER_HPP
