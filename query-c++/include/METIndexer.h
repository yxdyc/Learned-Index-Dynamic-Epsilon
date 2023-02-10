//
// Created by daoyuan on 2021/1/11.
//

#ifndef LEARNED_INDEX_METINDEXER_H
#define LEARNED_INDEX_METINDEXER_H

#include <random>

#include "cnpy.h"

#include "FittingTreeIndexer.hpp" // We re-use the segment and prediction functions of FITingTree


template<typename KeyType, typename Pos, typename Floating, typename DataIterator, typename Payload_type>
class METIndexer {
    using segment_type = FittingSegment<KeyType, Floating>;

protected:
    std::vector<segment_type> learned_segments_;
    std::vector<KeyType> learned_segments_first_keys_;

    size_t error_;   ///< the maximum allowed error in the last level of the MET index
    DataIterator first_data_iter_;    ///< The iterator of the smallest element in the data.
    DataIterator last_data_iter_;    ///< The (iterator + 1) of the largest element in the data.
    size_t data_size_;    ///< The number of elements in the data.

    // dynamic epsilon or fix epsilon
    bool use_dynamaic_epsilon_ = false;
    std::vector<unsigned int> seg_epsilons_ = {}; // for a convenient c++/python running, load the epsilons from python code's output
    int cur_seg_epsilon_idx_ = 0;

    // for analysis
    std::set<KeyType> sampled_data_positions; // store the positions of the sampled data
    size_t upper_bound_iter_count_; // statistic for the exp_search
    size_t lower_bound_iter_count_; // statistic for the exp_search
    long binary_search_len_; // statistic for the exp_search

public:
    METIndexer(DataIterator first_key_iter, DataIterator last_key_iter, size_t data_size) :
            first_data_iter_(first_key_iter), last_data_iter_(last_key_iter), data_size_(data_size) {};

    template<class epsilon_type>
    void set_dynamic_epsilons(std::vector<epsilon_type> &seg_epsilon) {
        use_dynamaic_epsilon_ = true;
        seg_epsilons_.reserve(seg_epsilon.size());
        for (auto epsilon : seg_epsilon) {
            seg_epsilons_.template emplace_back(static_cast<unsigned int>(epsilon));
        }
    }

    /**
    * building related functions
    */
    std::chrono::system_clock::time_point
    learn_index(const DataIterator first_iter, const DataIterator last_iter, size_t error,
                double sample_rate = 1.0, bool use_complete_segments = false,
                std::string sample_strategy = "random", int seed = 1234) {
        std::vector<std::pair<KeyType, size_t>> sampled_data;
        std::chrono::system_clock::time_point t0;

        this->error_ = error;
        if (sample_rate != 1.0) {
            std::cout << "We won't use sampling in the dynamic epsilon version" << std::endl;
            // sample_on_strategy(first_iter, last_iter, sampled_data, sample_rate,
            //                    std::mt19937{seed}, sample_strategy);
            // std::sort(sampled_data.begin(), sampled_data.end());
            // size_t sample_size = sampled_data.size();
            // std::cout << "The sampled strategy, sampled rate, sampled size are " << sample_strategy << ", "
            //           << sample_rate << ", " << sample_size <<
            //           "\nThe |y_r-y_l| of sampled data statistic: " << std::endl;

            // std::vector<long> diff_y_right_left;
            // diff_y_right_left.emplace_back(std::abs(sampled_data[0].second - 0)); // unify by setting the y'_0 = 0
            // for (int i = 0; i < sample_size - 1; i++) {
            //     diff_y_right_left.emplace_back(std::abs(sampled_data[i + 1].second - sampled_data[i].second));
            // }
            // long ori_data_size = std::distance(first_iter, last_iter);
            // std::cout << "The last sampled y and ori_data_size are " << long(sampled_data[sample_size - 1].second)
            //           << ", "
            //           << ori_data_size << std::endl;
            // diff_y_right_left.emplace_back(std::abs(
            //         ori_data_size - long(sampled_data[sample_size - 1].second))); // unify by setting the y'_n = n
            // basic_statistic(diff_y_right_left);

            // for (auto &x : sampled_data) {
            //     sampled_data_positions.insert(x.second);
            // }
            // t0 = std::chrono::high_resolution_clock::now();
            // auto min_key = sampled_data.front().first;
            // auto max_key = sampled_data.back().first;
            // rs::Builder<KeyType> rsb(min_key, max_key, num_radix_bits_, error_);
            // if (use_dynamaic_epsilon_) rsb.load_dynamic_epsilons(seg_epsilons_);
            // for (const auto &key : sampled_data) rsb.AddKey(key.first, key.second);
            // learned_rs_ = rsb.Finalize();
        } else {
            t0 = std::chrono::high_resolution_clock::now();
            learned_segments_ = learn_segments(first_iter, last_iter);
        }
        return t0;
    }

    std::vector<FittingSegment<KeyType, Floating>>
    learn_segments(
            DataIterator first_iter, DataIterator last_iter) {
        std::vector<segment_type> segments;
        segments.reserve(8192);

        size_t input_data_size = std::distance(first_iter, last_iter);
        std::cout << "Input data size: " << input_data_size << std::endl;
        if (input_data_size == 0)
            return segments;
        assert(std::is_sorted(first_iter, last_iter));

        int lookahead_n = 404;
        double look_ahead_mean_key_interval = calculate_mean_key_interval(first_iter, first_iter + lookahead_n);
        double cur_segment_slope = 1 / look_ahead_mean_key_interval;
        KeyType segment_start = (*first_iter).first;
        size_t intercept = 0;
        segments.emplace_back(segment_start, cur_segment_slope, intercept);
        learned_segments_first_keys_.template emplace_back(segment_start);
        auto cur_seg = segments.back();
        double cumulated_mean_seg_length = 0.0;
        int cumulated_n_seg_length = 0;


        for (DataIterator it = first_iter; it != last_iter; it++) {
            auto cur_y = it->second;
            auto cur_key = it->first;
            auto y_hat = cur_seg(cur_key);
            auto pred_error = y_hat > cur_y ? y_hat - cur_y : cur_y - y_hat;
            if (pred_error > error_) {
                // update mean_seg_length of learned segments
                cumulated_n_seg_length++;
                auto cur_seg_len = cur_y - intercept;
                if (cumulated_mean_seg_length == 0){
                    cumulated_mean_seg_length = cur_seg_len;
                } else{
                    cumulated_mean_seg_length = (cur_seg_len + cumulated_n_seg_length * cumulated_mean_seg_length) /
                                                (cumulated_n_seg_length + 1);
                }
                int possible_lookahead_n = std::round(cumulated_mean_seg_length * 0.4);
                auto remain_data_n = std::distance(it, last_iter);
                lookahead_n = possible_lookahead_n < remain_data_n ? possible_lookahead_n : remain_data_n;

                // new segment
                segment_start = cur_key;
                intercept = cur_y;
                look_ahead_mean_key_interval = calculate_mean_key_interval(it, it + lookahead_n);
                cur_segment_slope = 1 / look_ahead_mean_key_interval;
                segments.emplace_back(segment_start, cur_segment_slope, intercept);
                learned_segments_first_keys_.template emplace_back(segment_start);
                cur_seg = segments.back();

                // dynamic epsilon
                if (use_dynamaic_epsilon_) {
                    cur_seg_epsilon_idx_++;
                    error_ = seg_epsilons_[cur_seg_epsilon_idx_ % seg_epsilons_.size()];
                }
            }
        }
        return segments;
    }


    inline
    double calculate_mean_key_interval(DataIterator first_iter, DataIterator last_iter) {
        int data_range_len = std::distance(first_iter, last_iter);
        double mean_key_interval_ = double(last_iter->first - first_iter->first) / (data_range_len-1);
        return mean_key_interval_;
    }


    /**
    * look-up related functions
    */

    inline Payload_type *get_payload_given_key(KeyType key) {
        auto predicted_pos = predict_position(key);
        auto res = correct_position(predicted_pos, key);
        return res;
    }

    inline Pos predict_position(KeyType key) const {
        // in sampling case, for the key before/after the first/last key, we return its y^ as y_0-1 or y_n+1
        auto first_data = (*(this->first_data_iter_)), last_data = (*(this->last_data_iter_ - 1));
        if (UNLIKELY(key < first_data.first))
            return {first_data.second - 1, 0, first_data.second + error_};
        if (UNLIKELY(key > last_data.first))
            return {last_data.second + 1, last_data.second - error_, data_size_ - 1};

        // binary find the segment that is responsible for the given key
        auto responsible_seg_iter = std::prev(
                std::upper_bound(learned_segments_first_keys_.begin(), learned_segments_first_keys_.end(), key));
        auto responsible_seg_idx = std::distance(learned_segments_first_keys_.begin(), responsible_seg_iter);
        auto responsible_seg = learned_segments_[responsible_seg_idx];
        size_t y_hat = responsible_seg(key);
        y_hat = y_hat > data_size_ ? data_size_ : y_hat;
        size_t seg_error = error_;
        if (use_dynamaic_epsilon_){
            auto seg_error = seg_epsilons_[responsible_seg_idx];
        }

        auto lo = SUB_ERR(y_hat, seg_error, data_size_);
        auto hi = ADD_ERR(y_hat, seg_error, data_size_);
        if (UNLIKELY(y_hat > hi))
            y_hat = hi;

        return {y_hat, lo, hi};

    }


    inline Payload_type *correct_position(Pos predicted_pos, KeyType key, bool exp_search = false) {
        DataIterator res_iter;
        if (exp_search) {
            // exponential search, for dynamic case that violates the error bounds.
            res_iter = exponential_search(this->first_data_iter_, this->last_data_iter_,
                                          key, predicted_pos.pos, CompareForDataPair<KeyType>(),
                                          upper_bound_iter_count_, lower_bound_iter_count_, binary_search_len_);
        } else {
            // error_bounded binary search
            auto lo = this->first_data_iter_ + predicted_pos.lo;
            auto hi = this->first_data_iter_ + predicted_pos.hi;
            res_iter = std::lower_bound(lo, hi, key, CompareForDataPair<KeyType>());
        }
        if (res_iter == this->last_data_iter_) return nullptr;
        else return &(*res_iter).second;

    }

    /**
      * evaluation related functions
     */

    long long data_size() {
        return data_size_ * sizeof(std::pair<KeyType, Payload_type>);
    }

    long long model_size() {
        long long size = sizeof(*this);
        size += learned_segments_.size() * sizeof(segment_type);
        size += learned_segments_first_keys_.size() * sizeof(KeyType);

        // exclude the variables for analysis
        size -= sampled_data_positions.size() * sizeof(KeyType);
        size -= sizeof(upper_bound_iter_count_);
        size -= sizeof(lower_bound_iter_count_);
        size -= sizeof(binary_search_len_);
        return size;
    }

    void print_stats() {
        std::cout << "number of all learned segments: " << learned_segments_.size() << std::endl;

        // size
        size_t data_size = this->data_size();
        size_t model_size = this->model_size();
        std::cout << "size of index model: " << model_size << std::endl;
        std::cout << "size of data: " << data_size << std::endl;
        std::cout << "total size: " << (model_size + data_size) << std::endl;
    }

    void evaluate_indexer(std::vector<std::pair<KeyType, Payload_type>> whole_data, std::string pred_file_name = "",
                          bool shuffle = true, bool exp_search = false) {
        print_stats();

        if (shuffle == true) {
            std::srand(1234);
            random_shuffle(whole_data.begin(), whole_data.end());
        }

        auto evaluated_data_size = whole_data.size();
        std::vector<Pos> all_predicted_pos;
        std::vector<size_t> all_true_pos, all_predicted_pos_without_metainfo;
        int i = 0;
        size_t wrong_return_count = 0;
        auto t0 = std::chrono::high_resolution_clock::now();
        auto accumulated_predict_time = std::chrono::duration_cast<std::chrono::nanoseconds>(t0 - t0).count();
        auto accumulated_correct_time = std::chrono::duration_cast<std::chrono::nanoseconds>(t0 - t0).count();
        upper_bound_iter_count_ = 0;
        lower_bound_iter_count_ = 0;
        binary_search_len_ = 0;
        std::vector<long> all_query_times(whole_data.size()); // used for statistics of the query times, e.g., 99 percent time
        for (DataIterator it = whole_data.begin(); it != whole_data.end(); it++, i++) {
            // test prediction stage
            auto key = (*it).first;
            auto tmp_t0 = std::chrono::high_resolution_clock::now();
            Pos predicted_pos = predict_position(key);
            // test correction stage
            auto tmp_t1 = std::chrono::high_resolution_clock::now();
            auto corrected_res = correct_position(predicted_pos, key, exp_search);
            auto tmp_t2 = std::chrono::high_resolution_clock::now();
            // track predict_time, correct_time, y and y^
            accumulated_predict_time += std::chrono::duration_cast<std::chrono::nanoseconds>(
                    tmp_t1 - tmp_t0).count();
            accumulated_correct_time += std::chrono::duration_cast<std::chrono::nanoseconds>(
                    tmp_t2 - tmp_t1).count();
            all_query_times.emplace_back(std::chrono::duration_cast<std::chrono::nanoseconds>(
                    tmp_t2 - tmp_t0).count());
            if (corrected_res and (*corrected_res) != (*it).second) {
                wrong_return_count++;
            }
            all_predicted_pos.emplace_back(predicted_pos);
            all_predicted_pos_without_metainfo.emplace_back(predicted_pos.pos);
            all_true_pos.emplace_back((*it).second);
        }
        assert(all_predicted_pos.size() == data_size_);

        // lookup-time per get_payload_given_key
        auto predict_time = accumulated_predict_time / evaluated_data_size;
        auto correct_time = accumulated_correct_time / evaluated_data_size;
        auto overall_time = predict_time + correct_time;
        std::sort(all_query_times.begin(), all_query_times.end());
        auto p99_time = all_query_times[int(all_query_times.size()/100*99)];
        std::cout << "p99 time: " << p99_time << std::endl;
        std::cout << "predict time: " << predict_time << std::endl;
        std::cout << "correct time: " << correct_time << std::endl;
        std::cout << "overall get_payload_given_key time: " << overall_time << std::endl;
        std::cout << "Upper and lower iter counts in exp_search are: " << double_t(upper_bound_iter_count_) / data_size_
                  << ", " << double_t(lower_bound_iter_count_) / data_size_ << std::endl;
        std::cout << "Binary search length in exp_search is: " << double_t(binary_search_len_) / data_size_
                  << std::endl;

        // ML-oriented metric
        dlib::matrix<double, 1, 4> results;
        bool write_to_file = !(pred_file_name.empty());
        std::cout << "write_to_file_name: " << pred_file_name << std::endl;
        std::cout << "wrong return count: " << wrong_return_count << std::endl;
        results = evaluate_regression_ML(all_predicted_pos_without_metainfo, all_true_pos, error_, write_to_file,
                                         pred_file_name);
        std::cout << "ML oriented matricx: " << results;
        std::cout << "Mean memory access number during the correction stage: " << std::log2(results(0, 2)) << std::endl;
    }

};


#endif //LEARNED_INDEX_METINDEXER_H
