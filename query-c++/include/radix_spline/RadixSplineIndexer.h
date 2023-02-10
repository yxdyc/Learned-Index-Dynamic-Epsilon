//
// Created by daoyuan on 2021/1/11.
//

#ifndef LEARNED_INDEX_RADIXSPLINEINDEXER_H
#define LEARNED_INDEX_RADIXSPLINEINDEXER_H

#include <random>

#include "radix_spline.h"
#include "builder.h"
#include "cnpy.h"


class RadixSpline;

/*
 * RadixSplineIndexer, which integrates the radix-spline index into the sampling enhancement.
 * Based on the official implementation: [Kipf, Andreas, et al. "RadixSpline: a single-pass learned index." aiDM. 2020.]
 */

template<typename KeyType, typename Pos, typename Floating, typename DataIterator, typename Payload_type>
class RadixSplineIndexer : public rs::RadixSpline<KeyType> {

protected:
    size_t error_;   ///< the maximum allowed error in the last level of the RadixSpline index
    size_t num_radix_bits_;   ///< the num_radix_bits of the RadixSpline index
    DataIterator first_data_iter_;    ///< The iterator of the smallest element in the data.
    DataIterator last_data_iter_;    ///< The (iterator + 1) of the largest element in the data.
    size_t data_size_;    ///< The number of elements in the data.
    rs::RadixSpline<KeyType> learned_rs_;

    // dynamic epsilon or fix epsilon
    bool dynamaic_epsilon_ = false;
    std::vector<unsigned int> seg_epsilons_ = {}; // for a convenient c++/python running, load the epsilons from python code's output

    // for analysis
    std::set<KeyType> sampled_data_positions; // store the positions of the sampled data
    size_t upper_bound_iter_count_; // statistic for the exp_search
    size_t lower_bound_iter_count_; // statistic for the exp_search
    long binary_search_len_; // statistic for the exp_search

public:
    RadixSplineIndexer(DataIterator first_key_iter, DataIterator last_key_iter, size_t data_size) :
            first_data_iter_(first_key_iter), last_data_iter_(last_key_iter), data_size_(data_size) {};

    template<class epsilon_type>
    void set_dynamic_epsilons(std::vector<epsilon_type> &seg_epsilon){
        dynamaic_epsilon_ = true;
        seg_epsilons_.reserve(seg_epsilon.size());
        for (auto epsilon : seg_epsilon){
            seg_epsilons_.template emplace_back(static_cast<unsigned int>(epsilon));
        }
    }

    /**
    * building related functions
    */
    std::chrono::system_clock::time_point
    learn_index(const DataIterator first_iter, const DataIterator last_iter, size_t error,
                size_t num_radix_bits = 18,
                double sample_rate = 1.0, bool use_complete_segments = false,
                std::string sample_strategy = "random", int seed = 1234) {
        std::vector<std::pair<KeyType, size_t>> sampled_data;
        std::chrono::system_clock::time_point t0;

        this->error_ = error;
        this->num_radix_bits_ = num_radix_bits;
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
            auto min_key = (*first_data_iter_).first;
            auto max_key = (*(last_data_iter_ - 1)).first;
            rs::Builder<KeyType> rsb(min_key, max_key, num_radix_bits_, error_);
            if (dynamaic_epsilon_) rsb.load_dynamic_epsilons(seg_epsilons_, error_);
            for (DataIterator iter = first_data_iter_; iter != last_data_iter_; iter++) {
                auto key = *iter;
                rsb.AddKey(key.first, key.second);
            }
            learned_rs_ = rsb.Finalize();
        }
        // if (use_complete_segments) {
        //     completed_learned_segments_.clear();
        //     complete_segments();
        //     learned_segments_ = completed_learned_segments_;
        // }
        // organize_segments(strategy, recursive_err);
        return t0;
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

        size_t estimate = learned_rs_.GetEstimatedPosition(key);
        // const size_t lo = (estimate < learned_rs_.max_error_) ? 0 : (estimate - learned_rs_.max_error_);
        // const size_t hi = (estimate + learned_rs_.max_error_ + 2 > learned_rs_.num_keys_) ? learned_rs_.num_keys_ : (
        //         estimate + learned_rs_.max_error_ + 2);
        auto lo = SUB_ERR(estimate, error_, data_size_);
        auto hi = ADD_ERR(estimate, error_, data_size_);
        if (UNLIKELY(estimate > hi))
            estimate = hi;

        return {estimate, lo, hi};

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
        size += learned_rs_.GetSize();

        // exclude the variables for analysis
        size -= sampled_data_positions.size() * sizeof(KeyType);
        size -= sizeof(upper_bound_iter_count_);
        size -= sizeof(lower_bound_iter_count_);
        size -= sizeof(binary_search_len_);
        return size;
    }

    void print_stats() {
        std::cout << "number of all spline points: " << learned_rs_.spline_points_.size() << std::endl;

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


    void evaluate_indexer_split_by_sampled_keys(bool save_x_y_y_hat = false) {
        std::vector<Pos> all_predicted_pos;
        auto t0 = std::chrono::high_resolution_clock::now();
        int i = 0;
        for (DataIterator it = first_data_iter_; it != last_data_iter_; it++, i++) {
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
        std::vector<size_t> all_true_pos_sampled, all_true_pos_non_sampled,
                all_predicted_pos_without_metainfo_within_seg, all_predicted_pos_without_metainfo_non_sampled;
        dlib::matrix<double, 1, 4> results;
        for (size_t i = 0; i < data_size_; i++) {
            size_t pos = (*(first_data_iter_ + i)).second;
            if (sampled_data_positions.find(pos) != sampled_data_positions.end()) {
                all_true_pos_sampled.push_back((*(first_data_iter_ + i)).second);
                size_t predi_pos = all_predicted_pos[i].pos;
                size_t pos_delta = predi_pos >= i ? (predi_pos - i) : (i - predi_pos);
                if (pos_delta > error_) {
                    Pos err_predicted_pos = predict_position((*(first_data_iter_ + i)).first);
                }
                all_predicted_pos_without_metainfo_within_seg.push_back(predi_pos);
            } else {
                all_true_pos_non_sampled.push_back((*(first_data_iter_ + i)).second);
                size_t predi_pos = all_predicted_pos[i].pos;
                size_t pos_delta = predi_pos >= i ? (predi_pos - i) : (i - predi_pos);
                if (pos_delta > error_) {
                    Pos err_predicted_pos = predict_position((*(first_data_iter_ + i)).first);
                }
                all_predicted_pos_without_metainfo_non_sampled.push_back(predi_pos);
            }
        }

        std::cout << "Upper and lower iter counts in exp_search are: " << double_t(upper_bound_iter_count_) / data_size_
                  << ", " << double_t(lower_bound_iter_count_) / data_size_ << std::endl;
        std::cout << "Binary search length in exp_search is: " << double_t(binary_search_len_) / data_size_
                  << std::endl;

        std::cout << "The size of sampled and none-sampled are: " << all_true_pos_sampled.size() << ", "
                  << all_true_pos_non_sampled.size() << std::endl;
        std::cout << "number of all spline points: " << learned_rs_.spline_points_.size() << std::endl;
        results = evaluate_regression_ML(all_predicted_pos_without_metainfo_within_seg, all_true_pos_sampled, error_);
        std::cout << "ML oriented matricx for sampled keys: " << results;
        results = evaluate_regression_ML(all_predicted_pos_without_metainfo_non_sampled, all_true_pos_non_sampled,
                                         error_);
        std::cout << "ML oriented matricx for non-sampled keys: " << results;

        all_predicted_pos_without_metainfo_within_seg.insert(all_predicted_pos_without_metainfo_within_seg.end(),
                                                             all_predicted_pos_without_metainfo_non_sampled.begin(),
                                                             all_predicted_pos_without_metainfo_non_sampled.end());
        all_true_pos_sampled.insert(all_true_pos_sampled.end(),
                                    all_true_pos_non_sampled.begin(),
                                    all_true_pos_non_sampled.end());
        results = evaluate_regression_ML(all_predicted_pos_without_metainfo_within_seg, all_true_pos_sampled,
                                         error_);
        std::cout << "ML oriented matricx for all keys: " << results;
        double mae = results(0, 2);
        std::cout << "log2(MAE)*(#Seg): " << log2(mae) * double(learned_rs_.spline_points_.size() - 1) << "\n"
                  << std::endl;

        if (save_x_y_y_hat) {
            // save the results for analysis
            int N_row = 3; // x, y, \hat{y}
            int N_col = data_size_;
            std::vector<double> all_x;
            std::vector<double> all_y;
            std::vector<double> all_y_hat;
            for (int i = 0; i < N_col; i++) {
                auto x = (*(first_data_iter_ + i)).first;
                auto y = (*(first_data_iter_ + i)).second;
                auto y_hat = all_predicted_pos[i].pos;
                all_x.emplace_back(x);
                all_y.emplace_back(y);
                all_y_hat.emplace_back(y_hat);
            }
            std::vector<double> all_spline_x;
            std::vector<double> all_spline_y;
            for (auto point : learned_rs_.spline_points_) {
                auto x = point.x;
                auto y = point.y;
                all_spline_x.emplace_back(x);
                all_spline_y.emplace_back(y);
            }

            cnpy::npy_save("tmp_out/data_x.npy", all_x, "w");
            cnpy::npy_save("tmp_out/data_y.npy", all_y, "w");
            cnpy::npy_save("tmp_out/data_y_hat.npy", all_y_hat, "w");
            std::cout << "Save x, y, y' to " << "tmp_out/data_x/y/yhat.npy" << std::endl;

            cnpy::npy_save("tmp_out/data_spline_x.npy", all_spline_x, "w");
            cnpy::npy_save("tmp_out/data_spline_y.npy", all_spline_y, "w");
            std::cout << "Save splines to " << "tmp_out/data_spline_x/y.npy" << std::endl;

        }

    }
};


#endif //LEARNED_INDEX_RADIXSPLINEINDEXER_H
