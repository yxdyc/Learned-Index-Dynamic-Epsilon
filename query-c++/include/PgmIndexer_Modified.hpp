//
// Patched PGM Indexer for sampling case.
//


#ifndef LEARNED_INDEX_PGMINDEXER_MODIFIED_HPP
#define LEARNED_INDEX_PGMINDEXER_MODIFIED_HPP



#include <vector>
#include <list>
#include <chrono>
#include <cstddef>
#include "IndexMechanism.hpp"
#include "Utilities.hpp"
#include <iterator>
#include <dlib/matrix.h>
#include <dlib/statistics.h>
#include <random>
#include <experimental/algorithm>




#define ADD_ERR(x, error, size) ((x) + (error) >= (size) ? (size) - 1 : (x) + (error))
#define SUB_ERR(x, error, size) ((x) <= (error) ? 0 : ((x) - (error)))
#define BIN_SEARCH_THRESHOLD 512

#define LIKELY(x)   __builtin_expect(!!(x), 1)
#define UNLIKELY(x) __builtin_expect(!!(x), 0)


/**
 * calculate the ML-oriented, regression metrics.
 * return a result matrix
 * - M(0) == the mean squared error.
     The MSE is given by: sum over i: pow(reg_funct(x_test[i]) - y_test[i], 2.0)
   - M(1) == the correlation between reg_funct(x_test[i]) and y_test[i].
     This is a number between -1 and 1.
   - M(2) == the mean absolute error.
     This is given by: sum over i: abs(reg_funct(x_test[i]) - y_test[i])
   - M(3) == the standard deviation of the absolute error.
 */




/**
 * A struct that stores a segment.
 * @tparam KeyType the type of the elements that the segment indexes
 * @tparam Floating the floating-point type of the segment's parameters
 */
template<typename KeyType, typename Floating>
struct SegmentModified {
    //static_assert(std::is_floating_point<Floating>());
    KeyType seg_start;              ///< The first key that the segment indexes.
    Floating seg_slope;     ///< The slope of the segment.
    Floating seg_intercept; ///< The intercept of the segment.
    KeyType seg_end;              ///< The last key that the segment indexes.
    KeyType number_of_seg_keys;      ///< The number of keys covered by this segment.



    SegmentModified() = default;

    /**
     * Constructs a new segment.
     * @param key the first_key_iter key that the segment indexes
     * @param slope the slope of the segment
     * @param intercept the intercept of the segment
     */
    SegmentModified(KeyType key, Floating slope, Floating intercept, KeyType end, KeyType number_keys) :
            seg_start(key), seg_slope(slope), seg_intercept(intercept), seg_end(end), number_of_seg_keys(number_keys) {};

    friend inline bool operator<(const SegmentModified &s, const KeyType k) {
        return s.seg_start < k;
    }

    friend inline bool operator<(const SegmentModified &s1, const SegmentModified &s2) {
        return s1.seg_start < s2.seg_start;
    }

    /**
     * Returns the approximate position of the specified key.
     * @param k the key whose position must be approximated
     * @return the approximate position of the specified key
     */
    inline size_t operator()(KeyType k) const {
        // Because of the sampling, we may query data before the seg_start of the fisrt seg
        assert(k >= seg_start | (k < seg_start and seg_intercept == 0));
        if (k < seg_start ) {return Floating(0);}
        Floating pos = seg_slope * (k - seg_start) + seg_intercept;
        return pos > Floating(0) ? round(pos) : 0ul;
    }
};


template<typename KeyType, typename Pos, typename Floating, typename Iterator>
class PgmIndexerModified {
    using floating_type = Floating;
    using segment_type = SegmentModified<KeyType, Floating>;
    using segment_data_type = SegmentData<Floating>;
    segment_type root_;
    size_t root_limit_;
    std::vector<segment_type> completed_learned_segments_;
    std::map<KeyType, KeyType> segments_head_tail_;
    std::set<KeyType> sampled_keys;
    std::vector<KeyType> place_holder_{0}; // have only one fake data, '0': size is 1,  used for query speedup
    std::vector<std::pair<KeyType, std::vector<KeyType>>> gapped_array_with_linking_array_;
    std::vector<KeyType> gapped_array_;
    std::vector<std::vector<KeyType>> linking_array_;
    int binary_search_threshold_; // the binary search threshold, below this value we will use linear scan on linking array


public:
    PgmIndexerModified(Iterator first_key_iter, Iterator last_key_iter, size_t data_size, int bin_search_t=128) :
            first_key_iter_(first_key_iter), last_key_iter_(last_key_iter), data_size_(data_size),
            binary_search_threshold_(bin_search_t){};

    std::chrono::system_clock::time_point learn_index(const Iterator first_iter, const Iterator last_iter, size_t error,
            std::string strategy, size_t recursive_err, double sample_rate = 1.0, bool use_complete_segments = false) {
        std::vector<std::pair<KeyType, size_t>> sampled_data;
        std::chrono::system_clock::time_point t0;
        this->error_ = error;
        this->recursive_error_ = recursive_err;
        if (sample_rate != 1.0){
            size_t sample_size = round(data_size_ * sample_rate);

            //std::sample(first_iter, last_iter, std::back_inserter(sampled_data), sample_size, std::mt19937{std::random_device{}()});
            std::experimental::sample(first_iter, last_iter, std::back_inserter(sampled_data),
                    sample_size, std::mt19937{std::random_device{}()});
            std::sort(sampled_data.begin(), sampled_data.end());
            for (auto &x : sampled_data){
                sampled_keys.insert(x.first);
            }
            t0 = std::chrono::high_resolution_clock::now();
            learned_segments_ = learn_segments(sampled_data.begin(), sampled_data.begin() + sample_size, error);
        } else{
            t0 = std::chrono::high_resolution_clock::now();
            learned_segments_ = learn_segments(first_iter, last_iter, error);
        }
        if (use_complete_segments){
            completed_learned_segments_.clear();
            complete_segments();
            learned_segments_ = completed_learned_segments_;
        }
        organize_segments(strategy, recursive_err);
        return t0;
    }



    std::chrono::system_clock::time_point re_learn_index_segment_wise(
            const Iterator first_iter, const Iterator last_iter, std::string strategy, size_t recursive_err) {
        std::chrono::system_clock::time_point t0;

        size_t re_learn_data_size = std::distance(first_iter, last_iter);
        size_t seg_begin_y(0), seg_end_y(0);
        for (int i = 0; i < learned_segments_.size(); i++){
            auto seg_i = learned_segments_[i];
            // find the begin key idx of the segment
            while ((*(first_iter + seg_begin_y)).first != learned_segments_[i].seg_start){
                seg_begin_y++;
                assert(seg_begin_y < re_learn_data_size);
            }
            // find the end key idx of the segment
            seg_end_y = seg_begin_y;
            if (i == (learned_segments_.size()-1)){
                seg_end_y = re_learn_data_size - 1;
            }else{
                while ((*(first_iter + seg_end_y)).first != learned_segments_[i].seg_end){
                    seg_end_y++;
                }
            }
            auto linear_para = train_linear_model(first_iter + seg_begin_y, first_iter + seg_end_y + 1);
            learned_segments_[i].seg_slope = linear_para.second;
            learned_segments_[i].seg_intercept = linear_para.first;
        }
        organize_segments(strategy, recursive_err);
        return t0;
    }


    inline KeyType query(KeyType key) {
        auto predicted_pos = predict_position(key);
        KeyType res = correct_position(predicted_pos, key);
        return res;
    }

    inline KeyType query_in_gapped_array(KeyType key) {
        auto pos = predict_position(key);
        auto res = correct_query_in_gapped_array(pos, key);
        return res;
    }



    /**
     * insertion for linking array strategy
     */
    void insert(KeyType key) {
#ifdef Debug
        if (key == 20170607040203){
            int stop = 0;
        }
#endif
        auto predicted_pos = predict_position(key).pos;
        // to maintain a total order on the gapped array after insertion, we need found the upper_bound_exp_search
        auto iter_upper_bound = upper_bound_exp_search_total_order(
                gapped_array_.begin(), gapped_array_.end(),
                key, predicted_pos, LessComparatorForGappedArray<KeyType>(),
                upper_bound_iter_count_, lower_bound_iter_count_);

        // the inserted key <= the first (also minimal) item of the stored keys of the indexer
        if (iter_upper_bound == gapped_array_.begin()){
            gapped_array_[0] = key;
            linking_array_[0].emplace_back(key);
            return;
        }
        // the inserted key <= the last gapped array item of the stored keys of the indexer
        if (iter_upper_bound == gapped_array_.end()){
            linking_array_[linking_array_.size()-1].emplace_back(key);
            return;
        }

        auto iter_upper_bound_minus_one = iter_upper_bound - 1; // "-1" means the last item by upper_bound
        auto pos_upper_bound_minus_one = std::distance(gapped_array_.begin(), iter_upper_bound_minus_one);
        auto key_upper_bound_minus_one = *iter_upper_bound_minus_one;
        auto key_upper_bound = *(iter_upper_bound_minus_one + 1);
        auto key_in_predicted_pos = *(gapped_array_.begin() + predicted_pos);
        auto link_array_upper_bound_minus_one = linking_array_[pos_upper_bound_minus_one];

        // track the max key to maintain the total order
        // for write-heavy case, we can save the max values on another array with O(n) space, and O(1) update time.
        auto max_key_upper_bound_minus_one = link_array_upper_bound_minus_one[0];
        if (link_array_upper_bound_minus_one.size() > 1){
            max_key_upper_bound_minus_one = *(std::max_element(
                    link_array_upper_bound_minus_one.begin(), link_array_upper_bound_minus_one.end()));
        }

        // toy data:   [7(1), 8(1), 10(0), 10(0), 10(0), 10(1), 13(1), 14(1)]
        // insert 9, res_iter=8(1), which is definitely >= key
        assert (key >= key_upper_bound_minus_one);
        assert (key < key_upper_bound);
        // loop (insertion) invariant: for all k_i^{A_j} in A_j,  k_i^{A_{j-1}} < k_i^{A_j} < k_i^{A_{j+1}}
        // 9 > 8
        if (key > key_upper_bound_minus_one){
            // e.g., predicted_pos = 0 or 1 & pos_upper_bound_minus_one = 1;
            if (predicted_pos <= pos_upper_bound_minus_one or (key < max_key_upper_bound_minus_one)) {
                // put into the linking array,
                predicted_pos = pos_upper_bound_minus_one;
                linking_array_[predicted_pos].emplace_back(key);
            } else {
                // e.g., predicted_pos = 6, key_in_predicted_pos = 13; key_upper_bound = 10;
                // this usually occurs at the large key gap regions
                if (key_in_predicted_pos > key_upper_bound){
                    // insert in the pos_upper_bound, e.g., the right position of 8(1)
                    predicted_pos = pos_upper_bound_minus_one + 1;
                }
                // insert in gap or replace the key, then update the gap data smaller than key
                // e.g., predicted_pos = 3, pos_upper_bound_minus_one = 1
                // fill keys to maintain the total order
                auto filled_num = (predicted_pos - pos_upper_bound_minus_one); // filled_num must >= 1
                assert((*(gapped_array_.begin() + pos_upper_bound_minus_one + filled_num)) <= key_upper_bound);
                std::fill_n(gapped_array_.begin() + pos_upper_bound_minus_one + 1, filled_num, key);
                linking_array_[predicted_pos].emplace_back(key);
            }
        } else {
            // key we implement a index which only stores distinct keys;
            // to store duplicate keys, we need emplace_back the key into its linking array
            return;
        }
    }

    void evaluate_indexer(int payload_size, std::string pred_file_name="", bool shuffle=true) {
        std::vector<std::pair<key_type_transformed, size_t>> tmp_data(first_key_iter_, last_key_iter_);
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
        assert(all_predicted_pos.size() == data_size_);
        upper_bound_iter_count_ = 0;
        lower_bound_iter_count_ = 0;
        binary_search_len_ = 0;
        auto t1 = std::chrono::high_resolution_clock::now();
        i = 0;
        for(Iterator it = tmp_data.begin() ; it != tmp_data.end(); it++, i++){
            auto res = correct_position(all_predicted_pos[i], (*it).first);
            assert(res ==  (*it).first);
        }
        auto t2 = std::chrono::high_resolution_clock::now();
        //lookup-time per query
        auto predict_time = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count() / data_size_;
        auto correct_time = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() / data_size_;
        //size
        size_t size_of_all_segments = this->size_segments_bytes();
        size_t size_of_payload = this->size_payloads_bytes(payload_size);

        // ML-oriented metric
        // regress_metric = ["RMSE", "MAE", "R2"]
        std::vector<size_t> all_true_pos, all_predicted_pos_without_metainfo;
        for (size_t i=0; i < data_size_; i++){
            //all_true_pos.push_back(i);
            all_true_pos.push_back((*(tmp_data.begin() + i)).second);
            size_t predi_pos = all_predicted_pos[i].pos;
            size_t pos_delta = predi_pos >= i ? (predi_pos - i) : (i - predi_pos);
            all_predicted_pos_without_metainfo.push_back(all_predicted_pos[i].pos);
        }
        dlib::matrix<double, 1, 4> results;
        bool write_to_file = (pred_file_name == "") ? false : true;
        std::cout<< "write_to_file_name: " << pred_file_name << std::endl;
        std::cout<< "predict time: " << predict_time << std::endl;
        std::cout<< "correct time: " << correct_time << std::endl;
        std::cout<< "overall query time: " << predict_time + correct_time << std::endl;
        std::cout<< "number of all segments: " << learned_segments_.size() << std::endl;
        std::cout<< "size of all segments: " << size_of_all_segments << std::endl;
        std::cout<< "size of payloads: " << size_of_payload<< std::endl;
        std::cout<< "total size: " << (size_of_payload + size_of_all_segments) << std::endl;
        std::cout<< "Upper and lower iter counts in exp_search are: " << double_t(upper_bound_iter_count_)/ data_size_
                 << ", " << double_t(lower_bound_iter_count_)/data_size_ << std::endl;
        std::cout<< "Binary search length in exp_search is: " << double_t(binary_search_len_)/ data_size_ << std::endl;
        results =  evaluate_regression_ML(all_predicted_pos_without_metainfo, all_true_pos, error_, write_to_file,
                pred_file_name);
        std::cout<< "ML oriented matricx: " << results;
    }

    template <typename T_data>
    void evaluate_indexer(T_data data, int payload_size, std::string pred_file_name="") {
        T_data random_data = data;
        auto tmp_first_key_iter = this->first_key_iter_;
        auto tmp_last_key_iter = this->last_key_iter_;

        this->first_key_iter_ = data.begin(); // we need search on the whole dataset
        this->last_key_iter_ = data.end(); // we need search on the whole dataset
        std::vector<Pos> all_predicted_pos;
        auto t0 = std::chrono::high_resolution_clock::now();
        int i = 0;

        std::srand(1234);
        random_shuffle(random_data.begin(), random_data.end());
        for(Iterator it = random_data.begin(); it != random_data.end(); it++, i++){
            Pos predicted_pos = predict_position((*it).first);
            all_predicted_pos.emplace_back(predicted_pos);
        }
        assert(all_predicted_pos.size() == data_size_);
        upper_bound_iter_count_ = 0;
        lower_bound_iter_count_ = 0;
        binary_search_len_ = 0;
        auto t1 = std::chrono::high_resolution_clock::now();
        i = 0;
        for(Iterator it = random_data.begin(); it != random_data.end(); it++, i++){
            auto res = correct_position(all_predicted_pos[i], (*it).first);
            assert(isEqual(res, (*it).first));
        }
        auto t2 = std::chrono::high_resolution_clock::now();
        //lookup-time per query
        auto predict_time = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count() / data_size_;
        auto correct_time = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() / data_size_;
        //size
        size_t size_of_all_segments = this->size_segments_bytes();
        size_t size_of_payload = this->size_payloads_bytes(payload_size);

        // ML-oriented metric
        // regress_metric = ["RMSE", "MAE", "R2"]
        std::vector<size_t> all_true_pos, all_predicted_pos_without_metainfo;
        for (size_t i=0; i < data_size_; i++){
            //all_true_pos.push_back(i);
            all_true_pos.push_back((*(data.begin() + i)).second);
            size_t predi_pos = all_predicted_pos[i].pos;
            size_t pos_delta = predi_pos >= i ? (predi_pos - i) : (i - predi_pos);
            all_predicted_pos_without_metainfo.push_back(all_predicted_pos[i].pos);
        }
        dlib::matrix<double, 1, 4> results;
        bool write_to_file = (pred_file_name == "") ? false : true;
        std::cout<< "write_to_file_name: " << pred_file_name << std::endl;
        std::cout<< "predict time: " << predict_time << std::endl;
        std::cout<< "correct time: " << correct_time << std::endl;
        std::cout<< "overall query time: " << predict_time + correct_time << std::endl;
        std::cout<< "number of all segments: " << learned_segments_.size() << std::endl;
        std::cout<< "size of all segments: " << size_of_all_segments << std::endl;
        std::cout<< "size of payloads: " << size_of_payload<< std::endl;
        std::cout<< "total size: " << (size_of_payload + size_of_all_segments) << std::endl;
        std::cout<< "Upper and lower iter counts in exp_search are: " << double_t(upper_bound_iter_count_)/ data_size_
            << ", " << double_t(lower_bound_iter_count_)/data_size_ << std::endl;
        std::cout<< "Binary search length in exp_search is: " << double_t(binary_search_len_)/ data_size_ << std::endl;
        results =  evaluate_regression_ML(all_predicted_pos_without_metainfo, all_true_pos, error_, write_to_file,
                                          pred_file_name);
        std::cout<< "ML oriented matricx: " << results;

        this->first_key_iter_ = tmp_first_key_iter;
        this->last_key_iter_ = tmp_last_key_iter;
    }

    void evaluate_gap_proportion(){
        gap_num_ = 0;
        for (auto link_array :linking_array_){
            if (link_array.size() == 0){
                gap_num_++;
            }
        }
        std::cout<< "Gaps and gap proportion are: "<< gap_num_ << ", " <<
        double(gap_num_) / gapped_array_.size() << std::endl;
    }

    void statistic_linking_array(){
        std::vector<double_t > conflicts;
        conflicts.reserve(linking_array_.size());
        for (auto array : linking_array_){
            size_t len = array.size();
            if (len > 1){
                conflicts.emplace_back(len);
            }
        }
        calculate_mean_std(conflicts);
    }

    void reset_gapped_linking_array(){
        gapped_array_.clear();
        linking_array_.clear();
    }

    void set_gapped_linking_array(
            std::vector<std::pair<KeyType, std::vector<KeyType>>> const & gapped_array_with_linking_array){
        reset_gapped_linking_array();
        size_t i = 0;
        gapped_array_.reserve(gapped_array_with_linking_array.size());
        linking_array_.reserve(gapped_array_with_linking_array.size());
        for (auto key_pair : gapped_array_with_linking_array){
            gapped_array_.emplace_back(key_pair.first);
            linking_array_.emplace_back(key_pair.second);
        }
    }

    /**
     * evaluate the index equiped with gapped array + linking array,
     * including the query time (prediction and correction) and index size (size of segements, gaps, and payloads)
     *
     * @whole_data_with_gapped_pos: the evaluated data
     * @gapped_array_with_linking_array: the evaluated data in gapped array form
     */
    void evaluate_index_with_gapped_array(std::vector<std::pair<key_type_transformed, size_t>> whole_data_with_gapped_pos,
            std::vector<std::pair<KeyType, std::vector<KeyType>>> gapped_array_with_linking_array,
            int payload_size, int number_of_non_gap) {
        this->gapped_array_with_linking_array_ = gapped_array_with_linking_array;
        set_gapped_linking_array(gapped_array_with_linking_array);
        std::vector<Pos> all_predicted_pos;
        auto evaluated_data_size = whole_data_with_gapped_pos.size();
        all_predicted_pos.reserve(evaluated_data_size);
        auto t0 = std::chrono::high_resolution_clock::now();
        int i = 0;
        std::srand(1234);
        std::random_shuffle(whole_data_with_gapped_pos.begin(), whole_data_with_gapped_pos.end());
        //auto random_test_data = whole_data_with_gapped_pos;

        // std::vector<std::pair<KeyType, std::vector<KeyType>>> random_test_data;
        std::vector<KeyType> random_test_data;
        random_test_data.reserve(whole_data_with_gapped_pos.size());
        std::vector<KeyType> dummy_payload {1};
        for (auto data_pair : whole_data_with_gapped_pos){
            //random_test_data.emplace_back(std::make_pair(data_pair.first, dummy_payload));
            random_test_data.emplace_back(data_pair.first);
        }
        // test prediction stage
        for(auto it = random_test_data.begin(); it != random_test_data.end(); it++, i++){
            //Pos predicted_pos = predict_position((*it).first);
            Pos predicted_pos = predict_position(*it);
            all_predicted_pos.emplace_back(predicted_pos);
        }
        assert(all_predicted_pos.size() == evaluated_data_size);
        upper_bound_iter_count_ = 0;
        lower_bound_iter_count_ = 0;
        binary_search_len_ = 0;
        i = 0;
        KeyType res;
        // test correction stage
        auto t1 = std::chrono::high_resolution_clock::now();
        for(auto it = random_test_data.begin(); it != random_test_data.end(); it++, i++){
            //res = correct_query_in_gapped_array(all_predicted_pos[i], (*it).first);
            // assert(isEqual(res, (*it).first)); // in static case, we should find the keys have been seen
            res = correct_query_in_gapped_array(all_predicted_pos[i], (*it));
            assert(isEqual(res, *it)); // in static case, we should find the keys have been seen
        }
        auto t2 = std::chrono::high_resolution_clock::now();
        //lookup-time per query
        auto predict_time = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count() / evaluated_data_size;
        auto correct_time = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() / evaluated_data_size;
        //size
        size_t size_of_all_segments = this->size_segments_bytes();
        // one virtual data used as placeholder
        size_t size_of_gaps = (gapped_array_with_linking_array.size() - number_of_non_gap) * sizeof(KeyType);
        size_t size_of_payload = random_test_data.size() * payload_size; //  size of actual data

        std::cout<< "predict time: " << predict_time << std::endl;
        std::cout<< "correct time: " << correct_time << std::endl;
        std::cout<< "overall query time: " << predict_time + correct_time << std::endl;
        std::cout<< "number of all segments: " << learned_segments_.size() << std::endl;
        std::cout<< "size of all segments: " << size_of_all_segments << std::endl;
        std::cout<< "size of gaps: " << size_of_gaps<< std::endl;
        std::cout<< "size of payloads: " << size_of_payload<< std::endl;
        std::cout<< "total size: " << (size_of_payload + size_of_all_segments + size_of_gaps) << std::endl;
        std::cout<< "Upper and lower iter counts in exp_search are: " << double_t(upper_bound_iter_count_)/ data_size_
                 << ", " << double_t(lower_bound_iter_count_)/data_size_ << std::endl;
        std::cout<< "Binary search length in exp_search is: " << double_t(binary_search_len_)/ data_size_ << std::endl;
    }



   /**
    * evaluate the index equiped with gapped array + linking array,
    * including the query time (prediction and correction) and index size (size of segements, gaps, and payloads)
    *
    */
    void evaluate_index_with_gapped_array(std::vector<key_type_transformed>  whole_data,
                                          int payload_size) {
        std::vector<Pos> all_predicted_pos;
        auto evaluated_data_size = whole_data.size();
        all_predicted_pos.reserve(evaluated_data_size);
        auto t0 = std::chrono::high_resolution_clock::now();
        int i = 0;
        std::srand(1234);
        std::random_shuffle(whole_data.begin(), whole_data.end());

        std::vector<KeyType> random_test_data;
        random_test_data.reserve(whole_data.size());
        std::vector<KeyType> dummy_payload {1};
        for (auto key : whole_data){
            random_test_data.emplace_back(key);
        }
        // test prediction stage
        for(auto it = random_test_data.begin(); it != random_test_data.end(); it++, i++){
            //Pos predicted_pos = predict_position((*it).first);
            Pos predicted_pos = predict_position(*it);
            all_predicted_pos.emplace_back(predicted_pos);
        }
        assert(all_predicted_pos.size() == evaluated_data_size);
        upper_bound_iter_count_ = 0;
        lower_bound_iter_count_ = 0;
        binary_search_len_ = 0;
        i = 0;
        KeyType res;
#ifdef Debug
        auto debug = correct_query_in_gapped_array(all_predicted_pos[370], random_test_data[370]);
#endif
        // test correction stage
        auto t1 = std::chrono::high_resolution_clock::now();
        for(auto it = random_test_data.begin(); it != random_test_data.end(); it++, i++){
            //res = correct_query_in_gapped_array(all_predicted_pos[i], (*it).first);
            // assert(isEqual(res, (*it).first)); // in static case, we should find the keys have been seen
            res = correct_query_in_gapped_array(all_predicted_pos[i], (*it));
            assert(isEqual(res, *it)); // in static case, we should find the keys have been seen
        }
        auto t2 = std::chrono::high_resolution_clock::now();
        //lookup-time per query
        auto predict_time = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count() / evaluated_data_size;
        auto correct_time = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() / evaluated_data_size;
        //size
        size_t size_of_all_segments = this->size_segments_bytes();
        // one virtual data used as placeholder
        size_t size_of_gaps = gap_num_ * sizeof(KeyType);
        size_t size_of_payload = random_test_data.size() * payload_size; //  size of actual data

        std::cout<< "predict time: " << predict_time << std::endl;
        std::cout<< "correct time: " << correct_time << std::endl;
        std::cout<< "overall query time: " << predict_time + correct_time << std::endl;
        std::cout<< "number of all segments: " << learned_segments_.size() << std::endl;
        std::cout<< "size of all segments: " << size_of_all_segments << std::endl;
        std::cout<< "size of gaps: " << size_of_gaps<< std::endl;
        std::cout<< "size of payloads: " << size_of_payload<< std::endl;
        std::cout<< "total size: " << (size_of_payload + size_of_all_segments + size_of_gaps) << std::endl;
        std::cout<< "Upper and lower iter counts in exp_search are: " << double_t(upper_bound_iter_count_)/ data_size_
                 << ", " << double_t(lower_bound_iter_count_)/data_size_ << std::endl;
        std::cout<< "Binary search length in exp_search is: " << double_t(binary_search_len_)/ data_size_ << std::endl;
        dlib::matrix<double, 1, 4> results;
        std::vector<size_t> all_true_pos, all_predicted_pos_without_metainfo;
        all_true_pos.reserve(all_predicted_pos.size());
        all_predicted_pos_without_metainfo.reserve(all_predicted_pos.size());
        i = 0;
        for(auto it = random_test_data.begin(); it != random_test_data.end(); it++, i++){
            size_t true_pos = true_pos_in_gapped_array(all_predicted_pos[i], (*it));
            all_true_pos.emplace_back(true_pos);
            all_predicted_pos_without_metainfo.emplace_back(all_predicted_pos[i].pos);
        }
        results =  evaluate_regression_ML(all_predicted_pos_without_metainfo, all_true_pos);
        std::cout<< "ML oriented matricx: " << results;

    }



    void print_segments_statistics(){
        std::vector<double_t > slopes;

        for (auto x : learned_segments_){
            slopes.emplace_back(x.seg_slope);
        }
        std::cout<< "The segment number is: " << learned_segments_.size() << "; statistics of slopes: ";
        calculate_mean_std(slopes, true);
        return;
    }



    void evaluate_indexer_split_by_segments() {
        std::vector<Pos> all_predicted_pos;
        auto t0 = std::chrono::high_resolution_clock::now();
        int i = 0;
        for(Iterator it = first_key_iter_; it != last_key_iter_; it++, i++){
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
        std::vector<size_t> all_true_pos_within, all_true_pos_outof_seg,
                all_predicted_pos_without_metainfo_within_seg, all_predicted_pos_without_metainfo_outof_seg;
        dlib::matrix<double, 1, 4> results;
        for (size_t i=0; i < data_size_; i++){
            size_t key = (*(first_key_iter_ + i)).first;
            if (check_within_segment(key)){
                all_true_pos_within.push_back((*(first_key_iter_ + i)).second);
                size_t predi_pos = all_predicted_pos[i].pos;
                size_t pos_delta = predi_pos >= i ? (predi_pos - i) : (i - predi_pos);
                if (pos_delta > error_){
                    Pos err_predicted_pos = predict_position((*(first_key_iter_+i)).first);
                }
                all_predicted_pos_without_metainfo_within_seg.push_back(predi_pos);
            }else{
                all_true_pos_outof_seg.push_back((*(first_key_iter_ + i)).second);
                size_t predi_pos = all_predicted_pos[i].pos;
                size_t pos_delta = predi_pos >= i ? (predi_pos - i) : (i - predi_pos);
                if (pos_delta > error_){
                    Pos err_predicted_pos = predict_position((*(first_key_iter_+i)).first);
                }
                all_predicted_pos_without_metainfo_outof_seg.push_back(predi_pos);
            }
        }
        std::cout<< "The size of within and outof are: " << all_true_pos_within.size() << ", "
                << all_true_pos_outof_seg.size() << std::endl;
        results = evaluate_regression_ML(all_predicted_pos_without_metainfo_within_seg, all_true_pos_within, 100000);
        std::cout<< "ML oriented matricx for keys within segments: " << results;
        results = evaluate_regression_ML(all_predicted_pos_without_metainfo_outof_seg, all_true_pos_outof_seg, 100000);
        std::cout<< "ML oriented matricx for keys out of segments:: " << results;
        std::vector<size_t> heads, tails;
        for (auto &seg : segments_head_tail_){
            heads.emplace_back(seg.first);
            tails.emplace_back(seg.second);
        }
        results = evaluate_regression_ML(heads, tails, 1000000);
        std::cout<< "ML oriented matricx for segments head and tail: " << results<< std::endl;
    }





    void evaluate_indexer_split_by_sampled_keys() {
        std::vector<Pos> all_predicted_pos;
        auto t0 = std::chrono::high_resolution_clock::now();
        int i = 0;
        for(Iterator it = first_key_iter_; it != last_key_iter_; it++, i++){
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
        for (size_t i=0; i < data_size_; i++){
            size_t key = (*(first_key_iter_ + i)).first;
            if (sampled_keys.find(key) != sampled_keys.end()){
                all_true_pos_sampled.push_back((*(first_key_iter_ + i)).second);
                size_t predi_pos = all_predicted_pos[i].pos;
                size_t pos_delta = predi_pos >= i ? (predi_pos - i) : (i - predi_pos);
                if (pos_delta > error_){
                    Pos err_predicted_pos = predict_position((*(first_key_iter_+i)).first);
                }
                all_predicted_pos_without_metainfo_within_seg.push_back(predi_pos);
            }else{
                all_true_pos_non_sampled.push_back((*(first_key_iter_ + i)).second);
                size_t predi_pos = all_predicted_pos[i].pos;
                size_t pos_delta = predi_pos >= i ? (predi_pos - i) : (i - predi_pos);
                if (pos_delta > error_){
                    Pos err_predicted_pos = predict_position((*(first_key_iter_+i)).first);
                }
                all_predicted_pos_without_metainfo_non_sampled.push_back(predi_pos);
            }
        }
        std::cout << "The size of sampled and none-sampled are: " << all_true_pos_sampled.size() << ", "
                    << all_true_pos_non_sampled.size() << std::endl;
        results = evaluate_regression_ML(all_predicted_pos_without_metainfo_within_seg, all_true_pos_sampled, 100000);
        std::cout<< "ML oriented matricx for sampled keys : " << results;
        results = evaluate_regression_ML(all_predicted_pos_without_metainfo_non_sampled, all_true_pos_non_sampled, 100000);
        std::cout<< "ML oriented matricx for non-sampled keys:: " << results;
        std::vector<size_t> heads, tails;
        for (auto &seg : segments_head_tail_){
            heads.emplace_back(seg.first);
            tails.emplace_back(seg.second);
        }
        results = evaluate_regression_ML(heads, tails, 1000000);
        std::cout<< "ML oriented matricx for segments head and tail: " << results<< std::endl;
    }





    /**
    * Returns the size in bytes of the recursive indexing data structure, including the last-level segments.
    * @return the size in bytes of the indexing data structure
    */
    size_t size_segments_bytes() const {
        auto total = 1;
        for (auto &l : layers_)
            total += l.size();
        return total * sizeof(segment_type);
    }

   /**
    * Returns the size in bytes of the payload, i.e.,  the data size restored in the last-level segments.
    * @return the size in bytes of the data payloads.
    */
    size_t size_payloads_bytes(int payload_size) const {
        return this->segments_count() * payload_size;
    }

    /**
     * Returns the number of segments in the last_key_iter layer of the approximate index.
     * @return the number of segments
     */
    size_t segments_count() const {
        return layers_.empty() ? 1 : layers_.back().size();
    }

    /**
     * Returns the number of layers of the recursive segmentation.
     * @return the number of layers of the recursive segmentation
     */
    size_t height() const {
        return 1 + layers_.size();
    }



    /** the layer struct that contains a sequence of segments, used for the recurisive organization of PGM-Index.
     *  each layer splits every segment into two parts: key and data (including slope and intercept)
     */
    struct Layer {
        std::vector<KeyType> segments_keys;
        std::vector<SegmentData<Floating>> segments_data;

        inline size_t size() const {
            return segments_keys.size();
        }

        template<typename S>
        explicit Layer(const S & learned_segments, size_t data_size) {
            segments_keys.reserve(data_size);
            segments_data.reserve(data_size + 1);
            for (auto &s : learned_segments) {
                segments_keys.push_back(s.seg_start);
                segments_data.emplace_back(s.seg_slope, s.seg_intercept);
            }
            segments_data.emplace_back(0, data_size);
        }
    };


    void out_segments_spans(std::string out_name);
    Pos predict_position(KeyType key) const;

    std::vector<segment_type> learned_segments_;
    size_t data_size_;

protected:
    std::reverse_iterator<typename std::vector<std::pair<KeyType, std::vector<KeyType>>>::iterator> query_tmp_iter_;
    std::vector<Layer> layers_;
    size_t error_;   ///< the maximum allowed error in the last level of the PGM index
    size_t recursive_error_;   ///< the maximum allowed error in the upper level of the PGM index
    ///< The number of elements in the data.
    Iterator first_key_iter_;    ///< The iterator of the smallest element in the data.
    Iterator last_key_iter_;    ///< The (iterator + 1) of the largest element in the data.

    size_t upper_bound_iter_count_; // statistic for the exp_search
    size_t lower_bound_iter_count_; // statistic for the exp_search
    long binary_search_len_; // statistic for the exp_search
    size_t gap_num_;



    std::vector<segment_type> learn_segments(Iterator first_iter, Iterator last_iter, size_t error);
    void organize_segments(std::string strategy, size_t recursive_err);
    inline bool check_within_segment(KeyType key);
    KeyType correct_position(Pos predicted_pos, KeyType key);

    KeyType correct_query_in_gapped_array(Pos const & predicted_pos, KeyType const & key);
    KeyType correct_query_in_gapped_array(size_t const & predicted_pos, KeyType const & key);
    KeyType correct_query_in_gapped_array(Pos const & predicted_pos,
            std::pair<KeyType, std::vector<KeyType>> const & key_pair);

    size_t true_pos_in_gapped_array(Pos const & predicted_pos, KeyType const & key);


    void complete_segments();



    /**
     * Finds the last-level segment responsible for the given key by searching in the recursive PGM-index structure.
     * @param key the value to search for
     * @return the segment responsible for the given key
     */
    inline const segment_type find_segment_for_key(KeyType key) const {
        auto slope = root_.seg_slope;
        auto intercept = root_.seg_intercept;
        auto node_key = root_.seg_start;
        auto end = root_.seg_end;
        size_t approx_pos = std::min(root_(key), root_limit_);
        size_t pos = 0;

        for (auto &it : layers_) {
            auto layer_size = it.size();

            auto lo = SUB_ERR(approx_pos, recursive_error_, layer_size);
            auto hi = ADD_ERR(approx_pos, recursive_error_ + 1, layer_size);
            // adjust the lo and hi due to the precision lost in the conversion from float to size_t.
            while (lo != 0 and it.segments_keys[lo] > key){
                lo -= 1;
            }
            while (hi < (layer_size - 1) and key > it.segments_keys[hi]){
                hi += 1;
            }
            //assert(it.segments_keys[lo] <= key);
            //assert(key <= it.segments_keys[hi] || key > it.segments_keys[layer_size - 1]);

            if (recursive_error_ >= BIN_SEARCH_THRESHOLD) { // use binary search for large "pages"
                auto layer_begin = it.segments_keys.cbegin();
                auto lo_it = layer_begin + lo;
                auto hi_it = layer_begin + hi;
                auto pos_it = std::lower_bound(lo_it, hi_it + 1, key);
                pos = (size_t) std::distance(layer_begin, pos_it);
                if (layer_begin[pos] > key && pos != lo)
                    pos--;
            } else {
                for (; lo <= hi && it.segments_keys[lo] <= key; ++lo);
                pos = lo - 1;
            }

            node_key = it.segments_keys[pos];
            approx_pos = it.segments_data[pos](key - node_key);
            slope = it.segments_data[pos].seg_slope;
            intercept = it.segments_data[pos].seg_intercept;
            //end = it.segments_data[pos].seg_end;
            if (pos + 1 <= it.size())
                approx_pos = std::min(approx_pos, (size_t) it.segments_data[pos + 1].seg_intercept);

            //assert(node_key <= key);
            //assert(pos + 1 >= hi || it.segments_keys[pos + 1] > key);
        }

        return {node_key, slope, intercept, 0, 0};
    }

};



/**
 * Learn the index of query. In PGM indexer, the index are segments including keys, slopes and intercepts.
 */
template<typename KeyType, typename Pos, typename Floating, typename Iterator>
std::vector<SegmentModified<KeyType, Floating>>
        PgmIndexerModified<KeyType, Pos, Floating, Iterator>::learn_segments(
                const Iterator first_iter, const Iterator last_iter, size_t err) {
    std::vector<segment_type> segments;
    segments.reserve(8192);

    size_t input_data_size = std::distance(first_iter, last_iter);
    std::cout<< "data size: " << input_data_size << std::endl;
    if (input_data_size == 0)
        return segments;
    //assert(std::is_sorted(first_iter, last_iter, CompareForDataPair<KeyType>()));
    for (int i = 0 ; i < input_data_size - 1; i ++){
        assert(first_iter[i+1].first >= first_iter[i].first);
    }


    PGMMechanism<KeyType, size_t> mechanism(err, err);
    mechanism.add_point((*first_iter).first, (*first_iter).second);
    KeyType key = (*first_iter).first;
    KeyType seg_first_y = (*first_iter).second;
    Iterator it = std::next(first_iter);
    Iterator segment_end_iter;

    for (size_t i = 1; i < input_data_size; ++i, ++it) {
        if ((*it).first == (*std::prev(it)).first)
            continue;

        if (!mechanism.add_point((*it).first, (*it).second)) {
            auto intercept = mechanism.get_intercept(key);
            //auto[min_slope, max_slope] = mechanism.get_slope_range();
            std::pair<float, float> slope_range;
            slope_range = mechanism.get_slope_range();
            auto slope = 0.5 * (slope_range.first + slope_range.second);
            segment_end_iter = it == first_iter ? first_iter : std::prev(it);
            KeyType seg_last = (*segment_end_iter).first;
            if (segments_head_tail_.count(key) == 0){
                segments_head_tail_[key] = seg_last;
            }
            //auto slope = 0.5 * (min_slope + max_slope);
            segments.emplace_back(key, slope, intercept, seg_last, (*it).second - seg_first_y);
            key = (*it).first;
            seg_first_y = (*it).second;
            --i;
            --it;
        }
    }

    // Last segment
    auto intercept = mechanism.get_intercept(key);
    std::pair<float, float> slope_range;
    slope_range = mechanism.get_slope_range();
    auto slope = 0.5 * (slope_range.first + slope_range.second);
    //auto[min_slope, max_slope] = mechanism.get_slope_range();
    //auto slope = 0.5 * (min_slope + max_slope);
    segment_end_iter = it == first_iter ? first_iter : std::prev(it);
    KeyType seg_last = (*segment_end_iter).first;
    if (segments_head_tail_.count(key) == 0){
        segments_head_tail_[key] = seg_last;
    }
    segments.emplace_back(key, slope, intercept, seg_last, (*segment_end_iter).second - seg_first_y);

    return segments;
}


template<typename KeyType, typename Pos, typename Floating, typename Iterator>
void PgmIndexerModified<KeyType, Pos, Floating, Iterator>::complete_segments() {
    KeyType cur_last, cur_next_first;

    std::vector<std::pair<KeyType, std::pair<Floating, size_t>>> pairs;
    size_t i = 0;
    auto first_seg = *(learned_segments_.begin());
    auto first_key = (*first_key_iter_).first;
    if (first_key < (first_seg.seg_start)){
        long cur_last_y, cur_next_first_y;
        cur_next_first_y = first_seg.seg_intercept;
        cur_last_y = 0;
        cur_next_first = first_seg.seg_start;
        double_t slope = double_t(cur_next_first_y - cur_last_y)  / double_t(cur_next_first - first_key);
        completed_learned_segments_.emplace_back(first_key, slope, cur_last_y, cur_next_first, 2);
    }
    for (auto it = learned_segments_.begin(); it != learned_segments_.end() - 1; it++, i++) {
        cur_last = (*it).seg_end;
        cur_next_first = (*(it+1)).seg_start;
        completed_learned_segments_.emplace_back(*it);
        if (cur_last != cur_next_first){
            long cur_last_y, cur_next_first_y;
            cur_next_first_y = (*(it+1)).seg_intercept;
            float tmp = (*(it))(cur_last);
            cur_last_y = round(tmp);
            double_t slope = double_t(cur_next_first_y - cur_last_y)  / double_t(cur_next_first - cur_last);
            completed_learned_segments_.emplace_back(cur_last, slope, cur_last_y, cur_next_first, 2);
        }
        pairs.emplace_back((*it).seg_start, std::make_pair((*it).seg_slope, (*it).seg_intercept));
    }
    completed_learned_segments_.emplace_back(*(learned_segments_.end()-1));

}



/*
 * Output spans of each segments
 */
template<typename KeyType, typename Pos, typename Floating, typename Iterator>
void PgmIndexerModified<KeyType, Pos, Floating, Iterator>::out_segments_spans(std::string out_path) {
    std::vector<KeyType> delta_x_of_each_seg;

    int cnt = 0, cnt_of_special_seg = 0;
    for (auto it = learned_segments_.begin(); it != learned_segments_.end() - 1; it++) {
        KeyType delta_x = (*it).seg_end - (*it).seg_start;
        //if (delta_x < 7000 and delta_x > 5000){
        //if (delta_x > 1000000){
        if (delta_x < 1000000 and delta_x > 100000){
            std::cout<<"from "<<(*it).seg_start<< " to "<< (*it).seg_end;
            std::cout<<"; #keys of this seg is: " << (*it).number_of_seg_keys;
            std::cout<<"; slope of this seg is: " << (*it).seg_slope << std::endl;
            cnt ++;
            if ((*it).number_of_seg_keys == 2){
                cnt_of_special_seg ++;
            }
        }
        delta_x_of_each_seg.emplace_back(delta_x);
    }
    std::cout<<"Filtered number is: " << cnt << std::endl;
    std::cout<<"Specially filtered segment number is: " << cnt_of_special_seg << std::endl;


    write_vector_to_f(delta_x_of_each_seg, out_path);

}




template<typename KeyType, typename Pos, typename Floating, typename Iterator>
void PgmIndexerModified<KeyType, Pos, Floating, Iterator>::organize_segments(std::string strategy, size_t recursive_err) {
    recursive_error_ = recursive_err;
    if (strategy == "Recursive"){
        if (learned_segments_.empty()) {
            root_ = segment_type(0, 0, 0, 0, 0);
            root_limit_ = 1;
            return;
        }

        if (learned_segments_.size() == 1) {
            root_ = segment_type(learned_segments_[0]);
            root_limit_ = data_size_;
            return;
        }


        std::list<Layer> tmp_layers;
        tmp_layers.emplace_front(Layer(learned_segments_, data_size_));
        std::vector<segment_type> recursived_learened_segments;

        while (tmp_layers.front().size() > 1) {
            //std::vector<KeyType> tmp_keys = tmp_layers.front().segments_keys;
            std::vector<std::pair<KeyType, size_t >> tmp_keys;
            std::vector<KeyType> tmp_keys_without_pos = tmp_layers.front().segments_keys;
            // the index does not matter for the non-last layer
            size_t tmp_idx = 0;
            for (auto key : tmp_keys_without_pos){
                tmp_keys.emplace_back(std::make_pair(key, tmp_idx));
                tmp_idx ++;
            }
            recursived_learened_segments = learn_segments(tmp_keys.begin(), tmp_keys.end(), recursive_error_);
            Layer tmp_l = Layer(recursived_learened_segments, tmp_keys.size());
            tmp_layers.emplace_front(tmp_l);
        }

        root_ = recursived_learened_segments[0];
        root_limit_ = recursived_learened_segments.size();
        layers_ = {std::make_move_iterator(std::next(tmp_layers.begin())), std::make_move_iterator(tmp_layers.end())};
    }


}



template<typename KeyType, typename Pos, typename Floating, typename Iterator>
inline bool PgmIndexerModified<KeyType, Pos, Floating, Iterator>::check_within_segment(KeyType key) {
    /**
      * Returns the approximate position of a key.
      * @param key the value to search for
      * @return a struct with the approximate position
      * @see approx_pos_t
      */
    auto segment = this->find_segment_for_key(key);
    KeyType seg_head = segment.seg_start;
    KeyType seg_tail = segments_head_tail_[seg_head];

    if (key >= seg_head and key <= seg_tail){
        return true;
    } else{
        return false;
    }
}




template<typename KeyType, typename Pos, typename Floating, typename Iterator>
inline Pos PgmIndexerModified<KeyType, Pos, Floating, Iterator>::predict_position(KeyType key) const {
/**
  * Returns the approximate position of a key.
  * @param key the value to search for
  * @return a struct with the approximate position
  * @see approx_pos_t
  */
        //if (UNLIKELY(key < *(this->first_key_iter_)))
        if (UNLIKELY(key < (*(this->first_key_iter_)).first))
            return {0, 0, 0};
        //if (UNLIKELY(key > *std::prev((this->last_key_iter_))))
        if (UNLIKELY(key > (*std::prev((this->last_key_iter_))).first))
            return {data_size_ - 1, data_size_ - 1, data_size_ - 1};

        auto segment = this->find_segment_for_key(key);
        auto pos = segment(key);
        auto lo = SUB_ERR(pos, error_, data_size_);
        auto hi = ADD_ERR(pos, error_, data_size_);
        if (UNLIKELY(pos > hi))
            pos = hi;

        return {pos, lo, hi};
    }


template<typename KeyType, typename Pos, typename Floating, typename Iterator>
inline KeyType PgmIndexerModified<KeyType, Pos, Floating, Iterator>::correct_position(Pos predicted_pos, KeyType key) {
    // error_bounded binary search
    //auto lo = this->first_key_iter_ + predicted_pos.lo;
    //auto hi = this->first_key_iter_ + predicted_pos.hi;
    //auto res_iter = std::lower_bound(lo, hi, key, CompareForDataPair<KeyType>());

    // exponential search, for sampling and dynamic case that violate the error bounds.
    auto res_iter = exponential_search(this->first_key_iter_, this->last_key_iter_,
            key, predicted_pos.pos, CompareForDataPair<KeyType>(),
                    upper_bound_iter_count_, lower_bound_iter_count_, binary_search_len_);
    // return std::distance(this->first_key_iter_, res_iter); // the true indexed position of query ``key''
    return (*res_iter).first; // the true indexed position of query ``key''
}





template<typename KeyType, typename Pos, typename Floating, typename Iterator>
inline KeyType PgmIndexerModified<KeyType, Pos, Floating, Iterator>::correct_query_in_gapped_array(
        Pos const & predicted_pos,
        std::pair<KeyType, std::vector<KeyType>> const & key_pair) {
    KeyType key = key_pair.first;
    // auto & res_iter = exponential_search_total_order(
    //         gapped_array_with_linking_array_.begin(), gapped_array_with_linking_array_.end(),
    //         key_pair, predicted_pos.pos, GreaterComparatorForGappedArray<KeyType>());
    exponential_search_total_order(query_tmp_iter_,
                                   gapped_array_with_linking_array_.begin(), gapped_array_with_linking_array_.end(),
                                   key_pair, predicted_pos.pos, GreaterComparatorForGappedArray<KeyType>(),
                                   upper_bound_iter_count_, lower_bound_iter_count_, binary_search_len_);
    // found in the gapped array
    //if ((*res_iter).first == key){
    if ((*query_tmp_iter_).first == key){
        return key;
    } else {
        // need to check the linking array, since res_iter is returned by lower_bound, i.e., >=,
        //auto linking_array = (*res_iter).second;
        auto & linking_array = (*query_tmp_iter_).second;
        int i = 0;
        while (i < linking_array.size()){
            if (isEqual(linking_array[i], key)){
                return key;
            }
            i++;
        }
        return NULL; // not found
    }
}


template<typename KeyType, typename Pos, typename Floating, typename Iterator>
inline size_t PgmIndexerModified<KeyType, Pos, Floating, Iterator>::true_pos_in_gapped_array(
        Pos const & predicted_pos, KeyType const & key) {
    // decrementing 1 since we need found the last item <= key,
    // which is equivalent to the last item returned by upper_bound
    auto iter_upper_bound_minus_one = upper_bound_exp_search_total_order(
            gapped_array_.begin(), gapped_array_.end(),
            key, predicted_pos.pos, LessComparatorForGappedArray<KeyType>(),
            upper_bound_iter_count_, lower_bound_iter_count_) - 1; // "-1" means the last item by upper_bound
    auto iter_pos = std::distance(gapped_array_.begin(), iter_upper_bound_minus_one);
    return iter_pos;
}



template<typename KeyType, typename Pos, typename Floating, typename Iterator>
inline KeyType PgmIndexerModified<KeyType, Pos, Floating, Iterator>::correct_query_in_gapped_array(
        Pos const & predicted_pos, KeyType const & key) {
    // decrementing 1 since we need found the last item <= key,
    // which is equivalent to the last item returned by upper_bound
    auto iter_upper_bound_minus_one = upper_bound_exp_search_total_order(
            gapped_array_.begin(), gapped_array_.end(),
            key, predicted_pos.pos, LessComparatorForGappedArray<KeyType>(),
            upper_bound_iter_count_, lower_bound_iter_count_) - 1; // "-1" means the last item by upper_bound
#ifdef Debug
    auto iter_pos = std::distance(gapped_array_.begin(), iter_upper_bound_minus_one);
    auto pre = *(iter_upper_bound_minus_one - 1);
    auto res = (*iter_upper_bound_minus_one);
    assert(res >= pre);
    assert(res <= key);
    if (res != key){ // linking_array
        assert(linking_array_[iter_pos].size() > 1);
    }
#endif
    // found in the gapped array
    if (*iter_upper_bound_minus_one == key){
        return key;
    } else {
        // need to check the linking array, since res_iter is returned by lower_bound, i.e., >=,
        //auto linking_array = (*res_iter).second;
        auto & linking_array = linking_array_[std::distance(gapped_array_.begin(), iter_upper_bound_minus_one)];
        for (auto key_in_linking: linking_array){
            if (isEqual(key_in_linking, key)){
                return key;
            }
        }
        return key-1; // Not found, return a result that is not equal to key

        // if (linking_array.size() >= binary_search_threshold_){
        //     auto pos_iter_in_linking_array = std::lower_bound(linking_array.begin(), linking_array.end(), key);
        //     return (*pos_iter_in_linking_array);
        // } else{
        //     for (auto key_in_linking: linking_array){
        //         if (key_in_linking == key){
        //             return key;
        //         }
        //     }
        //     return NULL;
        // }
    }
}


template<typename KeyType, typename Pos, typename Floating, typename Iterator>
inline KeyType PgmIndexerModified<KeyType, Pos, Floating, Iterator>::correct_query_in_gapped_array(
        size_t const & predicted_pos, KeyType const & key) {
    // decrementing 1 since we need found the last item <= key,
    // which is equivalent to the last item returned by upper_bound
    auto iter_upper_bound_minus_one = upper_bound_exp_search_total_order(
            gapped_array_.begin(), gapped_array_.end(),
            key, predicted_pos, LessComparatorForGappedArray<KeyType>(),
            upper_bound_iter_count_, lower_bound_iter_count_) - 1; // "-1" means the last item by upper_bound
    // found in the gapped array
    if (*iter_upper_bound_minus_one == key){
        return key;
    } else {
        // need to check the linking array, since res_iter is returned by lower_bound, i.e., >=,
        //auto linking_array = (*res_iter).second;
        auto & linking_array = linking_array_[std::distance(gapped_array_.begin(), iter_upper_bound_minus_one)];
        for (auto key_in_linking: linking_array){
            if (isEqual(key_in_linking, key)){
                return key;
            }
        }
        return key-1; // Not found, return a result that is not equal to key

    }
}







#endif //LEARNED_INDEX_PGMINDEXER_MODIFIED_HPP
