//
// Implementation of PGM Indexer
//
// Paolo Ferragina and Giorgio Vinciguerra. The PGM-index: a fully-dynamic compressed learned index with provable
// worst-case bounds. PVLDB, 13(8): 1162-1175, 2020.
//


#ifndef LEARNED_INDEX_PGMINDEXER_HPP
#define LEARNED_INDEX_PGMINDEXER_HPP



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
 * A struct that stores the parameters (slope and intercept) of a segment.
 * @tparam Floating the floating-point type of the segment's parameters
 */
template<typename Floating>
struct SegmentData {
    //static_assert(std::is_floating_point<Floating>());
    Floating seg_slope;     ///< The slope of the segment.
    Floating seg_intercept; ///< The intercept of the segment.

    SegmentData(Floating slope, Floating intercept) : seg_slope(slope), seg_intercept(intercept) {};

    template<typename K>
    inline size_t operator()(K k) const {
        Floating pos = seg_slope * k + seg_intercept;
        return pos > Floating(0) ? round(pos) : 0ul;
    }
};

/**
 * A struct that stores a segment.
 * @tparam KeyType the type of the elements that the segment indexes
 * @tparam Floating the floating-point type of the segment's parameters
 */
template<typename KeyType, typename Floating>
struct Segment {
    //static_assert(std::is_floating_point<Floating>());
    KeyType seg_key;              ///< The first key that the segment indexes.
    Floating seg_slope;     ///< The slope of the segment.
    Floating seg_intercept; ///< The intercept of the segment.

    Segment() = default;

    /**
     * Constructs a new segment.
     * @param key the first_key_iter key that the segment indexes
     * @param slope the slope of the segment
     * @param intercept the intercept of the segment
     */
    Segment(KeyType key, Floating slope, Floating intercept) : seg_key(key), seg_slope(slope), seg_intercept(intercept) {};

    friend inline bool operator<(const Segment &s, const KeyType k) {
        return s.seg_key < k;
    }

    friend inline bool operator<(const Segment &s1, const Segment &s2) {
        return s1.seg_key < s2.seg_key;
    }

    /**
     * Returns the approximate position of the specified key.
     * @param k the key whose position must be approximated
     * @return the approximate position of the specified key
     */
    inline size_t operator()(KeyType k) const {
        Floating pos = seg_slope * (k - seg_key) + seg_intercept;
        return pos > Floating(0) ? round(pos) : 0ul;
    }
};


template<typename KeyType, typename Pos, typename Floating, typename Iterator>
class PgmIndexer {
    using floating_type = Floating;
    using segment_type = Segment<KeyType, Floating>;
    using segment_data_type = SegmentData<Floating>;
    segment_type root_;
    size_t root_limit_;
    std::vector<segment_type> learned_segments_;
    std::map<KeyType, KeyType> segments_head_tail_;
    std::set<KeyType> sampled_keys;

public:
    PgmIndexer(Iterator first_key_iter, Iterator last_key_iter, size_t data_size) :
            first_key_iter_(first_key_iter), last_key_iter_(last_key_iter), data_size_(data_size){};

    void learn_index(Iterator first_iter, Iterator last_iter, size_t error, std::string strategy,
            size_t recursive_err, double sample_rate = 1.0, bool complete_segments = false) {
        std::vector<std::pair<KeyType, size_t>> sampled_data;
        this->error_ = error;
        this->recursive_error_ = recursive_err;
        if (sample_rate != 1.0){
            size_t sample_size = round(data_size_ * sample_rate);
            //std::sample(first_iter, last_iter, std::back_inserter(sampled_data), sample_size, std::mt19937{std::random_device{}()});
            std::experimental::sample(first_iter, last_iter, std::back_inserter(sampled_data), sample_size, std::mt19937{std::random_device{}()});
            std::sort(sampled_data.begin(), sampled_data.end());
            for (auto &x : sampled_data){
                sampled_keys.insert(x.first);
            }
            learned_segments_ = learn_segments(sampled_data.begin(), sampled_data.begin() + sample_size, error);
        } else{
            learned_segments_ = learn_segments(first_iter, last_iter, error);
        }
        organize_segments(strategy, recursive_err);
    }

    KeyType query(KeyType key) {
        auto predicted_pos = predict_position(key);
        KeyType res = correct_position(predicted_pos, key);
        return res;
    }

    void evaluate_indexer(int payload_size, std::string pred_file_name="", bool shuffle=true) {
        std::vector<std::pair<key_type_transformed, size_t>> tmp_data(first_key_iter_, last_key_iter_);
        if (shuffle == true){
            std::srand(1234);
            random_shuffle(tmp_data.begin(), tmp_data.end());
        }

        std::vector<Pos> all_predicted_pos;
        std::vector<long> all_query_times(tmp_data.size()); // used for statistics of the query times, e.g., 99 percent time
        auto t0 = std::chrono::high_resolution_clock::now();
        int i = 0;
        for(Iterator it = tmp_data.begin(); it != tmp_data.end(); it++, i++){
            Pos predicted_pos = predict_position((*it).first);
            all_predicted_pos.emplace_back(predicted_pos);
        }
        assert(all_predicted_pos.size() == data_size_);
        auto t1 = std::chrono::high_resolution_clock::now();
        i = 0;
        size_t wrong_return_count = 0;
        for(Iterator it = tmp_data.begin() ; it != tmp_data.end(); it++, i++){
            auto corrected_res = correct_position(all_predicted_pos[i], (*it).first);
            if (corrected_res != (*it).second) {
                wrong_return_count ++;
            }
        }
        auto t2 = std::chrono::high_resolution_clock::now();
        //lookup-time per query
        auto predict_time = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count() / data_size_;
        auto correct_time = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() / data_size_;

        all_query_times.emplace_back(std::chrono::duration_cast<std::chrono::nanoseconds>(
                t2 - t0).count());
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
        std::sort(all_query_times.begin(), all_query_times.end());
        auto p99_time = all_query_times[int(all_query_times.size()/100*99)];
        std::cout << "p99 time: " << p99_time << std::endl;
        std::cout<< "write_to_file_name: " << pred_file_name << std::endl;
        std::cout<< "predict time: " << predict_time << std::endl;
        std::cout<< "correct time: " << correct_time << std::endl;
        std::cout<< "overall query time: " << predict_time + correct_time << std::endl;
        std::cout<< "number of all segments: " << learned_segments_.size() << std::endl;
        std::cout<< "size of all segments: " << size_of_all_segments << std::endl;
        std::cout<< "size of payloads: " << size_of_payload<< std::endl;
        std::cout<< "total size: " << (size_of_payload + size_of_all_segments) << std::endl;
        std::cout << "wrong return count: " << wrong_return_count << std::endl;
        results =  evaluate_regression_ML(all_predicted_pos_without_metainfo, all_true_pos, error_, write_to_file,
                pred_file_name);
        std::cout<< "ML oriented matricx: " << results<< std::endl;

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
        std::vector<size_t> all_true_pos_within, all_true_pos_outof_seg, all_predicted_pos_without_metainfo_within_seg, all_predicted_pos_without_metainfo_outof_seg;
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
        std::cout<< "The size of within and outof are: " << all_true_pos_within.size() << ", " << all_true_pos_outof_seg.size() << std::endl;
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
        std::vector<size_t> all_true_pos_sampled, all_true_pos_non_sampled, all_predicted_pos_without_metainfo_within_seg, all_predicted_pos_without_metainfo_non_sampled;
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
        std::cout << "The size of sampled and none-sampled are: " << all_true_pos_sampled.size() << ", " << all_true_pos_non_sampled.size() << std::endl;
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
    * Returns the size in bytes including the payloads, i.e.,  the data size restored in the last-level segments.
    * @return the size in bytes including the data payloads.
    */
    size_t size_payloads_bytes(int payload_size) const {
        return this->data_size_ * payload_size;
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
                segments_keys.push_back(s.seg_key);
                segments_data.emplace_back(s.seg_slope, s.seg_intercept);
            }
            segments_data.emplace_back(0, data_size);
        }
    };


protected:
    std::vector<Layer> layers_;
    size_t error_;   ///< the maximum allowed error in the last level of the PGM index
    size_t recursive_error_;   ///< the maximum allowed error in the upper level of the PGM index
    size_t data_size_;    ///< The number of elements in the data.
    Iterator first_key_iter_;    ///< The iterator of the smallest element in the data.
    Iterator last_key_iter_;    ///< The (iterator + 1) of the largest element in the data.


    std::vector<segment_type> learn_segments(Iterator first_iter, Iterator last_iter, size_t error);
    void organize_segments(std::string strategy, size_t recursive_err);
    Pos predict_position(KeyType key) const;
    inline bool check_within_segment(KeyType key);
    KeyType correct_position(Pos predicted_pos, KeyType key);


    /**
     * Finds the last-level segment responsible for the given key by searching in the recursive PGM-index structure.
     * @param key the value to search for
     * @return the segment responsible for the given key
     */
    inline const segment_type find_segment_for_key(KeyType key) const {
        auto slope = root_.seg_slope;
        auto intercept = root_.seg_intercept;
        auto node_key = root_.seg_key, first_node_key(KeyType(0));
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

            // on sampling case, there may be keys smaller than the key of first segment
            first_node_key = it.segments_keys[0];
            if (key <= first_node_key){
                auto first_seg = it.segments_data[0];
                return {first_node_key, first_seg.seg_slope, first_seg.seg_intercept};
            }

            node_key = it.segments_keys[pos];
            approx_pos = it.segments_data[pos](key - node_key);
            slope = it.segments_data[pos].seg_slope;
            intercept = it.segments_data[pos].seg_intercept;
            if (pos + 1 <= it.size())
                approx_pos = std::min(approx_pos, (size_t) it.segments_data[pos + 1].seg_intercept);

            //assert(node_key <= key);
            //assert(pos + 1 >= hi || it.segments_keys[pos + 1] > key);
        }

        return {node_key, slope, intercept};
    }


};



/**
* A struct that stores the result of a query for PGM indexer.
*/
struct PGMPos {
    size_t pos; ///< The approximate position.
    size_t lo;  ///< The lower bound of the range of size no more than 2*error where key can be found.
    size_t hi;  ///< The upper bound of the range of size no more than 2*error where key can be found.
};

/**
 * Learn the index of query. In PGM indexer, the index are segments including keys, slopes and intercepts.
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
std::vector<Segment<KeyType, Floating>>
        PgmIndexer<KeyType, Pos, Floating, Iterator>::learn_segments(Iterator first_iter, Iterator last_iter, size_t err) {
    std::vector<segment_type> segments;
    segments.reserve(8192);

    size_t input_data_size = std::distance(first_iter, last_iter);
    std::cout<< "data size: " << input_data_size << std::endl;
    if (input_data_size == 0)
        return segments;
    assert(std::is_sorted(first_iter, last_iter));


    PGMMechanism<KeyType, size_t> mechanism(err, err);
    mechanism.add_point((*first_iter).first, (*first_iter).second);
    KeyType key = (*first_iter).first;
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
            //auto slope = 0.5 * (min_slope + max_slope);
            segments.emplace_back(key, slope, intercept);
            segment_end_iter = it == first_iter ? first_iter : std::prev(it);
            KeyType seg_last = (*segment_end_iter).first;
            if (segments_head_tail_.count(key) == 0){
                segments_head_tail_[key] = seg_last;
            }
            key = (*it).first;
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
    segments.emplace_back(key, slope, intercept);
    segment_end_iter = it == first_iter ? first_iter : std::prev(it);
    KeyType seg_last = (*segment_end_iter).first;
    if (segments_head_tail_.count(key) == 0){
        segments_head_tail_[key] = seg_last;
    }

    return segments;
}

template<typename KeyType, typename Pos, typename Floating, typename Iterator>
void PgmIndexer<KeyType, Pos, Floating, Iterator>::organize_segments(std::string strategy, size_t recursive_err) {
    recursive_error_ = recursive_err;
    if (strategy == "Recursive"){
        if (learned_segments_.empty()) {
            root_ = segment_type(0, 0, 0);
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
inline bool PgmIndexer<KeyType, Pos, Floating, Iterator>::check_within_segment(KeyType key) {
    /**
      * Returns the approximate position of a key.
      * @param key the value to search for
      * @return a struct with the approximate position
      * @see approx_pos_t
      */
    auto segment = this->find_segment_for_key(key);
    KeyType seg_head = segment.seg_key;
    KeyType seg_tail = segments_head_tail_[seg_head];

    if (key >= seg_head and key <= seg_tail){
        return true;
    } else{
        return false;
    }
}




template<typename KeyType, typename Pos, typename Floating, typename Iterator>
inline Pos PgmIndexer<KeyType, Pos, Floating, Iterator>::predict_position(KeyType key) const {
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
inline KeyType PgmIndexer<KeyType, Pos, Floating, Iterator>::correct_position(Pos predicted_pos, KeyType key) {
    auto lo = this->first_key_iter_ + predicted_pos.lo;
    auto hi = this->first_key_iter_ + predicted_pos.hi;
    auto res_iter = std::lower_bound(lo, hi, key, CompareForDataPair<KeyType>());
    return std::distance(this->first_key_iter_, res_iter); // the true indexed position of query ``key''
}




#endif //LEARNED_INDEX_PGMINDEXER_HPP
