//
// Patched FITingTree Index for sampling case
//


#ifndef LEARNED_INDEX_FITTINGTREEINDEXER_MODIFIED_HPP
#define LEARNED_INDEX_FITTINGTREEINDEXER_MODIFIED_HPP



#include <vector>
#include <list>
#include <chrono>
#include <cstddef>
#include "BtreeIndexer.hpp"
#include "Utilities.hpp"
#include <iterator>
#include <dlib/matrix.h>
#include <dlib/statistics.h>
#include <random>
#include <experimental/algorithm>



//using size_t = unsigned long;


template<typename KeyType, typename Pos, typename Floating, typename Iterator>
class FittingTreeIndexerModified {
    using floating_type = Floating;
    using segment_type = FittingSegmentModified<KeyType, Floating>;
    //using btree_const_iterator = typename stx::btree<KeyType, std::pair<float, size_t>, btreeFriend::btree_traits>::const_iterator;
    //using btree_const_iterator = stx::btree<size_t, std::pair<float, size_t>, std::pair<size_t, std::pair<float, size_t>>,std::less<KeyType>, btreeFriend::btree_traits>::const_iterator;

public:
    std::vector<segment_type> used_learned_segments_;
    std::vector<segment_type> learned_segments_;
    std::vector<segment_type> completed_learned_segments_;
    // organize learned segments: the data is theis slope and intercepet of each segment
    btreeFriend<stx::btree_map_traits<KeyType, std::pair<Floating, Floating>>> btree_storing_segs_;
    std::map<KeyType, KeyType> segments_head_tail_;  // mapping the head key into tail key for each segements

    FittingTreeIndexerModified(Iterator first_key_iter, Iterator last_key_iter, size_t data_size) :
            first_key_iter_(first_key_iter), last_key_iter_(last_key_iter), data_size_(data_size){};

    std::chrono::system_clock::time_point learn_index(const Iterator first_iter, const Iterator last_iter, size_t error,
            double sample_rate = 1.0, bool use_complete_segments = false) {
        std::vector<std::pair<KeyType, size_t>> sampled_data;
        std::chrono::system_clock::time_point t0;
        if (sample_rate != 1.0){
            size_t sample_size = round(data_size_ * sample_rate);
            //std::sample(first_iter, last_iter, std::back_inserter(sampled_data), sample_size, std::mt19937{std::random_device{}()}); // gcc7 +
            std::experimental::sample(first_iter, last_iter, std::back_inserter(sampled_data), sample_size, std::mt19937{std::random_device{}()});
            std::sort(sampled_data.begin(), sampled_data.end());
            t0 = std::chrono::high_resolution_clock::now();
            learned_segments_ = learn_segments(sampled_data.begin(), sampled_data.end(), error);
        } else{
            t0 = std::chrono::high_resolution_clock::now();
            learned_segments_ = learn_segments(first_iter, last_iter, error);
        }
        completed_learned_segments_.clear();
        complete_segments(first_iter);
        if (use_complete_segments){
            used_learned_segments_ = completed_learned_segments_;
        } else{
            used_learned_segments_ = learned_segments_;
        }
        organize_segments();
        return t0;
    }


    std::chrono::system_clock::time_point re_learn_index_segment_wise(
            const Iterator first_iter, const Iterator last_iter) {
        std::chrono::system_clock::time_point t0;

        size_t re_learn_data_size = std::distance(first_iter, last_iter);
        size_t seg_begin_y(0), seg_end_y(0);
        for (int i = 0; i < used_learned_segments_.size(); i++){
            auto seg_i = used_learned_segments_[i];
            // find the begin key idx of the segment
            while ((*(first_iter + seg_begin_y)).first != used_learned_segments_[i].seg_start){
                seg_begin_y++;
                assert(seg_begin_y < re_learn_data_size);
            }
            // find the end key idx of the segment
            seg_end_y = seg_begin_y;
            if (i == (used_learned_segments_.size()-1)){
                seg_end_y = re_learn_data_size - 1;
            }else{
                while ((*(first_iter + seg_end_y)).first != used_learned_segments_[i].seg_end){
                    seg_end_y++;
                }
            }
            auto linear_para = train_linear_model(first_iter + seg_begin_y, first_iter + seg_end_y + 1);
            used_learned_segments_[i].seg_slope = linear_para.second;
            used_learned_segments_[i].seg_intercept = linear_para.first;
        }
        organize_segments();
        return t0;
    }



    void re_organize_segments(bool use_complete_segments=true){
        if (use_complete_segments){
            used_learned_segments_ = completed_learned_segments_;
        } else{
            used_learned_segments_ = learned_segments_;
        }
        organize_segments();
    }

    KeyType query(KeyType key) {
        auto predicted_pos = predict_position(key);
        KeyType res = correct_position(predicted_pos, key);
        return res;
    }

    void complete_seg_last_y(){
        int i(0), j(1);
        std::vector<size_t> seg_last_y;
        int seg_count = used_learned_segments_.size();
        size_t data_size = std::distance(first_key_iter_, last_key_iter_);
        while (j < seg_count){
            auto seg_start = used_learned_segments_[j].seg_start;
            auto data_i = (*(first_key_iter_+i)).first;
            if (seg_start != data_i){
                i ++;
            }
            else{
                //seg_last_x.emplace_back(i - 1);
                size_t last_y = (*(first_key_iter_ + i - 1)).second;
                last_y = (last_y < (data_size - 1)) ? last_y : (data_size - 1);
                seg_last_y.emplace_back(last_y);
                j ++;
            }
        }

        i = 0;
        for (auto it = used_learned_segments_.begin(); it != used_learned_segments_.end(); it++) {
            (*it).seg_last_y = seg_last_y[i];
            i++ ;
        }

    }

    void save_segments(std::string file_name) {
        // int i(0), j(1);
        // std::vector<int> seg_last_x;
        // int seg_count = used_learned_segments_.size();
        // while (j < seg_count){
        //     auto seg_start = used_learned_segments_[j].seg_start;
        //     auto data_i = (*(first_key_iter_+i)).first;
        //     if (seg_start != data_i){
        //         i ++;
        //     }
        //     else{
        //         //seg_last_x.emplace_back(i - 1);
        //         seg_last_x.emplace_back((*(first_key_iter_+i-1)).second);
        //         j ++;
        //     }
        // }
        complete_seg_last_y();


        std::ofstream outFile;
        std::cout<< "write segments to file:" << file_name << std::endl;
        outFile.open(file_name, std::ios::out);
        //i = 0;
        for (auto it = used_learned_segments_.begin(); it != used_learned_segments_.end(); it++) {
            std::stringstream stream;
            stream << std::fixed << std::setw(20) << std::setprecision(6) << (*it).seg_start
                //<< "," << (*it).seg_slope << "," << (*it).seg_intercept << "," <<  seg_last_x[i] << std::endl;
                << "," << (*it).seg_slope << "," << (*it).seg_intercept << "," <<  (*it).seg_last_y <<std::endl;
            //i++ ;
            outFile << stream.str();
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
            //Pos predicted_pos = predict_position(*it);
            Pos predicted_pos = predict_position((*it).first);
            all_predicted_pos.emplace_back(predicted_pos);
            //size_t delta = predicted_pos.pos > i ? predicted_pos.pos - i : i - predicted_pos.pos;
            //if (delta > 1000000){
            //    Pos err_predicted_pos = predict_position((*it).first);
            //}
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        i = 0;
        size_t wrong_return_count = 0;
        for (Iterator it = tmp_data.begin(); it != tmp_data.end(); it++, i++) {
            KeyType corrected_res = correct_position(all_predicted_pos[i], (*it).first);
            if (corrected_res != (*it).first) {
                wrong_return_count ++;
            }
        }

        auto t2 = std::chrono::high_resolution_clock::now();
        //lookup-time per query
        auto predict_time = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count() / data_size_;
        auto correct_time = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() / data_size_;
        //size
        size_t segments_count = used_learned_segments_.size();
        size_t size_of_index = this->index_size_bytes();

        // ML-oriented metric
        // regress_metric = ["RMSE", "MAE", "R2"]
        std::vector<size_t> all_true_pos, all_predicted_pos_without_metainfo;
        for (size_t i=0; i < data_size_; i++){
            //all_true_pos.push_back(i);
            all_true_pos.push_back((*(tmp_data.begin() + i)).second);
            //size_t key = *(first_key_iter_ + i);
            size_t key = (*(tmp_data.begin() + i)).first;
            //size_t predi_pos = round(all_predicted_pos[i](key));
            size_t predi_pos = all_predicted_pos[i].pos;
            size_t pos_delta = predi_pos >= i ? (predi_pos - i) : (i - predi_pos);
            if (pos_delta > error_){
                Pos err_predicted_pos = predict_position((*(tmp_data.begin()+i)).first);
            }
            //assert(pos_delta <= error_); // on down-sampling situation, this assert will fail.
            all_predicted_pos_without_metainfo.push_back(predi_pos);
        }
        dlib::matrix<double, 1, 4> results;
        bool write_to_file = (pred_file_name == "") ? false : true;
        results = evaluate_regression_ML(all_predicted_pos_without_metainfo, all_true_pos,
                                         error_, write_to_file, pred_file_name);

        std::cout<< "predict time: " << predict_time << std::endl;
        std::cout<< "correct time: " << correct_time << std::endl;
        std::cout<< "overall query time: " << predict_time + correct_time << std::endl;
        std::cout << "the number of all segments: " << segments_count << std::endl;
        std::cout << "size of index w/o payloads: " << size_of_index << std::endl;
        std::cout << "size of index including payloads: " << size_of_index + data_size_ * sizeof(KeyType) << std::endl;
        std::cout << "wrong return count: " << wrong_return_count << std::endl;
        std::cout<< "ML oriented matricx: " << results<< std::endl;
    }

    void out_segments_spans(std::string out_path){
        std::vector<KeyType> delta_x_of_each_seg;

        int cnt = 0, cnt_of_special_seg = 0;
        for (auto it = learned_segments_.begin(); it != learned_segments_.end() - 1; it++) {
            KeyType delta_x = (*it).seg_end - (*it).seg_start;
            //if (delta_x < 7000 and delta_x > 5000){
            //if (delta_x > 1000000){
            if (delta_x < 1000000 and delta_x > 100000){
                std::cout<<"from "<<(*it).seg_start<< " to "<< (*it).seg_end;
                std::cout<<"; #keys of this seg is: " << (*it).seg_last_y - (*it).seg_intercept;
                std::cout<<"; slope of this seg is: " << (*it).seg_slope << std::endl;
                cnt ++;
                if (((*it).seg_last_y - (*it).seg_intercept)== 2){
                    cnt_of_special_seg ++;
                }
            }
            delta_x_of_each_seg.emplace_back(delta_x);
        }
        std::cout<<"Filtered number is: " << cnt << std::endl;
        std::cout<<"Specially filtered segment number is: " << cnt_of_special_seg << std::endl;

        write_vector_to_f(delta_x_of_each_seg, out_path);

    }




    void evaluate_indexer_split() {
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
        results = evaluate_regression_ML(all_predicted_pos_without_metainfo_within_seg, all_true_pos_within, error_);
        std::cout<< "ML oriented matricx for keys within segments: " << results<< std::endl;
        results = evaluate_regression_ML(all_predicted_pos_without_metainfo_outof_seg, all_true_pos_outof_seg, error_);
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
        for(Iterator it = first_key_iter_; it != last_key_iter_; it++, i++){
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


    void print_segments_statistics(){
        std::vector<double_t > slopes;

        for (auto x : used_learned_segments_){
            slopes.emplace_back(x.seg_slope);
        }
        std::cout<< "The segment number is: " << used_learned_segments_.size() << "; ";
        calculate_mean_std(slopes, true);
        return;
    }




    /**
    * Returns the size in bytes of the payload w.r.t. the data size restored in the segments.
    * @return the size in bytes of the data payloads.
    */
    size_t index_size_bytes() const {
       auto btree_stats = btree_storing_segs_.btree_map_storing_segs_of_fitting.get_stats();
       size_t size_in_bytes = (sizeof(KeyType) + sizeof(void*)) * btree_stats.innernodes * btree_stats.innerslots
               + (sizeof(KeyType) + sizeof(FittingSegmentModified<KeyType, float>)) * btree_stats.leaves * btree_stats.leafslots;
       return size_in_bytes;
    }


    KeyType find_seg_key(KeyType key) const;

    inline Pos predict_position(KeyType key) const;


protected:
    size_t error_;   ///< the maximum allowed error in the last_key_iter level of the FittingTree index
    size_t data_size_;    ///< The number of elements in the data.
    Iterator first_key_iter_;    ///< The iterator of the smallest element in the data.
    Iterator last_key_iter_;    ///< The iterator of the largest element in the data.


    std::vector<segment_type> learn_segments(Iterator first_iter, Iterator last_iter, size_t error);
    void organize_segments();
    inline bool check_within_segment(KeyType key);
    KeyType correct_position(Pos predicted_pos, KeyType key);


    void complete_segments(Iterator first);
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
std::vector<FittingSegmentModified<KeyType, Floating>>
        FittingTreeIndexerModified<KeyType, Pos, Floating, Iterator>::learn_segments(Iterator first_iter, Iterator last_iter, size_t err) {
    std::vector<segment_type> segments;
    segments.reserve(8192);

    size_t input_data_size = std::distance(first_iter, last_iter);
    std::cout<< "data size: " << input_data_size << std::endl;
    if (input_data_size == 0)
        return segments;
    assert(std::is_sorted(first_iter, last_iter, CompareForDataPair<KeyType>()));
    this->error_ = err;

    double_t slope_high = std::numeric_limits<double>::max();
    double_t slope_low = 0.0;
    //size_t segment_start = *first_iter;
    KeyType segment_start = (*first_iter).first;
    //size_t intercept = std::distance(first_iter, first_iter);
    size_t intercept = 0;
    Iterator segment_start_iter = first_iter;
    Iterator segment_end_iter = first_iter;
    KeyType seg_last;
    for (Iterator it = first_iter; it != last_iter; it++){
        // size_t debug_i = std::distance(first_iter, it);
        //double_t delta_y = std::distance(segment_start_iter, it);
        double_t delta_y = (*it).second - (*segment_start_iter).second;
        double_t delta_x = (*it).first - (*segment_start_iter).first;
        //double_t delta_x = *it - segment_start;
        double_t slope = (delta_x == 0) ? 0.0 : (delta_y / delta_x);
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
            seg_last = (*segment_end_iter).first;
            if (segments_head_tail_.count(segment_start) == 0){
                segments_head_tail_[segment_start] = seg_last;
            }
            segments.emplace_back(segment_start, 0.5 * (slope_high + slope_low), intercept, seg_last, (*it).second);
            //segment_start = *it;
            segment_start = (*it).first;
            //intercept = std::distance(first_iter, it);
            intercept = (*it).second - (*first_iter).second;
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
    segments.emplace_back(segment_start, 0.5 * (slope_high + slope_low), intercept, seg_last, (*segment_end_iter).second);


    return segments;
}


template<typename KeyType, typename Pos, typename Floating, typename Iterator>
void FittingTreeIndexerModified<KeyType, Pos, Floating, Iterator>::complete_segments(Iterator first_key_iter) {
    KeyType cur_last, cur_next_first;

    std::vector<std::pair<KeyType, std::pair<Floating, Floating>>> pairs;
    size_t i = 0;
    auto first_seg = *(learned_segments_.begin());
    auto first_key = (*first_key_iter).first;
    if (first_key < (first_seg.seg_start)){
        long cur_last_y, cur_next_first_y;
        cur_next_first_y = first_seg.seg_intercept;
        cur_last_y = 0;
        cur_next_first = first_seg.seg_start;
        double_t slope = double_t(cur_next_first_y - cur_last_y)  / double_t(cur_next_first - first_key);
        completed_learned_segments_.emplace_back(first_key, slope, cur_last_y, cur_next_first, cur_last_y+1);
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
            completed_learned_segments_.emplace_back(cur_last, slope, cur_last_y, cur_next_first, cur_last_y+1);
        }
        pairs.emplace_back((*it).seg_start, std::make_pair((*it).seg_slope, (*it).seg_intercept));
    }
    completed_learned_segments_.emplace_back(*(learned_segments_.end()-1));

    // <typename _Key, typename _Data,
    //	  typename _Compare = std::less<_Key>,
    //	  typename _Traits = btree_default_map_traits<_Key, _Data>,
    //	  typename _Alloc = std::allocator<std::pair<_Key, _Data> >
    // The data of fitting is <float, size_t>, indicating the slope and intercept
    stx::btree_map<KeyType, std::pair<Floating, Floating>, std::less<KeyType>,
            stx::btree_map_traits<KeyType, std::pair<Floating, Floating>>>
            constructed_btree(pairs.begin(), pairs.end());
    btree_storing_segs_.btree_map_storing_segs_of_fitting = constructed_btree;

}

template<typename KeyType, typename Pos, typename Floating, typename Iterator>
void FittingTreeIndexerModified<KeyType, Pos, Floating, Iterator>::organize_segments() {
    std::vector<std::pair<KeyType, std::pair<Floating, size_t>>> pairs;
    size_t i = 0;
    for (auto it = used_learned_segments_.begin(); it != used_learned_segments_.end(); it++, i++) {
        pairs.emplace_back((*it).seg_start, std::make_pair((*it).seg_slope, (*it).seg_intercept));
    }
    stx::btree_map<KeyType, std::pair<Floating, Floating>, std::less<KeyType>,
            stx::btree_map_traits<KeyType, std::pair<Floating, Floating>>>
            constructed_btree(pairs.begin(), pairs.end());
    btree_storing_segs_.btree_map_storing_segs_of_fitting = constructed_btree;

}



template<typename KeyType, typename Pos, typename Floating, typename Iterator>
inline KeyType FittingTreeIndexerModified<KeyType, Pos, Floating, Iterator>::find_seg_key(KeyType key) const {
    auto seg_leaf_iter = btree_storing_segs_.btree_map_storing_segs_of_fitting.tree.lower_bound_smallerthan(key);
    return seg_leaf_iter.key();

}



template<typename KeyType, typename Pos, typename Floating, typename Iterator>
inline Pos FittingTreeIndexerModified<KeyType, Pos, Floating, Iterator>::predict_position(KeyType key) const {
    /**
      * Returns the approximate position of a key.
      * @param key the value to search for
      * @return a struct with the approximate position
      * @see approx_pos_t
      */
    //btree_const_iterator seg_leaf_iter = btree_friend_.btree_map_slot_float.tree.lower_bound(key);
    auto seg_leaf_iter = btree_storing_segs_.btree_map_storing_segs_of_fitting.tree.lower_bound_smallerthan(key);
    KeyType start, intercept;
    Floating slope;
    start = seg_leaf_iter.key();
    slope = seg_leaf_iter.data().first;
    intercept = seg_leaf_iter.data().second;
    segment_type predicted_pos(start, slope, intercept, 0, 0);

    size_t predi_pos = round(predicted_pos(key));
    size_t delta_lo = predi_pos >= error_ ? predi_pos - error_ : 0;
    size_t delta_hi = predi_pos + error_ <= data_size_ ? predi_pos + error_ : predi_pos + error_ - data_size_;
    if (seg_leaf_iter.key() > key){
        predi_pos = 0;
        delta_lo = 0;
        int i(0);
        while (((*(first_key_iter_ + i)).first) < seg_leaf_iter.key()){
            i++;
        }
        delta_hi = i;
    }

    //return {seg_leaf_iter.key(), seg_leaf_iter.data().first, seg_leaf_iter.data().second};
    return {predi_pos, delta_lo, delta_hi};

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
inline bool FittingTreeIndexerModified<KeyType, Pos, Floating, Iterator>::check_within_segment(KeyType key) {
    /**
      * Returns the approximate position of a key.
      * @param key the value to search for
      * @return a struct with the approximate position
      * @see approx_pos_t
      */
    //btree_const_iterator seg_leaf_iter = btree_friend_.btree_map_slot_float.tree.lower_bound(key);
    auto seg_leaf_iter = btree_storing_segs_.btree_map_storing_segs_of_fitting.tree.lower_bound_smallerthan(key);
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
        static size_t as_key(size_t key){
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
inline KeyType FittingTreeIndexerModified<KeyType, Pos, Floating, Iterator>::correct_position(Pos predicted_pos, KeyType key) {
    // auto lo = this->first_key_iter_ + predicted_pos.lo;
    // auto hi = this->first_key_iter_ + predicted_pos.hi;
    // auto res_iter = std::lower_bound(lo, hi, key, CompareForDataPair<KeyType>());
    // return std::distance(this->first_key_iter_, res_iter); // the true indexed position of query ``key''

    // exponential search, for sampling and dynamic case that violates the error bounds.
    auto res_iter = exponential_search(this->first_key_iter_, this->last_key_iter_,
                                       key, predicted_pos.pos, CompareForDataPair<KeyType>());
    // return std::distance(this->first_key_iter_, res_iter); // the true indexed position of query ``key''
    return (*res_iter).first; // the true indexed position of query ``key''

}



#endif //LEARNED_INDEX_FITTINGTREEINDEXER_MODIFIED_HPP
