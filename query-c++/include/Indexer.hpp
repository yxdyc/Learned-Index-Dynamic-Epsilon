//
// Created by daoyuan on 2019/12/25.
//

#ifndef LEARNED_INDEX_INDEXER_HPP
#define LEARNED_INDEX_INDEXER_HPP


#include <vector>
#include <list>
#include <chrono>
#include <cstddef>
#include <dlib/matrix.h>
#include <dlib/statistics.h>
#include "IndexMechanism.hpp"
#include "cstddef"
#include "cstdio"
#include "cstdlib"
#include "cstring"
#include "PgmIndexer.hpp"
#include "ctime"


class Indexer {

public:
    void learn_index(Iterator first_iter, Iterator last_iter, size_t error, std::string strategy, size_t recursive_err) {
        this->learned_segments = learn_segments(first_iter, last_iter, error);
        organize_segments(strategy, recursive_err);
    }

    KeyType query(KeyType key) {
        auto predicted_pos = predict_position(key);
        KeyType res = correct_position(predicted_pos, key);
        return res;
    }

    void evaluate_indexer(int payload_size) {
        std::vector<Pos> all_predicted_pos;
        auto t0 = std::chrono::high_resolution_clock::now();
        for(Iterator it = first_key_iter; it != last_key_iter; it++){
            Pos predicted_pos = predict_position(*it);
            all_predicted_pos.emplace_back(predicted_pos);
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        int i = 0;
        for(Iterator it = first_key_iter ; it != last_key_iter; it++, i++){
            correct_position(all_predicted_pos[i], *it);
        }
        auto t2 = std::chrono::high_resolution_clock::now();
        //lookup-time per query
        auto predict_time = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count() / data_size;
        auto correct_time = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() / data_size;
        //size
        size_t size_of_all_segments = this->size_segments_bytes();
        size_t size_of_payload = this->size_payloads_bytes(payload_size);

        // ML-oriented metric
        // regress_metric = ["RMSE", "MAE", "R2"]
        std::vector<long> all_true_pos, all_predicted_poss;
        for (long i=0; i < data_size; i++){
            all_true_pos.push_back(i);
        }
        dlib::matrix<double, 1, 4> results;
        results =  evaluate_regression_ML(all_predicted_pos, all_true_pos);

        std::cout<< "predict time: " << predict_time << std::endl;
        std::cout<< "correct time: " << correct_time << std::endl;
        std::cout<< "size of all segments: " << size_of_all_segments << std::endl;
        std::cout<< "size of payloads: " << size_of_payload<< std::endl;
        std::cout<< "ML oriented matricx: " << results<< std::endl;

    }

/**
* Returns the size in bytes of the payload, i.e.,  the data size restored in the last_key_iter-level segments.
* @return the size in bytes of the data payloads.
*/
size_t size_payloads_bytes(int payload_size) const {
    return this->segments_count() * payload_size;
}
};


#endif //LEARNED_INDEX_INDEXER_HPP
