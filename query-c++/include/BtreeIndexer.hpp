//
// Created by daoyuan on 2019/12/25.
//

#ifndef LEARNED_INDEX_BTREE_INDEXER_HPP
#define LEARNED_INDEX_BTREE_INDEXER_HPP


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

template <typename btree_traits>
class btreeFriend;


//template <typename btree_traits=stx::btree_default_map_traits<size_t, size_t>> friend class ::btreeFriend;
#define BTREE_FRIENDS   \
    template <typename btree_traits=stx::btree_map_traits<size_t, size_t>> friend class ::btreeFriend;

#include "stx/btree.h"
#include "stx/btree_map.h"
#include "Utilities.hpp"


/** Generates default traits for a B+ tree used as a map.
 * following "The case for learned index structures", the page_size indicates the number of keys */
template <typename _Key, typename _Data, int _page_size=256>
struct btree_map_traits
{
    /// If true, the tree will self verify it's invariants after each insert()
    /// or erase(). The header must have been compiled with BTREE_DEBUG defined.
    static const bool   selfverify = false;

    /// If true, the tree will print out debug information and a tree dump
    /// during insert() or erase() operation. The header must have been
    /// compiled with BTREE_DEBUG defined and key_type must be std::ostream
    /// printable.
    static const bool   debug = false;

    /// Number of slots in each leaf of the tree. Estimated so that each node
    /// has a size of about 256 bytes.
    //static const int    leafslots = BTREE_MAX( 8, 256 / (sizeof(_Key) + sizeof(_Data)) );
    static const int    leafslots = _page_size;

    /// Number of slots in each inner node of the tree. Estimated so that each node
    /// has a size of about 256 bytes.
    //static const int    innerslots = BTREE_MAX( 8, 256 / (sizeof(_Key) + sizeof(void*)) );
    static const int    innerslots = _page_size;

    /// As of stx-btree-0.9, the code does linear search in find_lower() and
    /// find_upper() instead of binary_search, unless the node size is larger
    /// than this threshold. See notes at
    /// http://panthema.net/2013/0504-STX-B+Tree-Binary-vs-Linear-Search
    static const size_t binsearch_threshold = 256;
    //static const size_t binsearch_threshold = 1;
    constexpr static const double leaf_filling_factor = 1;  //default is 2, "1" indicates the full-paged filling
    constexpr static const double inner_filling_factor = 1; //default is 2, "1" indicates the full-paged filling
};



//#define PAGE_SIZE 512;
constexpr int PAGE_SIZE = 128;

//
//using key_type = double;
//template <typename btree_traits=stx::btree_default_map_traits<size_t, size_t>>
template <typename btree_traits=stx::btree_map_traits<key_type_transformed, size_t>>
class btreeFriend{
public:
    //using btree_traits = btree_map_traits<size_t, size_t, PAGE_SIZE> ;
    stx::btree_map<key_type_transformed, size_t, std::less<key_type_transformed>, btree_traits> btree_map_slot;
    stx::btree<key_type_transformed, size_t, std::pair<key_type_transformed, size_t>, std::less<key_type_transformed>, btree_traits> btree_slot;
    //stx::btree_map<size_t, size_t> btree_map_slot;
    //stx::btree<size_t, size_t> btree_slot;
    stx::btree_map<key_type_transformed, std::pair<double, double>, std::less<key_type_transformed>,
            stx::btree_map_traits<key_type_transformed, std::pair<double, double>>> btree_map_storing_segs_of_fitting;
};

//
//using key_type = double;
//template <typename btree_traits=stx::btree_default_map_traits<size_t, size_t>>
template <typename btree_traits=stx::btree_map_traits<key_type_transformed, std::pair<key_type_transformed, size_t>>>
class btreeFriendForFintingTree{
public:
    //using btree_traits = btree_map_traits<size_t, size_t, PAGE_SIZE> ;
    stx::btree_map<key_type_transformed, size_t, std::less<key_type_transformed>, btree_traits> btree_map_slot;
    stx::btree<key_type_transformed, size_t, std::pair<key_type_transformed, size_t>, std::less<key_type_transformed>, btree_traits> btree_slot;
    //stx::btree_map<size_t, size_t> btree_map_slot;
    //stx::btree<size_t, size_t> btree_slot;
    stx::btree_map<key_type_transformed, std::pair<key_type_transformed, size_t>, std::less<key_type_transformed>,
            btree_traits> btree_map_slot_fitting;
};



template <typename KeyType, typename Iterator, typename btree_traits>
class BtreeIndexer{

public:
    //typedef stx::btree<key_type, data_type, value_type, key_compare,
            //traits, false, allocator_type, false> btree_impl;
    //typedef typename stx::btree_map<KeyType, KeyType>::btree_friend btreeFriend;

    btreeFriend<btree_traits> btree_friend_;
    Iterator first_key_iter_, last_key_iter_;
    size_t data_size_;
    size_t payload_size_;

    BtreeIndexer(Iterator first_key_iter, Iterator last_key_iter, size_t data_size) :
        first_key_iter_(first_key_iter), last_key_iter_(last_key_iter), data_size_(data_size) {};

    BtreeIndexer(Iterator first_key_iter, Iterator last_key_iter, size_t data_size, size_t payload_size) :
            first_key_iter_(first_key_iter), last_key_iter_(last_key_iter), data_size_(data_size), payload_size_(payload_size) {};

    //using btree_node_ptr = decltype(btree_friend_.btree_slot.m_root);
    //using btree_leaf_ptr = decltype(btree_friend_.btree_slot.m_headleaf);
    size_t tmp_size_t_value;
    using btree_leaf_ptr = decltype(btree_friend_.btree_map_slot.tree.find_approx_leaf(tmp_size_t_value));
    unsigned short tmp_value;
    //using btree_inner_node_ptr = decltype(btree_friend_.btree_slot.allocate_inner(tmp_value));
    //using btree_const_iterator = typename stx::btree<KeyType, KeyType>::const_iterator;


    std::chrono::system_clock::time_point learn_index(Iterator first_iter, Iterator last_iter) {
        assert(std::distance(first_iter, last_iter) == data_size_);
        // std::vector<std::pair<size_t, size_t>> pairs;
        // size_t i = 0;
        // for (Iterator it = first_iter; it != last_iter; it++, i++) {
        //     pairs.emplace_back(*it, i);
        // }
        // stx::btree_map<key_type_transformed, size_t, std::less<key_type_transformed>, btree_traits>
        //         constructed_btree(pairs.begin(), pairs.end());
        std::chrono::system_clock::time_point t0 = std::chrono::system_clock::now();
        //using btree_traits = btree_map_traits<KeyType, KeyType, PAGE_SIZE> ;
        stx::btree_map<key_type_transformed, size_t, std::less<key_type_transformed>, btree_traits>
                constructed_btree(first_iter, last_iter);
        btree_friend_.btree_map_slot = constructed_btree;
        return t0;
    }

    KeyType query(KeyType key) {
        return btree_friend_.btree_map_slot[key];
    }

    void evaluate_indexer(int payload_size, bool shuffle=true) {
        std::vector<std::pair<key_type_transformed, size_t>> tmp_data(first_key_iter_, last_key_iter_);
        if (shuffle == true){
            std::srand(1234);
            random_shuffle(tmp_data.begin(), tmp_data.end());
        }

        std::vector<btree_leaf_ptr> all_predicted_pos;
        auto t0 = std::chrono::high_resolution_clock::now();
        for(Iterator it = tmp_data.begin(); it != tmp_data.end(); it++){
            btree_leaf_ptr leaf = predict_position((*it).first);
            all_predicted_pos.emplace_back(leaf);
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        int i = 0, wrong_return_count = 0;
        for(Iterator it = tmp_data.begin(); it != tmp_data.end(); it++, i++){
            size_t corrected_res = correct_position(all_predicted_pos[i], (*it).first);
            if (corrected_res != (*it).second) {
                wrong_return_count ++;
            }
        }
        auto t2 = std::chrono::high_resolution_clock::now();
        //lookup-time per query
        auto predict_time = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count() / data_size_;
        auto correct_time = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() / data_size_;
        //size including the payload
        size_t size_of_payload = this->size_payloads_bytes(payload_size);
        //size without the payloads
        size_t size_of_index = this->size_index_bytes();

        // ML-oriented metric
        // regress_metric = ["RMSE", "MAE", "R2"]
        std::vector<size_t> all_true_pos, all_predicted_pos_without_metainfo;
        for (size_t i=0; i < data_size_; i++){
            all_true_pos.push_back(tmp_data[i].second);
            size_t predicted_pos_without_metainfo = all_predicted_pos[i]->slotdata[0];
            all_predicted_pos_without_metainfo.push_back(predicted_pos_without_metainfo);
        }
        dlib::matrix<double, 1, 4> results;
        results =  evaluate_regression_ML(all_predicted_pos_without_metainfo, all_true_pos);

        std::cout<< "predict time: " << predict_time << std::endl;
        std::cout<< "correct time: " << correct_time << std::endl;
        std::cout<< "overall time: " << predict_time + correct_time << std::endl;
        std::cout<< "size of index including payloads: " << size_of_payload<< std::endl;
        std::cout<< "size of index without payloads): " << size_of_index << std::endl;
        std::cout<< "wrong return count: " << wrong_return_count << std::endl;
        std::cout<< "ML oriented matricx: " << results<< std::endl;
    }

    /**
     *
     * Tries to locate a key in the B+ tree and returns an constant iterator
     * to the leaf node if found.
     */
    btree_leaf_ptr predict_position(KeyType key) {
        /*
         * the find_approx_leaf function is implemented in the btree.h as a public method as follow:
         *
        const btree_node_ptr *n = btree_friend_.btree_map_slot.tree.m_root;
        if (!n) return btree_friend_.btree_map_slot.tree.end();

        while(!n->isleafnode())
            {
                const btree_inner_node_ptr *inner = static_cast<const btree_inner_node_ptr*>(n);
                int slot = btree_friend_.btree_map_slot.tree.find_lower(inner, key);

                n = inner->childid[slot];
            }

        const btree_leaf_ptr *leaf = static_cast<const btree_leaf_ptr*>(n);

        return leaf;
         */

        btree_leaf_ptr leaf = btree_friend_.btree_map_slot.tree.find_approx_leaf(key);

        return leaf;


    }
     //btree_const_iterator correct_position(btree_leaf_ptr leaf, KeyType key){
     auto correct_position(btree_leaf_ptr leaf, KeyType key){
        /*
        unsigned short slot = btree_friend_.btree_map_slot.tree.find_lower(leaf, key);
        return (slot < leaf->slotuse && btree_friend_.btree_map_slot.tree.key_equal(key, leaf->slotkey[slot]))
               ? btree_friend_.btree_map_slot.tree.const_iterator(leaf, slot) : btree_friend_.btree_map_slot.tree.end();
        */
        auto res = btree_friend_.btree_map_slot.tree.find_exact_leaf(key, leaf);
        return res.data(); // data is the position
    };

/**
* @return the size in bytes of the data payloads.
*/
size_t size_payloads_bytes(int payload_size) const {
    auto btree_stats = btree_friend_.btree_map_slot.get_stats();
    size_t size_in_bytes = (sizeof(KeyType) + sizeof(void*)) * btree_stats.innernodes * btree_stats.innerslots
             + (sizeof(KeyType) + payload_size) * btree_stats.leaves * btree_stats.leafslots;

    return size_in_bytes;
}

/**
* @return the size in bytes of the index (without the payload).
*/
    size_t size_index_bytes() const {
        auto btree_stats = btree_friend_.btree_map_slot.get_stats();
        size_t size_in_bytes = (sizeof(KeyType) + sizeof(void*)) * (btree_stats.innernodes * btree_stats.innerslots +
                btree_stats.leaves * btree_stats.leafslots);

        return size_in_bytes;
    }

};





#endif //LEARNED_INDEX_BTREE_INDEXER_HPP
