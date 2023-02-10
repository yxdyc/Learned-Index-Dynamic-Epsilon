#include <iostream>
#include <vector>
#include <stdio.h>
#include <unistd.h>
#include <gflags/gflags.h>
#include <ALEX-RMI.hpp>

#include "DataProcessor.hpp"
#include "PgmIndexer.hpp"
#include "PgmIndexer_Modified.hpp"
#include "FittingTreeIndexer.hpp"
#include "FittingTreeIndexer_Modified.hpp"
#include "RegressionIndexer.hpp"
#include "BtreeIndexer.hpp"
#include "RMIIndexerLinear.hpp"
#include <cxxopts.hpp>


using size_t = unsigned long;
using namespace std;


/*
 * Test flags for ablation study and analytic experiments
 */
bool INSERT_GAP = false;
bool LOGICAL_MAE = false;

bool STATIC_GAPS_MAE = false; // calculate static mae after gap insertion

// In static case, re-train linear model globally if true, else re-train locally.
// ``globally'' indicates learning from scratch, getting varied new segements;
// ``locally'' indicates learning each segment respectively, getting the same number of segments before re-train;
bool GLOBAL_RE_TRAIN = true;


bool ESTIMATE_AND_ALLOCATE = false;
bool DELTA_Y_BY_SEGMENTS = false;
bool COMPARE_SEGMENTS_LEARNED_BY_DIFFERENT_Y = false;
bool SPANS_OF_STATIC_SEGMENTS = false; // statistic for the spans of each segments in static case


// using key_type = size_t;
// using key_type = double;   // For map data
// using key_type_transformed = double_t;

using pgm_indexer_type = PgmIndexerModified<key_type_transformed, PGMPos, float_t,
        std::vector<std::pair<key_type_transformed, size_t>>::iterator>;

template <typename KEY_TYPE>
void calculate_logical_mae(size_t error, float sample_rate,
                           vector<std::pair<KEY_TYPE, size_t>> &data_with_sampled_original_pos,
                           FittingTreeIndexerModified<key_type, PGMPos, double, typename std::vector<std::pair
                                   <KEY_TYPE, size_t>>::iterator> &fitting_tree_indexer_completed_from_sampled_y,
                           vector<std::pair<KEY_TYPE, size_t>> &data_with_sampled_incremental_pos,
                           vector<std::pair<KEY_TYPE, size_t>> &data_with_gap_inserted_pos);

void place_all_data_by_gap_y(std::vector<std::pair<key_type_transformed, size_t>> &whole_data_with_gapped_pos,
                             size_t &number_of_non_gaps,
                             std::vector<std::pair<key_type_transformed,
                             std::vector<key_type_transformed>>> &gapped_array_with_linking_array);

void insert_gap_and_evaluate(const vector<std::pair<key_type_transformed, size_t>> &data_type_transformed_with_pos,
                             int payload_size_in_bytes, size_t error,
                             const std::vector<std::pair<key_type_transformed, size_t>> &sampled_data_with_pos,
                             bool use_complete_segments, const std::string &organize_strategy, size_t recurrsive_error,
                             PgmIndexerModified<key_type_transformed, PGMPos, float_t, std::vector<std::pair<key_type_transformed, size_t>>::iterator> &pgm_indexer,
                             double gap_rate, std::vector<std::pair<key_type_transformed, size_t>> (*insert_fp)(
        vector<std::pair<key_type_transformed, size_t>>, double, vector<SegmentModified<key_type_transformed, float_t>>,
        bool, std::string), void (*update_fp)(
        vector<std::pair<key_type_transformed, size_t>> &, vector<std::pair<key_type_transformed, size_t>> const &,
        pgm_indexer_type &), const std::string &strategy,
                             vector<std::pair<key_type_transformed, size_t>> &data_with_predicted_pos,
                             vector<std::pair<key_type_transformed, size_t>> &data_with_gap_inserted_pos
                             );

vector<pair<key_type_transformed, size_t>> &
sample_data_for_index(vector<pair<key_type_transformed, size_t>> &data_type_transformed_with_pos,
                      vector<pair<key_type_transformed, size_t>> &sampled_data_with_pos, double sample_rate);

void generate_batch_from_timestamps(vector<vector<std::pair<key_type_transformed, size_t>>> & generated_batch,
                    const std::vector<std::pair<key_type_transformed, size_t>> original_data,
                    std::string generate_strategy);

struct BtreeIndexerFunctor
{
    // to access B-trees with different page size
    template<typename T>
    void operator()(T& t) const {
        // B-tree indexer build
        std::cout << "Begin to construct B-tree indexer" << std::endl;
        auto t0 = std::chrono::high_resolution_clock::now();
        t.learn_index(t.first_key_iter_, t.last_key_iter_);
        auto t1 = std::chrono::high_resolution_clock::now();
        auto construct_time = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
        std::cout << "Finished B-tree indexer construction, elapse time is " << construct_time << std::endl;
        // B-tree indexer evaluation
        std::cout << "Evaluating setting for B-tree indexer" << std::endl
                  << " The key_size, page_size and payload_size are: ";
        std::cout << sizeof(key_type) << ", " << t.btree_friend_.btree_map_slot.leafslotmax
                  << ", " << t.payload_size_ << std::endl;
        t.evaluate_indexer(t.payload_size_);
    }
};

int main(int argc, char* argv[]) {
    char buf[200];
    getcwd(buf, sizeof(buf));
    printf("current working directory : %s\n", buf);

    //test_simple_linear_regression_with_mult_prev();
    cxxopts::Options options("Static_gap", "Test the position adjustment technique.");
    options.add_options()
            ("pgm", "test pgm indexer", cxxopts::value<bool>()->default_value("false"))
            ("b-tree", "test b-tree indexer", cxxopts::value<bool>()->default_value("false"))
            ("rmi", "test rmi indexer", cxxopts::value<bool>()->default_value("false"))
            ("regression", "test pure-ml regression indexer",
                    cxxopts::value<bool>()->default_value("false"))
            ("fitting-tree", "test fitting-tree indexer",
                    cxxopts::value<bool>()->default_value("false"))

            ("iot", "test iot dataset", cxxopts::value<bool>()->default_value("false"))
            ("web", "test weblog dataset", cxxopts::value<bool>()->default_value("false"))
            ("longtitude", "test longtitude map dataset", cxxopts::value<bool>()->default_value("false"))
            ("latilong", "test latilong map dataset", cxxopts::value<bool>()->default_value("false"))

            ("s,save_data_binary", "read data from tsv, then save to binary form",
                    cxxopts::value<bool>()->default_value("false"))
            ("r,read_data_binary", "read data from binary file",
             cxxopts::value<bool>()->default_value("true"))
            ("w_rate", "the insertion proportion of original dataset",
                    cxxopts::value<float>()->default_value("0.3"))
            ;
    auto opt_result = options.parse(argc, argv);


    std::unordered_map<std::string, bool> test_flags;

    // models
    test_flags["PGM"] = opt_result["pgm"].as<bool>();
    test_flags["B-tree"] = opt_result["b-tree"].as<bool>();
    test_flags["RMI"]= opt_result["rmi"].as<bool>();
    test_flags["Regression"] = opt_result["regression"].as<bool>();
    test_flags["Fitting-tree"] = opt_result["fitting-tree"].as<bool>();

    // dataset
    test_flags["iot"] = opt_result["iot"].as<bool>();
    test_flags["web"]= opt_result["web"].as<bool>();
    test_flags["longtitude"] = opt_result["longtitude"].as<bool>();
    test_flags["latilong"] = opt_result["latilong"].as<bool>();



    // compare fitting-tree and rmi
    test_flags["fitting_vs_rmi"] = 0;

    srand(1234);
    test_flags["write_data_to_binary"] = opt_result["s"].as<bool>();
    test_flags["read_from_binary"] = opt_result["r"].as<bool>();
    for(auto flag : test_flags) {
        std::cout << flag.first << ": " << flag.second << std::endl;
    }

    // data prepare
    std::vector<key_type> data_original;
    std::vector<key_type> data;
    std::vector<key_type_transformed> data_type_transformed;
    std::string data_path;
    if (test_flags["iot"]){
        IotProcessor iotProcessor;
        if (test_flags["read_from_binary"]){
            data_path = "/home/xxx/work/learned_index/data/iot.uint64";
            data = iotProcessor.read_data_binary(data_path, 30000000);
        }
        else {
            data_path = "/home/xxx/work/learned_index/data/iot.csv";
            data_original = iotProcessor.read_data_csv(data_path, 30000000);
        }
    }
    else if (test_flags["web"]){
        WebBlogsProcessor webProcessor;
        if (test_flags["read_from_binary"]){
            data_path = "/home/xxx/work/learned_index/data/weblogs.uint64";
            data = webProcessor.read_data_binary(data_path, 720000000);
        }
        else{
            data_path = "/home/xxx/work/learned_index/data/weblogs.csv";
            data_original = webProcessor.read_data_csv(data_path, 720000000);
        }
    } else if (test_flags["longtitude"]){
        MapProcessor mapProcessor;
        //data_path = "/home/xxx/work/learned_index/data/longtitude.f64";
        data_path = "/home/xxx/work/learned_index/data/longtitude_newyork.double";
        data_path = "/home/xxx/work/learned_index/data/longtitude_china.double";
        data = mapProcessor.read_data_binary(data_path, 720000000);
    } else if (test_flags["latilong"]){
        MapProcessor mapProcessor;
        //data_path = "/home/xxx/work/learned_index/data/latilong.f64";
        data_path = "/home/xxx/work/learned_index/data/latilong_newyork.double";
        data_path = "/home/xxx/work/learned_index/data/latilong_china.double";
        data = mapProcessor.read_data_binary(data_path, 720000000);
    }else {
        std::cout<< "You should choose one dataset to test."<<std::endl;
    }

    // from "data_original" to "data": sort and de-duplicate
    std::sort(data_original.begin(), data_original.end());
    std::unique_copy(data_original.begin(), data_original.end(), back_inserter(data));
    size_t data_original_size = data_original.size(), data_size = data.size();
    std::vector<std::pair<key_type_transformed, size_t>> data_type_transformed_with_pos,
                init_data_type_transformed_with_pos, query_data_type_transformed_with_pos;
    std::cout<< "The size of original data and distinct data are " << data_original_size << ", "<< data_size <<std::endl;
    std::vector<size_t> pos;

    // cache pre-proccessed data into binary file
    if (test_flags["write_data_to_binary"]){
        int found_pos=data_path.find("csv",0);
        std::string out_data_path = data_path.replace(found_pos, 3, "uint64");
        write_vector_to_f<key_type>(data, out_data_path);
    }

    std::srand(1234);
    // std::random_shuffle(data.begin(), data.end());

    double write_proportion = opt_result["w_rate"].as<float>();
    size_t write_data_size = floor(double(data.size()) * write_proportion);
    size_t init_data_size = data.size() - write_data_size;
    assert(data.size() > write_data_size);
    for (size_t i = 0; i < data_size; i++){
        auto key = key_type_transformed(data[i]);
        data_type_transformed.emplace_back(key_type_transformed(key));
        pos.emplace_back(i);

        std::pair<key_type_transformed, size_t> data_type_transformed_pair = std::make_pair(key, i);
        if (i < init_data_size){
            init_data_type_transformed_with_pos.emplace_back(data_type_transformed_pair);
        } else{
            query_data_type_transformed_with_pos.emplace_back(data_type_transformed_pair);
        }
        data_type_transformed_with_pos.emplace_back(data_type_transformed_pair);
    }

    vector<vector<std::pair<key_type_transformed, size_t> >> query_data_batches;
    vector<std::string> batch_generate_strategies =
            {"month_level",
             "day_level",
             "hour_level",
            };
    std::string batch_generate_strategy = batch_generate_strategies[1];
    std::cout << "Begin to generate batched data, the write rate is " << write_proportion << std::endl;
    generate_batch_from_timestamps(query_data_batches, query_data_type_transformed_with_pos, batch_generate_strategy);
    std::cout << "Generated " << query_data_batches.size() << " batches. " << std::endl;


    // // insertion and evaluation on K-batched data
    // int K = 10;
    // std::cout << "Write proportion and the number of batch are: " << write_proportion << ", " << K << std::endl;
    // size_t batch_size = ceil(double (query_data_type_transformed.size()) / double(K));
    // vector<vector<key_type_transformed>> query_data_batches(K);
    // for(size_t i = 0; i < query_data_type_transformed.size(); i += batch_size) {
    //     auto last = std::min(query_data_type_transformed.size(), i + batch_size);
    //     auto index = i / batch_size;
    //     auto & vec = query_data_batches[index];
    //     vec.reserve(last - i);
    //     move(query_data_type_transformed.begin() + i, query_data_type_transformed.begin() + last, back_inserter(vec));
    // }

    std::chrono::system_clock::time_point t0, t1;

    int payload_size_in_bytes = 64 / 8; // dummy payload, only used for index size estimation
    std::vector<size_t> error_list = {16, 32, 64, 128, 256, 512, 1024};
    // std::vector<size_t> error_list = {256};
    std::vector<double > sample_rate_list = {1, 0.8, 0.6, 0.4, 0.2, 0.1, 0.05, 0.025, 0.01, 0.005, 0.0025, 0.001};


    std::string data_name;
    if (test_flags["longtitude"]){
        data_name = "longtitude";
        data_name = "longtitude_china";
    }else if (test_flags["latilong"]) {
        data_name = "latilong";
        data_name = "latilong_china";
    }else{
        data_name = (test_flags["iot"]) == 1 ? "iot" : "weblog";
    }

    size_t error = 256; // the alpha of PGM and Fitting-Tree: predefined error-bound
    if (test_flags["longlati"] or test_flags["latilong"] or test_flags["longtitude"]){
        error = 64;
    }
    std::vector<std::pair<key_type_transformed, size_t>> data_with_predicted_pos;

    if (test_flags["Fitting-tree"]){

        FittingTreeIndexer<key_type_transformed, PGMPos, double, std::vector<std::pair<key_type_transformed, size_t>>::iterator>
                fitting_tree_index_D1t(data_type_transformed_with_pos.begin(), data_type_transformed_with_pos.end(), data_type_transformed_with_pos.size());
        FittingTreeIndexer<key_type_transformed, PGMPos, double, std::vector<std::pair<key_type_transformed, size_t>>::iterator>
                fitting_tree_index_Dt(data_type_transformed_with_pos.begin(), data_type_transformed_with_pos.end(), data_type_transformed_with_pos.size());

        std::cout << "Begin to construct Fitting-tree indexer" <<std::endl;
        fitting_tree_index_D1t.learn_index(init_data_type_transformed_with_pos.begin(),
                                           init_data_type_transformed_with_pos.end(), error, 1.0);
        fitting_tree_index_Dt.learn_index(init_data_type_transformed_with_pos.begin(),
                                           init_data_type_transformed_with_pos.end(), error, 1.0);
        // evaluate after init
        data_with_predicted_pos.clear();
        for(auto x : init_data_type_transformed_with_pos){
            PGMPos predicted_pos = fitting_tree_index_D1t.predict_position(x.first);
            std::pair<key_type_transformed , size_t> predicted_res = std::make_pair(x.first, predicted_pos.pos);
            data_with_predicted_pos.emplace_back(predicted_res);
        }
        bool out_y1y2 = true;
        pair_MAE<key_type_transformed>(data_with_predicted_pos, init_data_type_transformed_with_pos, out_y1y2);

        size_t cur_data_idx = init_data_size;
        std::chrono::system_clock::time_point t0, t1;
        decltype(std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count()) during_time_D_t(0), during_time_D_1t(0);
        int batch_cnt = 0;
        for (auto batch : query_data_batches){
            std::cout << std::endl << "Begin to insert batched data, learn from scratch on D_{1,t}. " << std::endl;
            cur_data_idx += batch.size();
            t0 = std::chrono::high_resolution_clock::now();
            t1 = fitting_tree_index_D1t.learn_index(data_type_transformed_with_pos.begin(),
                    data_type_transformed_with_pos.begin() + cur_data_idx, error, 1.0);
            auto tmp_time = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
            during_time_D_1t += tmp_time;
            fitting_tree_index_D1t.evaluate_indexer(payload_size_in_bytes);
            batch_cnt ++;
            std::cout << "Average learning time on D_{1,t} is: " << during_time_D_1t / batch_cnt <<  std::endl;

            std::cout << std::endl << "Begin to insert batched data, learn from D_t. " << std::endl;
            t0 = std::chrono::high_resolution_clock::now();
            fitting_tree_index_Dt.append_batch(batch.begin(), batch.end());
            t1 = std::chrono::high_resolution_clock::now();
            tmp_time = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
            during_time_D_t += tmp_time;
            fitting_tree_index_Dt.evaluate_indexer(payload_size_in_bytes);
            batch_cnt ++;
            std::cout << "Average learning time on D_t is: " << during_time_D_t / batch_cnt <<  std::endl;
        }

    }


    return 0;
}

/***
 *  simulate online-learning batches from sorted data whose key type is timestamp
 * @param generated_batch
 * @param original_data
 * @param generate_strategy, split data according to different time granularities
 *          {"month_level",
             "day_level",
             "hour_level",
            };
 */
void generate_batch_from_timestamps(vector<vector<std::pair<key_type_transformed, size_t>>> & generated_batch,
                    const std::vector<std::pair<key_type_transformed, size_t>> original_data,
                    std::string generate_strategy) {
    assert(std::is_sorted(original_data.begin(), original_data.end()));
    double sample_rate_estimated, temp_rate, min_x_delta_mean = 1000000000, min_x_delta_std = 1000000000;
    // format of timestamp key: 20170325120000
    int year = int(original_data[0].first / 10000000000);
    long base_year = year * 10000000000;

    bool new_batch(false);
    int cur_month, cur_day, cur_hour;
    int last_month(-1), last_day(-1), last_hour(-1);
    for (int j = 0; j < original_data.size(); ++j) {
        auto data = original_data[j];
        key_type_transformed month_day_hour = data.first - base_year;
        if (generate_strategy == "month_level"){
            cur_month = floor(month_day_hour / 100000000) - 1 ;
            if (cur_month != last_month){
                new_batch = true;
                last_month = cur_month;
            }
        } else if (generate_strategy == "day_level"){
            cur_day = int(floor(month_day_hour / 1000000)) % 100 ;
            if (cur_day != last_day){
                new_batch = true;
                last_day = cur_day;
            }
        } else if (generate_strategy == "hour_level"){
            cur_hour = int(floor(month_day_hour / 10000)) % 100 ;
            if (cur_hour != last_hour){
                new_batch = true;
                last_hour = cur_hour;
            }
        } else {
            throw "Un-supported generate strategy, the strategy string must be [month_day, day_level, hour_level].";
        }
        if (new_batch){
            vector<std::pair<key_type_transformed, size_t>> tmp_batch;
            generated_batch.emplace_back(tmp_batch);
            generated_batch.back().emplace_back(data);
            new_batch = false;
        } else {
            generated_batch.back().emplace_back(data);
        }

    }
    return;

}

vector<pair<key_type_transformed, size_t>> &
sample_data_for_index(vector<pair<key_type_transformed, size_t>> &data_type_transformed_with_pos,
                      vector<pair<key_type_transformed, size_t>> &sampled_data_with_pos, double sample_rate) {
    size_t sample_size = round(data_type_transformed_with_pos.size() * sample_rate);
    size_t data_size = data_type_transformed_with_pos.size();
    // to simplify the complement of learned index, we fixed the smallest and largest elements at head and tail.
    if (data_size < 4){
        // if there is only one element between head and tail, we do not need shuffle them.
        return data_type_transformed_with_pos;
    }
    auto first_data = *(data_type_transformed_with_pos.begin());
    auto last_data = *(data_type_transformed_with_pos.end()-1);
    bool hold_head_and_tail = true;
    sample_size = sample_size > 3 ? sample_size : 3; // at lease 3 items to handle the corner case of random_sample
    sampled_data_with_pos = random_sample_without_replacement(
            data_type_transformed_with_pos, sample_size, hold_head_and_tail);
    sampled_data_with_pos.emplace_back(first_data);
    sampled_data_with_pos.emplace_back(last_data);
    sort(sampled_data_with_pos.begin(), sampled_data_with_pos.end());
#ifdef Debug
    for(int i = 0; i < sampled_data_with_pos.size() - 1; i++){
        auto cur_item = sampled_data_with_pos[i];
        auto next_item = sampled_data_with_pos[i+1];
        assert(cur_item.first < next_item.first);
        assert(cur_item.second < next_item.second);
    }
#endif
    return sampled_data_with_pos;
}

/*
 * for
 */
void insert_gap_and_evaluate(const std::vector<std::pair<key_type_transformed, size_t>> &data_type_transformed_with_pos,
                             int payload_size_in_bytes, size_t error,
                             const std::vector<std::pair<key_type_transformed, size_t>> &sampled_data_with_pos,
                             bool use_complete_segments, const std::string &organize_strategy, size_t recurrsive_error,
                             PgmIndexerModified<key_type_transformed, PGMPos, float_t,
                                    std::vector<std::pair<key_type_transformed, size_t>>::iterator> &pgm_indexer,
                             double gap_rate,
                             vector<std::pair<key_type_transformed, size_t>> (*insert_fp)(
                                     vector<std::pair<key_type_transformed, size_t>>, double,
                                     vector<SegmentModified<key_type_transformed, float_t>>,
                                     bool, std::string),
                             void (*update_fp)(vector<std::pair<key_type_transformed, size_t>> &,
                                               vector<std::pair<key_type_transformed, size_t>> const &,
                                               pgm_indexer_type &),
                             const std::string &strategy,
                             vector<std::pair<key_type_transformed, size_t>> &data_with_predicted_pos,
                             vector<std::pair<key_type_transformed, size_t>> &data_with_gap_inserted_pos
) {
    vector<std::pair<key_type_transformed, size_t>> whole_data_with_gapped_pos = data_type_transformed_with_pos;
    auto learned_segments = pgm_indexer.learned_segments_;
    // insert gaps using the learned segments.    D'  ->   D''
    data_with_gap_inserted_pos = insert_fp
            (sampled_data_with_pos, gap_rate, learned_segments, true, "");
#ifdef Debug
    for (int i = 0; i < data_type_transformed_with_pos.size()-1; i ++){
        assert(data_type_transformed_with_pos[i+1].second >= data_type_transformed_with_pos[i].second);
    }
#endif
    std::cout << std::endl << "Strategy is " << strategy << std::endl;
    pgm_indexer.data_size_ = data_with_gap_inserted_pos[data_with_gap_inserted_pos.size()-1].second + 1;
    std::cout << "Gap inserted size: "<< pgm_indexer.data_size_ <<std::endl;
    assert(std::is_sorted(data_with_gap_inserted_pos.begin(), data_with_gap_inserted_pos.end()));

    // learn segments on D''
    if (GLOBAL_RE_TRAIN){
        pgm_indexer.learn_index(data_with_gap_inserted_pos.begin(), data_with_gap_inserted_pos.end(), error,
                                organize_strategy, recurrsive_error, 1.0, use_complete_segments);
    } else{
        pgm_indexer.re_learn_index_segment_wise(data_with_gap_inserted_pos.begin(),
                                                data_with_gap_inserted_pos.end(), organize_strategy, recurrsive_error);
    }

    data_with_predicted_pos.clear();
    // update D using D'', resulting the completed data D'''.
    update_fp(whole_data_with_gapped_pos, data_with_gap_inserted_pos, pgm_indexer);
    // evaluate the index learned from D'',  on the D'''.
    for (auto x : whole_data_with_gapped_pos){
        PGMPos predicted_pos = pgm_indexer.predict_position(x.first);
        std::pair<key_type_transformed, size_t> predicted_res = std::make_pair(x.first, predicted_pos.pos);
        data_with_predicted_pos.emplace_back(predicted_res);
    }
    std::cout << "Statistics for the PGM learned on D'', evaluated on D''': " << std::endl;
    pgm_indexer.print_segments_statistics();
    pair_MAE<key_type_transformed>(data_with_predicted_pos, whole_data_with_gapped_pos);

    // physically put the key from D''' into the new arrays
    size_t number_of_non_gaps(0);
    std::vector<std::pair<key_type_transformed, std::vector<key_type_transformed>>> gapped_array_with_linking_array;
    place_all_data_by_gap_y(whole_data_with_gapped_pos, number_of_non_gaps,
                            gapped_array_with_linking_array);

    // physically query on the gapped_array_with_linking_array
    pgm_indexer.evaluate_index_with_gapped_array(whole_data_with_gapped_pos, gapped_array_with_linking_array,
            payload_size_in_bytes, number_of_non_gaps);

}

/*
 *
 physically put the key from D''' into the new arrays
 the format of the new array, including 3 types of data:
 1. un-conflicting key:  <k, v>, v has only one element, a virtual placeholder;
 2. gap:  <k, null>;
 3. conflicting key:  <k, v>,  v has more than 1 elements, which are in [k, key(k+1)).
 then we can distinguish the three kind by their size of the value vector: 1, 0, >1 respectively
 *
 */
void place_all_data_by_gap_y(std::vector<std::pair<key_type_transformed, size_t>> &whole_data_with_gapped_pos,
                             size_t &number_of_non_gaps,
                             std::vector<std::pair<key_type_transformed,
                             std::vector<key_type_transformed>>> &gapped_array_with_linking_array){

    auto gapped_array_size = whole_data_with_gapped_pos.back().second + 1;
    gapped_array_with_linking_array.reserve(gapped_array_size);
    std::unordered_map<size_t, double_t > counter_of_round_ys;
    size_t largest_len_of_linking_array= 0;
    for (auto data_pos : whole_data_with_gapped_pos) {
        counter_of_round_ys[data_pos.second]++;
        largest_len_of_linking_array = counter_of_round_ys[data_pos.second] > largest_len_of_linking_array ?
                                       counter_of_round_ys[data_pos.second] : largest_len_of_linking_array;
    }
    number_of_non_gaps = counter_of_round_ys.size();
    size_t new_array_current_i(0), orginal_array_current_i(0);
    std::vector<key_type_transformed> place_holder{0}; // have only one fake data: size is 1
    std::vector<key_type_transformed> gap_indicator; // have zero element: size is 0
    std::vector<key_type_transformed> tmp_linking_array;
    tmp_linking_array.reserve(largest_len_of_linking_array);
    while (orginal_array_current_i < whole_data_with_gapped_pos.size()){
    //for (auto data : whole_data_with_gapped_pos){
        auto & data = whole_data_with_gapped_pos[orginal_array_current_i];
        auto current_pos = data.second;
        // case 1 and the first key of case 3
        if (new_array_current_i == current_pos){
            auto linking_array_len = counter_of_round_ys[current_pos];
            // case 1, un-conflict key:  <k, v>, v has only one element, k;
            if (linking_array_len == 1){
                place_holder[0] = data.first;
                auto key_pair = std::make_pair(data.first, place_holder);
                gapped_array_with_linking_array.emplace_back(key_pair);
                new_array_current_i ++;
                orginal_array_current_i ++;
            }
            // case 3, conflicting keys, data is the first key of case 3
            else if (linking_array_len > 1) {
                // put all the keys having the same y^ into a vector
                tmp_linking_array.clear();
                for (int j = 0; j < int(linking_array_len); j++) {
                    auto key = whole_data_with_gapped_pos[orginal_array_current_i + j].first;
                    tmp_linking_array.emplace_back(key);
                }
                auto key_pair = std::make_pair(data.first, tmp_linking_array);
                gapped_array_with_linking_array.emplace_back(key_pair);
                new_array_current_i++;
                orginal_array_current_i += linking_array_len;
            }
        } else if (new_array_current_i < current_pos){
            // case 2: add (current_pos - new_array_current_i) gaps,  <k, null>, to hold a total order
            auto gap_data = std::make_pair(data.first, gap_indicator);
            size_t gap_num = current_pos - new_array_current_i;
            gapped_array_with_linking_array.insert(gapped_array_with_linking_array.end(), gap_num, gap_data);
            new_array_current_i += gap_num;
            // while (new_array_current_i < current_pos){
            //     gapped_array_with_linking_array.emplace_back(std::make_pair(data.first, gap_indicator));
            //     new_array_current_i ++;
            // }
        } else {
            throw "Assumption violation: current_pos should >= new_array_current_i"s;
        }
    }
    assert(gapped_array_size == gapped_array_with_linking_array.size());
    std::cout << "In data complement stage, statistics for conflicts" << std::endl;
    vector<double_t > conflicts;
    conflicts.reserve(counter_of_round_ys.size());
    for (auto kv : counter_of_round_ys){
        if (kv.second > 1){
            conflicts.emplace_back(kv.second);
        }
    }
    calculate_mean_std(conflicts);
}


template <typename KEY_TYPE>
void calculate_logical_mae(size_t error, float sample_rate,
                           vector<std::pair<KEY_TYPE, size_t>> &data_with_sampled_original_pos,
                           FittingTreeIndexerModified<KEY_TYPE, PGMPos, double, typename std::vector<std::pair<KEY_TYPE, size_t>>::iterator> &fitting_tree_indexer_completed_from_sampled_y,
                           vector<std::pair<KEY_TYPE, size_t>> &data_with_sampled_incremental_pos,
                           vector<std::pair<KEY_TYPE, size_t>> &data_with_gap_inserted_pos) {
    std::cout << "Begin to construct Fitting-tree indexer" << std::endl;
    fitting_tree_indexer_completed_from_sampled_y.learn_index(data_with_sampled_incremental_pos.begin(),
            data_with_sampled_incremental_pos.end(), error, 1.0, false);
    fitting_tree_indexer_completed_from_sampled_y.complete_seg_last_y();
    bool estimate_by_segments = true;
    if (estimate_by_segments){
        // analysis
        // double sample_rate_estimated = estimate_sample_rate_time_slots(data_with_sampled_incremental_pos);
        std::vector<int> top_ns = {10, 30, 50, 100, 200, 500};
        // std::vector<int> top_ns = {10};
        //top_ns.emplace_back(fitting_tree_indexer_completed_from_sampled_y.used_learned_segments_.size());
        for (auto top_n : top_ns){
            double sample_rate_estimated = estimate_sample_rate_auto(data_with_sampled_incremental_pos,
                                                                fitting_tree_indexer_completed_from_sampled_y.used_learned_segments_, top_n);
            std::cout<<"true sample_rate, estimated_rate, top_n are:" << sample_rate << ", " <<
                     sample_rate_estimated << ", " << top_n << std::endl;
        }
    }

    double sample_rate_estimated = sample_rate-0.05;
    data_with_gap_inserted_pos = insert_gaps(data_with_sampled_incremental_pos,
            sample_rate_estimated, fitting_tree_indexer_completed_from_sampled_y.used_learned_segments_);
    triple_MAE(sample_rate, data_with_sampled_incremental_pos, data_with_sampled_original_pos,
               data_with_gap_inserted_pos);
}



