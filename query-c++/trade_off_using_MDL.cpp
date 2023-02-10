#include <iostream>
#include <vector>
#include <stdio.h>
#include <unistd.h>
#include <gflags/gflags.h>

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
using namespace dlib;
void test_simple_linear_regression_with_mult_prev()
{
    srand(1000);
    //print_spinner();
    const int num_samples = 1000;
    ::std::vector<matrix<double>> x(num_samples);
    ::std::vector<float> y(num_samples);
    const float true_slope = 2.0;
    for ( int ii = 0; ii < num_samples; ++ii )
    {
        const double val = static_cast<double>(ii-500)/100;
        matrix<double> tmp(1,1);
        tmp = val;
        x[ii] = tmp;
        y[ii] = ( true_slope*static_cast<float>(val*val));
    }

    randomize_samples(x,y);

    using net_type = loss_mean_squared<fc<1, mult_prev1<fc<2,tag1<fc<2,input<matrix<double>>>>>>>>;
    net_type net;
    sgd defsolver(0,0.9);
    dnn_trainer<net_type> trainer(net, defsolver);
    trainer.set_learning_rate(1e-5);
    trainer.set_min_learning_rate(1e-11);
    trainer.set_mini_batch_size(10);
    trainer.set_max_num_epochs(2000);
    trainer.be_verbose();
    trainer.train(x, y);

    running_stats<double> rs;
    for (size_t i = 0; i < x.size(); ++i)
    {
        double val = y[i];
        double out = net(x[i]);
        rs.add(std::abs(val-out));
    }
}


/*
 * Test flags for ablation study and analytic experiments
 */
bool INSERT_GAP = false;
bool LOGICAL_MAE = false;

bool STATIC_GAPS_MAE = false; // calculate static mae after gap insertion
bool GLOBAL_RE_TRAIN = true; // in static case, re-train linear model globally if true, else re-train locally

bool ESTIMATE_AND_ALLOCATE = false;
bool DELTA_Y_BY_SEGMENTS = false;
bool COMPARE_SEGMENTS_LEARNED_BY_DIFFERENT_Y = false;
bool SPANS_OF_STATIC_SEGMENTS = false; // statistic for the spans of each segments in static case


template <typename KEY_TYPE>
void calculate_logical_mae(size_t error, float sample_rate,
                           vector<std::pair<KEY_TYPE, size_t>> &data_with_sampled_original_pos,
                           FittingTreeIndexerModified<key_type, PGMPos, double, typename std::vector<std::pair<KEY_TYPE, size_t>>::iterator> &fitting_tree_indexer_completed_from_sampled_y,
                           vector<std::pair<KEY_TYPE, size_t>> &data_with_sampled_incremental_pos,
                           vector<std::pair<KEY_TYPE, size_t>> &data_with_gap_inserted_pos);

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


// using key_type = size_t;
// using key_type = double;   // For map data
// using key_type_transformed = double_t;

int main(int argc, char* argv[]) {
    char buf[200];
    getcwd(buf, sizeof(buf));
    printf("current working directory : %s\n", buf);

    //test_simple_linear_regression_with_mult_prev();
    cxxopts::Options options("Trade_off_using_MDL", "Compare different trade-offs using MDL");
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
            ;
    auto opt_result = options.parse(argc, argv);


    std::map<std::string, bool> test_flags;

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
    std::sort(data_original.begin(), data_original.end());
    std::unique_copy(data_original.begin(), data_original.end(), back_inserter(data));
    size_t data_original_size = data_original.size(), data_size = data.size();
    std::vector<std::pair<key_type, size_t>> data_with_pos;
    std::vector<std::pair<key_type_transformed, size_t>> data_type_transformed_with_pos;
    std::vector<size_t> pos;
    std::cout<< "The distinct data size is " << data_size <<std::endl;
    for (size_t i = 0; i < data.size(); i++){
        std::pair<key_type, size_t> data_pair = std::make_pair(data[i], i);
        data_with_pos.emplace_back(data_pair);
        std::pair<key_type_transformed, size_t> data_type_transformed_pair = std::make_pair(key_type_transformed(data[i]), i);
        data_type_transformed_with_pos.emplace_back(data_type_transformed_pair);
        data_type_transformed.emplace_back(data[i]);
        pos.emplace_back(i);
    }


    if (test_flags["write_data_to_binary"]){
        int found_pos=data_path.find("csv",0);
        std::string out_data_path = data_path.replace(found_pos, 3, "uint64");
        write_vector_to_f<key_type>(data, out_data_path);
    }


    std::vector<double> rmi_segment_stat, fitting_segment_stat;


    std::chrono::system_clock::time_point t0, t1;
    //auto t0 = std::chrono::high_resolution_clock::now();
    //auto t1 = std::chrono::high_resolution_clock::now();

    int payload_size_in_bytes = 64 / 8; // dummy payload, only used for index size estimation
    std::vector<size_t> error_list = {16, 32, 64, 128, 256, 512, 1024};
    std::vector<double > sample_rate_list = {1};
    std::vector<double > sample_rate_list_explore = {0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05};
    std::vector<size_t > rmi_second_model_size_list = {5000, 10000, 20000, 50000, 100000, 200000};


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

    if (test_flags["Regression"]) {
        // Regression indexer based on pure machine learning algorithms
        sgd defsolver(0,0.9);
        //adam defsolver(0.0005, 0.9, 0.999);
        size_t learn_first_keys_num = size_t (data.size() * 1);
        //size_t learn_first_keys_num = data.size();
        RegressionIndexer<key_type, std::vector<key_type>::iterator> regression_indexer(data.begin(), data.begin() + learn_first_keys_num,
                                                                                    learn_first_keys_num, defsolver);
        double regular_para = 1.0, epsilon_insensive = 1, kernel_gamma = 0.1;
        //regression_indexer.set_SVR_parameters(regular_para, epsilon_insensive, kernel_gamma);
        //std::cout << " The svr parameters regular_para, epsilon_insensive and kernel_gamma are: ";
        //regression_indexer.set_optimizer(1e-5, 1e-11, 10, 2000);
        regression_indexer.set_optimizer(0.01, 0.00001, 10, 20000);
        std::cout << regular_para << ", " << epsilon_insensive << ", " << kernel_gamma << std::endl;
        regression_indexer.learn_index(data.begin(), data.begin() + learn_first_keys_num);
        std::cout << "Evaluating the regression indexer" << std::endl;
        regression_indexer.evaluate_indexer();
    }

    if (test_flags["RMI"]){
        // RMI indexer based on recursive two-stage learning algorithms
        for(float sample_rate: sample_rate_list){
            for (auto second_model_size : rmi_second_model_size_list) {
                std::cout << "Begin to construct RMI indexer, data size is:" <<
                          size_t(sample_rate * data_with_pos.size()) << std::endl;
                RMILinearIndexer<key_type_transformed, std::vector<std::pair<key_type_transformed, size_t>>::iterator>
                        rmi_indexer(data_type_transformed_with_pos.begin(), data_type_transformed_with_pos.end(),
                                    data_type_transformed_with_pos.size(), second_model_size);
                rmi_indexer.compress_key_ = false;
                // the two flags "complete_submodels_" and "find_near_seg_"
                // indicate two strategies to handle large errors when sampling case
                rmi_indexer.complete_submodels_ = false;
                rmi_indexer.find_near_seg_ = true;

                t0 = rmi_indexer.learn_index(data_type_transformed_with_pos.begin(),
                                             data_type_transformed_with_pos.end(), sample_rate);
                t1 = std::chrono::high_resolution_clock::now();
                auto construct_time = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
                std::cout << "Finished RMI indexer construction, elapse time is " << construct_time << std::endl;

                if (SPANS_OF_STATIC_SEGMENTS) {
                    std::string spans_out_f_name = "segments_spans_of_RMI-";

                    std::ostringstream file_name_seg_spans;
                    file_name_seg_spans << "/home/xxx/work/learned_index/build-release/" <<
                                        spans_out_f_name << data_name << ".int64";
                    std::string f_name_seg_spans(file_name_seg_spans.str());
                    rmi_indexer.out_segments_spans(f_name_seg_spans);
                    return 0; // we only save the segments for sample rate = 1
                }

                std::string compress = "";
                if (rmi_indexer.compress_key_) { compress = "Compress key"; }
                std::cout << "Evaluating setting for RMI indexer, " << compress << std::endl
                          << " The second_model_size,  payload_size and sample_rate are: ";
                std::cout << second_model_size << ", " << payload_size_in_bytes << ", " << sample_rate << std::endl;
                rmi_indexer.evaluate_indexer(payload_size_in_bytes, false);
            }
        }
    }

    if (test_flags["B-tree"]) {
        using btree_traits_16 = btree_map_traits<key_type_transformed, size_t, 16>;
        using btree_traits_32 = btree_map_traits<key_type_transformed, size_t, 32>;
        using btree_traits_64 = btree_map_traits<key_type_transformed, size_t, 64>;
        using btree_traits_128 = btree_map_traits<key_type_transformed, size_t, 128>;
        using btree_traits_256 = btree_map_traits<key_type_transformed, size_t, 256>;
        using btree_traits_512 = btree_map_traits<key_type_transformed, size_t, 512>;
        using btree_traits_1024 = btree_map_traits<key_type_transformed, size_t, 1024>;
        BtreeIndexer<key_type_transformed, std::vector<std::pair<key_type_transformed, size_t>>::iterator, btree_traits_16>
                btree_indexer16(data_type_transformed_with_pos.begin(), data_type_transformed_with_pos.end(), data.size(), payload_size_in_bytes);
        BtreeIndexer<key_type_transformed, std::vector<std::pair<key_type_transformed, size_t>>::iterator, btree_traits_32>
                btree_indexer32(data_type_transformed_with_pos.begin(), data_type_transformed_with_pos.end(), data.size(), payload_size_in_bytes);
        BtreeIndexer<key_type_transformed, std::vector<std::pair<key_type_transformed, size_t>>::iterator, btree_traits_64>
                btree_indexer64(data_type_transformed_with_pos.begin(), data_type_transformed_with_pos.end(), data.size(), payload_size_in_bytes);
        BtreeIndexer<key_type_transformed, std::vector<std::pair<key_type_transformed, size_t>>::iterator, btree_traits_128>
                btree_indexer128(data_type_transformed_with_pos.begin(), data_type_transformed_with_pos.end(), data.size(), payload_size_in_bytes);
        BtreeIndexer<key_type_transformed, std::vector<std::pair<key_type_transformed, size_t>>::iterator, btree_traits_256>
                btree_indexer256(data_type_transformed_with_pos.begin(), data_type_transformed_with_pos.end(), data.size(), payload_size_in_bytes);
        BtreeIndexer<key_type_transformed, std::vector<std::pair<key_type_transformed, size_t>>::iterator, btree_traits_512>
                btree_indexer512(data_type_transformed_with_pos.begin(), data_type_transformed_with_pos.end(), data.size(), payload_size_in_bytes);
        BtreeIndexer<key_type_transformed, std::vector<std::pair<key_type_transformed, size_t>>::iterator, btree_traits_1024>
                btree_indexer1024(data_type_transformed_with_pos.begin(), data_type_transformed_with_pos.end(), data.size(), payload_size_in_bytes);
        auto btree_indexers = std::make_tuple(btree_indexer16, btree_indexer32, btree_indexer64, btree_indexer128,
                btree_indexer256, btree_indexer512, btree_indexer1024);
        for_each(std::tie(btree_indexer16, btree_indexer32, btree_indexer64, btree_indexer128,
                          btree_indexer256, btree_indexer512, btree_indexer1024),
                                  [](auto &index){BtreeIndexerFunctor()(index);});
    }

    if (test_flags["PGM"]) {
        for(size_t error: error_list){
            for(float sample_rate: sample_rate_list){
                // Original PGM indexer
                std::cout << "Begin to construct Original PGM indexer" << std::endl;
                PgmIndexer<key_type_transformed, PGMPos, double, std::vector<std::pair<key_type_transformed, size_t>>::iterator>
                        pgm_indexer(data_type_transformed_with_pos.begin(), data_type_transformed_with_pos.end(),
                                    data_type_transformed_with_pos.size());
                //pgm_indexer(data_with_pos.begin(), data_with_pos.begin()+100, 100);
                std::string organize_strategy = "Recursive";
                size_t recurrsive_error = 4;
                t0 = std::chrono::high_resolution_clock::now();
                //pgm_indexer.learn_index(data_with_pos.begin(), data_with_pos.begin()+100, error, organize_strategy,
                pgm_indexer.learn_index(data_type_transformed_with_pos.begin(), data_type_transformed_with_pos.end(),
                                        error, organize_strategy, recurrsive_error, sample_rate);
                t1 = std::chrono::high_resolution_clock::now();
                auto construct_time = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
                std::cout << "Finished Original PGM indexer construction, elapse time is " << construct_time << std::endl;
                std::cout << "Evaluating setting for Original PGM indexer" << std::endl
                          << " The error, recurrsive_error, payload_size and sample_rate are: ";
                std::cout << error << ", " << recurrsive_error << ", " << payload_size_in_bytes << ", " << sample_rate << std::endl;
                pgm_indexer.evaluate_indexer(payload_size_in_bytes);


            }
        }
    }

    if (test_flags["Fitting-tree"]) {
        // Fitting tree indexer build
        for(size_t error: error_list){
            for(float sample_rate: sample_rate_list){
                std::vector<std::pair<key_type, size_t>> data_with_sampled_incremental_pos, data_with_gap_inserted_pos;
                auto data_with_sampled_original_pos = data_type_transformed_with_pos;

                std::cout << "Begin to construct Original FITing-Tree indexer" << std::endl;
                FittingTreeIndexer<key_type_transformed, PGMPos, double, std::vector<std::pair<key_type_transformed, size_t>>::iterator>
                        fitting_tree_indexer(data_type_transformed_with_pos.begin(), data_type_transformed_with_pos.end(), data_type_transformed_with_pos.size());
                t0 = fitting_tree_indexer.learn_index(data_with_sampled_original_pos.begin(), data_with_sampled_original_pos.end(), error, sample_rate);
                t1 = std::chrono::high_resolution_clock::now();
                auto construct_time = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
                std::cout << "Finished Fitting-tree indexer construction, elapse time is " << construct_time << std::endl;
                std::cout << "Evaluating setting for Fitting-tree indexer" << std::endl
                          << " The size of key, payload_size, error_bound and sample_rate are: ";
                std::cout << sizeof(key_type) << ", " << payload_size_in_bytes << ", " << error << ", "<<sample_rate<< std::endl;
                fitting_tree_indexer.evaluate_indexer(payload_size_in_bytes);

            }
        }
    }

    if(test_flags["fitting_vs_rmi"]){
        assert(fitting_segment_stat.size() == rmi_segment_stat.size());
        std::vector<double> segment_stat_delta;
        for (int j = 0; j < fitting_segment_stat.size(); ++j) {
            double delta = fitting_segment_stat[j] > rmi_segment_stat[j] ? (fitting_segment_stat[j] - rmi_segment_stat[j]) :
                           (rmi_segment_stat[j] - fitting_segment_stat[j]);
            segment_stat_delta.emplace_back(delta);
        }
        std::cout<<"Fitting segment stats:"<<std::endl;
        calculate_mean_std(fitting_segment_stat);
        std::cout<<"RMI segment stats:"<<std::endl;
        calculate_mean_std(rmi_segment_stat);
        std::cout<<"Their segment delta stats:"<<std::endl;
        calculate_mean_std(segment_stat_delta);
    }


    return 0;
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



