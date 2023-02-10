#include <iostream>
#include <random>
#include <vector>
#include <stdio.h>
#include <unistd.h>
#include <gflags/gflags.h>
#include <ALEX-RMI.hpp>

#include "DataProcessor.hpp"
#include "PgmIndexer.hpp"
#include "METIndexer.h"
// #include "PgmIndexer_Modified.hpp"
// #include "PgmIndexer_Gapped.hpp"
// #include "FitingTreeIndexer.hpp"
// #include "FitingTreeIndexer_Modified.hpp"
#include "RegressionIndexer.hpp"
#include "include/radix_spline/RadixSplineIndexer.h"
#include "BtreeIndexer.hpp"
#include "RMIIndexerLinear.hpp"
// #include <alex/core_int/alex.h>
// #include <alex/benchmark_helper.hpp>
#include <cxxopts.hpp>
#include "Utilities.hpp"


using size_t = unsigned long;
using namespace std;

using data_type = std::pair<key_type_transformed, payload_type>;

int main(int argc, char *argv[]) {
    char buf[200];
    getcwd(buf, sizeof(buf));
    printf("current working directory : %s\n", buf);

    /**
     * exp setting init
     */
    cxxopts::Options options("Dynamic-Epsilon-Lookup",
                             "Test the performance of learned index with dynamic epsilon on static case.");
    options.add_options()
            // Baseline Methods
            ("pgm", "test pgm indexer", cxxopts::value<bool>()->default_value("false"))
            ("fiting-tree", "test fiting-tree indexer",
             cxxopts::value<bool>()->default_value("false"))
            ("radix-spline", "test radix-spline indexer",
             cxxopts::value<bool>()->default_value("false"))
            ("met", "test met indexer",
             cxxopts::value<bool>()->default_value("false"))

            // Datasets
            ("iot", "test iot dataset", cxxopts::value<bool>()->default_value("false"))
            ("web", "test weblog dataset", cxxopts::value<bool>()->default_value("false"))
            ("longtitude", "test longtitude map dataset", cxxopts::value<bool>()->default_value("false"))
            ("longtitude_200M", "test longtitude_200M dataset", cxxopts::value<bool>()->default_value("false"))
            ("latilong", "test latilong map dataset", cxxopts::value<bool>()->default_value("false"))
            ("lognormal", "test lognormal dataset", cxxopts::value<bool>()->default_value("false"))
            ("evt", "test evt dataset (adversarial case)", cxxopts::value<bool>()->default_value("false"))
            ("new_evt", "re-generate evt dataset", cxxopts::value<bool>()->default_value("false"))
            ("s,save_data_binary", "read data from tsv, then save to binary form",
             cxxopts::value<bool>()->default_value("false"))
            ("r,read_data_binary", "read data from binary file",
             cxxopts::value<bool>()->default_value("true"))

            // Model Settings
            ("epsilon", "tested epsilon list, separated by ',' ",
             cxxopts::value<std::vector<size_t >>()->default_value("64,128,256"))
            ("sample_rates", "tested sample rate list, separated by ',' ",
             cxxopts::value<std::vector<double>>()->default_value("1.0,0.8,0.4"))

            // Exp Settings
            ("seed", "random seed", cxxopts::value<int>()->default_value("1234"));

    auto opt_result = options.parse(argc, argv);

    std::unordered_map<std::string, bool> test_flags;

    // models
    test_flags["PGM"] = opt_result["pgm"].as<bool>();
    test_flags["fiting-tree"] = opt_result["fiting-tree"].as<bool>();
    test_flags["radix-spline"] = opt_result["radix-spline"].as<bool>();
    test_flags["met"] = opt_result["met"].as<bool>();

    // dataset
    test_flags["iot"] = opt_result["iot"].as<bool>();
    test_flags["web"] = opt_result["web"].as<bool>();
    test_flags["longtitude"] = opt_result["longtitude"].as<bool>();
    test_flags["longtitude_200M"] = opt_result["longtitude_200M"].as<bool>();
    test_flags["latilong"] = opt_result["latilong"].as<bool>();
    test_flags["lognormal"] = opt_result["lognormal"].as<bool>();

    //random seed
    int random_seed = opt_result["seed"].as<int>();
    srand(random_seed);
    test_flags["write_data_to_binary"] = opt_result["s"].as<bool>();
    test_flags["read_from_binary"] = opt_result["r"].as<bool>();
    for (auto flag : test_flags) {
        std::cout << flag.first << ": " << flag.second << std::endl;
    }


    /**
     *  data preparation
     */
    std::vector<key_type> data_original;
    std::vector<key_type> data;
    std::vector<key_type_transformed> data_type_transformed;
    std::string data_path;
    std::string dataset_name;
    if (test_flags["iot"]) {
        IotProcessor iotProcessor;
        if (test_flags["read_from_binary"]) {
            data_path = "/home/xxx/work/learned_index/data/iot.uint64";
            data_path = "/home/xxx/work/learned_index/data/iot_unix.uint64";
            data = iotProcessor.read_data_binary(data_path, 30000000);
        } else {
            data_path = "/home/xxx/work/learned_index/data/iot.csv";
            data_original = iotProcessor.read_data_csv(data_path, 30000000);
        }
        dataset_name = "IoT";
    } else if (test_flags["web"]) {
        WebBlogsProcessor webProcessor;
        if (test_flags["read_from_binary"]) {
            data_path = "/home/xxx/work/learned_index/data/weblogs.uint64";
            data_path = "/home/xxx/work/learned_index/data/weblogs_unix.uint64";
            data = webProcessor.read_data_binary(data_path, 720000000);
        } else {
            data_path = "/home/xxx/work/learned_index/data/weblogs.csv";
            data_original = webProcessor.read_data_csv(data_path, 720000000);
        }
        dataset_name = "Weblogs";
    } else if (test_flags["longtitude_200M"]) {
        MapProcessor mapProcessor;
        data_path = "/home/xxx/work/learned_index/data/longitudes-200M.bin.data";
        mapProcessor.read_data_binary(data, data_path, 720000000, 200000000);
    } else if (test_flags["longtitude"]) {
        MapProcessor mapProcessor;
        //data_path = "/home/xxx/work/learned_index/data/longtitude.f64";
        data_path = "/home/xxx/work/learned_index/data/longtitude_newyork.double";
        data_path = "/home/xxx/work/learned_index/data/longtitude_china.double";
        if (test_flags["read_from_binary"]) {
            mapProcessor.read_data_binary(data, data_path, 720000000);
        }
    } else if (test_flags["latilong"]) {
        MapProcessor mapProcessor;
        //data_path = "/home/xxx/work/learned_index/data/latilong.f64";
        data_path = "/home/xxx/work/learned_index/data/latilong_newyork.double";
        data_path = "/home/xxx/work/learned_index/data/latilong_china.double";
        if (test_flags["read_from_binary"]) {
            mapProcessor.read_data_binary(data, data_path, 720000000);
        }
        dataset_name = "Map";
    } else if (test_flags["lognormal"]) {
        LognormalProcessor lognormalProcessor;
        //data_path = "/home/xxx/work/learned_index/data/lognormal_20M.uint64";
        data_path = "/home/xxx/work/learned_index/data/lognormal_1B.double";
        data_path = "/home/xxx/work/learned_index/MetaLearnedIndex/data/lognormal.double";
        std::ifstream fin(data_path);
        if (!fin) {
            size_t synthetic_data_size = 1000000000; // 1 billion
            //size_t synthetic_data_size = 20000000;
            //lognormalProcessor.generate_data_save_binary(synthetic_data_size, data_path, 0.0, 2.0, false);
            //lognormalProcessor.generate_data_save_binary(synthetic_data_size, data_path, 0.0, 0.5, true, 1e8);
            std::cout << "Not found lognormal dataset, please check the data path!" << std::endl;
        }
        if (test_flags["read_from_binary"]) {
            data = lognormalProcessor.read_data_binary(data_path, 7200000000);
        }
        dataset_name = "Lognormal";
    } else if (test_flags["evt"]) {
        ExtremelyVariedIntervalProcessor evtProcessor;
        data_path = "/home/xxx/work/learned_index/data/evt.uint64";
        std::ifstream fin(data_path);
        if (!fin or test_flags["new_evt"]) {
            size_t repeated_num = 100;
            size_t moderate_data_num = 3;
            size_t generated_epsilon = 32;
            evtProcessor.generate_data_save_binary(repeated_num, data_path, moderate_data_num, generated_epsilon);
        }
        if (test_flags["read_from_binary"]) {
            data = evtProcessor.read_data_binary(data_path, 7200000000);
        } else {
            std::cout << "You should choose one dataset to test." << std::endl;
        }
    }

    if (not test_flags["read_from_binary"]) { // the saved binary file has been sorted and removed duplicates
        std::sort(data_original.begin(), data_original.end());
        std::unique_copy(data_original.begin(), data_original.end(), back_inserter(data));
    }
    if (test_flags["write_data_to_binary"]) {
        int found_pos = data_path.find("csv", 0);
        std::string out_data_path = data_path.replace(found_pos, 3, "uint64");
        write_vector_to_f<key_type>(data, out_data_path);
    }

    size_t data_size = data.size();
    std::cout << "The distinct data size is " << data_size << std::endl;
    std::vector<data_type> all_keys_type_transformed_with_pos;
    for (size_t i = 0; i < data_size; i++) {
        auto tmp_data = key_type_transformed(data[i]);
        //init_keys_type_transformed.emplace_back(tmp_data);
        data_type data_type_transformed_pair = std::make_pair(tmp_data, i);
        all_keys_type_transformed_with_pos.emplace_back(data_type_transformed_pair);
    }

    // Model settings
    std::vector<size_t> epsilon_list = {16, 32, 64, 128, 256, 512, 1024};
    std::vector<double> sample_rate_list = {1};

    epsilon_list = opt_result["epsilon"].as<std::vector<size_t>>();

    /**
     * test learned index with fixed epsilon and dynamic epsilon
     */
    // bool use_complete_segments = false; // in non-sample case, we do not use the completed segments by default.
    for (double epsilon : epsilon_list) {
        if (test_flags["radix-spline"]) {
            RadixSplineIndexer<key_type_transformed, PGMPos, double, std::vector<std::pair<key_type_transformed, size_t>>::iterator, payload_type>
                    radix_original_indexer(all_keys_type_transformed_with_pos.begin(),
                                           all_keys_type_transformed_with_pos.end(),
                                           all_keys_type_transformed_with_pos.size());
            std::cout << std::endl << "Begin to construct original RadixSpline index" << std::endl;
            radix_original_indexer.learn_index(all_keys_type_transformed_with_pos.begin(),
                                               all_keys_type_transformed_with_pos.end(),
                                               epsilon);
            std::cout << "Evaluate the original RadixSpline: " << std::endl;
            radix_original_indexer.evaluate_indexer(all_keys_type_transformed_with_pos, "", true, true);


            RadixSplineIndexer<key_type_transformed, PGMPos, double, std::vector<std::pair<key_type_transformed, size_t>>::iterator, payload_type>
                    radix_dynamic_indexer(all_keys_type_transformed_with_pos.begin(),
                                          all_keys_type_transformed_with_pos.end(),
                                          all_keys_type_transformed_with_pos.size());
            std::cout << std::endl << "Begin to construct dynamic RadixSpline index" << std::endl;
            // Load dynamic epsilons generated from python files
            std::ostringstream epsilon_data_path_f;
            epsilon_data_path_f << "/home/xxx/work/learned_index/data/varied_epsilons/"
                                << dataset_name
                                << "_radix_" << static_cast<unsigned int>(epsilon) << ".double";
            std::string epsilon_data_path_f_name(epsilon_data_path_f.str());
            MapProcessor epsilonProcessor;
            std::vector<key_type> seg_epsilons;
            epsilonProcessor.read_data_binary(seg_epsilons, epsilon_data_path_f_name, 7200000000, -1, false);
            basic_statistic(seg_epsilons);
            radix_dynamic_indexer.set_dynamic_epsilons(seg_epsilons);

            radix_dynamic_indexer.learn_index(all_keys_type_transformed_with_pos.begin(),
                                              all_keys_type_transformed_with_pos.end(),
                                              epsilon);
            std::cout << "Evaluate the dynamic RadixSpline: " << std::endl;
            radix_dynamic_indexer.evaluate_indexer(all_keys_type_transformed_with_pos, "", true, true);
        }

        if (test_flags["met"]) {
            METIndexer<key_type_transformed, PGMPos, double, std::vector<std::pair<key_type_transformed, size_t>>::iterator, payload_type>
                    met_original_indexer(all_keys_type_transformed_with_pos.begin(),
                                         all_keys_type_transformed_with_pos.end(),
                                         all_keys_type_transformed_with_pos.size());
            std::cout << std::endl << "Begin to construct original MET index" << std::endl;
            met_original_indexer.learn_index(all_keys_type_transformed_with_pos.begin(),
                                             all_keys_type_transformed_with_pos.end(),
                                             epsilon);
            std::cout << "Evaluate the original MET: " << std::endl;
            met_original_indexer.evaluate_indexer(all_keys_type_transformed_with_pos, "", true, true);


            METIndexer<key_type_transformed, PGMPos, double, std::vector<std::pair<key_type_transformed, size_t>>::iterator, payload_type>
                    met_dynamic_indexer(all_keys_type_transformed_with_pos.begin(),
                                        all_keys_type_transformed_with_pos.end(),
                                        all_keys_type_transformed_with_pos.size());
            std::cout << std::endl << "Begin to construct dynamic MET index" << std::endl;
            // Load dynamic epsilons generated from python files
            std::ostringstream epsilon_data_path_f;
            epsilon_data_path_f << "/home/xxx/work/learned_index/data/varied_epsilons/"
                                << dataset_name
                                << "_met_" << static_cast<unsigned int>(epsilon) << ".double";
            std::string epsilon_data_path_f_name(epsilon_data_path_f.str());
            MapProcessor epsilonProcessor;
            std::vector<key_type> seg_epsilons;
            epsilonProcessor.read_data_binary(seg_epsilons, epsilon_data_path_f_name, 7200000000, -1, false);
            basic_statistic(seg_epsilons);
            met_dynamic_indexer.set_dynamic_epsilons(seg_epsilons);

            met_dynamic_indexer.learn_index(all_keys_type_transformed_with_pos.begin(),
                                            all_keys_type_transformed_with_pos.end(),
                                            epsilon);
            std::cout << "Evaluate the dynamic MET: " << std::endl;
            met_dynamic_indexer.evaluate_indexer(all_keys_type_transformed_with_pos, "", true, true);
        }



        if (test_flags["PGM"]) {
            PgmIndexer<key_type_transformed, PGMPos, double, std::vector<std::pair<key_type_transformed, size_t>>::iterator>
                    pgm_original_indexer(all_keys_type_transformed_with_pos.begin(),
                                         all_keys_type_transformed_with_pos.end(),
                                         all_keys_type_transformed_with_pos.size());
            std::cout << std::endl << "Begin to construct original PGM index" << std::endl;
            pgm_original_indexer.learn_index(all_keys_type_transformed_with_pos.begin(),
                                             all_keys_type_transformed_with_pos.end(),
                                             epsilon,
                                             "Recursive",
                                             8);
            std::cout << "Evaluate the original PGM: " << std::endl;
            pgm_original_indexer.evaluate_indexer(128, "", true);

            // PGMIndexer<key_type_transformed, PGMPos, double, std::vector<std::pair<key_type_transformed, size_t>>::iterator, payload_type>
            //         pgm_dynamic_indexer(all_keys_type_transformed_with_pos.begin(),
            //                             all_keys_type_transformed_with_pos.end(),
            //                             all_keys_type_transformed_with_pos.size());
            // std::cout << std::endl << "Begin to construct dynamic PGM index" << std::endl;
            // // Load dynamic epsilons generated from python files
            // std::ostringstream epsilon_data_path_f;
            // epsilon_data_path_f << "/home/xxx/work/learned_index/data/varied_epsilons/"
            //                     << dataset_name
            //                     << "_pgm_" << static_cast<unsigned int>(epsilon) << ".double";
            // std::string epsilon_data_path_f_name(epsilon_data_path_f.str());
            // MapProcessor epsilonProcessor;
            // std::vector<key_type> seg_epsilons;
            // epsilonProcessor.read_data_binary(seg_epsilons, epsilon_data_path_f_name, 7200000000, -1, false);
            // basic_statistic(seg_epsilons);
            // pgm_dynamic_indexer.set_dynamic_epsilons(seg_epsilons);

            // pgm_dynamic_indexer.learn_index(all_keys_type_transformed_with_pos.begin(),
            //                                 all_keys_type_transformed_with_pos.end(),
            //                                 epsilon);
            // std::cout << "Evaluate the dynamic PGM: " << std::endl;
            // pgm_dynamic_indexer.evaluate_indexer(all_keys_type_transformed_with_pos, "", true, true);
        }

    }

    return 0;
}
