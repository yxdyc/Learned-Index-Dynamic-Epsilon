#ifndef LEARNED_INDEX_UTILITIES_HPP
#define LEARNED_INDEX_UTILITIES_HPP


#include <vector>
#include <list>
#include <chrono>
#include <cstddef>
#include <iterator>
#include <unordered_set>
#include <dlib/matrix.h>
#include <dlib/statistics.h>
#include <random>
#include <experimental/algorithm>
#include "IndexMechanism.hpp"
//#include "DataProcessor.hpp"
#include <numeric>
#include <queue>
#include <type_traits>
#include <typeinfo>

#ifndef _MSC_VER
#include <cxxabi.h>
#endif

#include <memory>
#include <string>
#include <cstdlib>
#include <cmath>


#define LIKELY(x)   __builtin_expect(!!(x), 1)
#define UNLIKELY(x) __builtin_expect(!!(x), 0)


#ifdef FLOAT_KEY
using key_type = double;  //map data
#else
using key_type = size_t;
#endif
using key_type_transformed = double_t;
using payload_type = size_t;


// templates used for comparison among integers
template<class T>
bool isEqual(T a, T b, typename std::enable_if<std::is_integral<T>::value>::type * = 0) {
    return a == b;
}

// templates used for comparison among floatings
template<class T>
bool isEqual(T a, T b, typename std::enable_if<std::is_floating_point<T>::value>::type * = 0) {
    return std::abs(a - b) < std::numeric_limits<T>::epsilon();
}


/***
 *  I/O related helper functions
 */

template<typename key_type>
void write_vector_to_f(const std::vector<key_type> &data, const std::string &out_data_path) {
    std::cout << "Begin to write to binary file: " << out_data_path << std::endl;
    std::ofstream fout;
    fout.open(out_data_path, std::ios::binary);
    size_t data_size = data.size();
    fout.write((char *) &data_size, sizeof(key_type));
    long count_i = 0;
    for (key_type data_item : data) {
        // if (count_i % 500000 == 0) std::cout<<"Write line: "<<count_i<<std::endl;
        fout.write((char *) &data_item, sizeof(key_type));
        count_i++;
    }
    std::cout << "Finished to write to binary file: " << out_data_path << std::endl;
}

void write_predictions(const std::vector<double> &x, const std::vector<double> &y, const std::vector<double> &y_hat,
                       const std::string &s);

void write_predictions(const std::vector<double> &x, const std::vector<double> &y, const std::vector<double> &y_hat,
                       const std::string &s) {
    std::ofstream outFile;
    std::cout << "write predictions to file:" << s << std::endl;
    outFile.open(s, std::ios::out);
    for (size_t i = 0; i < x.size(); i++) {
        std::stringstream stream;
        stream << std::fixed << std::setw(20) << std::setprecision(6) << x[i] << "," << y[i] << "," << y_hat[i]
               << std::endl;
        outFile << stream.str();
    }
}

template<typename KEY_TYPE>
void write_predictions(const std::vector<std::pair<KEY_TYPE, size_t>> &y,
                       const std::vector<std::pair<KEY_TYPE, size_t>> &y_hat, const std::string &s,
                       size_t write_begin = 0,
                       size_t wirte_end = 1) {
    std::ofstream outFile;
    wirte_end = y.size();
    std::cout << "write predictions to file:" << s << std::endl;
    outFile.open(s, std::ios::out);
    for (int i = write_begin; i < wirte_end; i++) {
        std::stringstream stream;
        stream << std::fixed << std::setw(20) << std::setprecision(6) << y[i].first << "," << y[i].second << ","
               << y_hat[i].second << std::endl;
        outFile << stream.str();
    }
}




/***
 *  evaluation related helper functions
 */

template<typename KeyType>
dlib::matrix<double, 1, 4>
evaluate_regression_ML(const std::vector<KeyType> &all_predicted_pos, const std::vector<size_t> &all_true_pos,
                       size_t error = 10000, bool write_to_file = false, std::string file_name = "") {
    dlib::running_stats<double> rs, rs_mae;
    dlib::running_scalar_covariance<double> rc;
    double max_diff = 0.0;

    for (unsigned long i = 0; i < all_predicted_pos.size(); ++i) {
        // compute error
        //const double temp = (all_predicted_pos[i].pos >= all_true_pos[i]) ?
        //      (all_predicted_pos[i].pos - all_true_pos[i]) : (all_true_pos[i] - all_predicted_pos[i].pos);
        const double temp = (all_predicted_pos[i] >= all_true_pos[i]) ?
                            (all_predicted_pos[i] - all_true_pos[i]) : (all_true_pos[i] - all_predicted_pos[i]);
        max_diff = temp > max_diff ? temp : max_diff;

        rs_mae.add(std::abs(temp));
        rs.add(temp * temp);
        rc.add(double(all_predicted_pos[i]), double(all_true_pos[i]));
    }
    std::cout << "Max |y-y'| is " << max_diff << ". ";

    dlib::matrix<double, 1, 4> result;
    result = rs.mean(), rc.correlation(), rs_mae.mean(), rs_mae.stddev();
    size_t reasonable_predi_count = 0, unreasonable_predi_count = 0;
    double reasonable_sum = 0, unreasonable_sum = 0;
    if (rs_mae.mean() > error + 1) {
        std::vector<double> x, y, y_hat;
        for (unsigned long i = 0; i < all_predicted_pos.size(); ++i) {
            // compute error
            const double temp = (all_predicted_pos[i] >= all_true_pos[i]) ?
                                (all_predicted_pos[i] - all_true_pos[i]) : (all_true_pos[i] - all_predicted_pos[i]);
            double delta = std::abs(temp);
            if (delta >= error) {
                unreasonable_sum += delta;
                unreasonable_predi_count++;
                x.emplace_back(i);
                y.emplace_back(all_true_pos[i]);
                y_hat.emplace_back(all_predicted_pos[i]);
            } else {
                reasonable_sum += delta;
                reasonable_predi_count++;
            }
        }
        std::cout << "Larger than " << error << " is: " << unreasonable_sum / unreasonable_predi_count << " count:"
                  << unreasonable_predi_count << std::endl;
        std::cout << "Smaller than " << error << " is: " << reasonable_sum / reasonable_predi_count << " count:"
                  << reasonable_predi_count << std::endl;
        std::string s;
        if (file_name == "") {
            std::ostringstream tmp_file_name;
            tmp_file_name << all_predicted_pos.size() << "-" << unreasonable_predi_count << ".tsv";
            s = tmp_file_name.str();
        } else {
            s = file_name;
        }
        if (write_to_file) {
            write_predictions(x, y, y_hat, s);
        }
    }
    return result;
}

template<typename data_type>
void basic_statistic(std::vector<data_type> v, bool print_res = true) {
    double sum = std::accumulate(v.begin(), v.end(), 0.0);
    double mean = sum / v.size();

    std::vector<double> diff(v.size());
    std::transform(v.begin(), v.end(), diff.begin(),
                   std::bind2nd(std::minus<double>(), mean));
    double sq_sum = std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
    double variance = sq_sum / v.size();
    double stdev = std::sqrt(variance);

    std::sort(v.begin(), v.end());
    auto max = v[v.size() - 1];
    auto min = v[0];
    auto median = v[floor(v.size() * 0.5)];
    auto quartile_1 = v[floor(v.size() * 0.25)];
    auto quartile_3 = v[floor(v.size() * 0.75)];

    if (print_res) {
        std::cout << "Size is " << v.size() << ", Max is " << max << ", Quartile_3 is " << quartile_3 <<
                  ", Median is " << median << ", Quartile_1 is " << quartile_1 << ", Min is " << min <<
                  ", Mean is " << mean << ", Std is " << stdev;
        //std::cout << std::setprecision(14) << ", Variance is " << variance << std::endl;
        std::cout << ", Variance is " << variance;
        std::cout << ", Sum is " << sum << std::endl;
    }
}

std::pair<double, double> calculate_mean_std(const std::vector<double> v, bool print_res = true) {
    double sum = std::accumulate(v.begin(), v.end(), 0.0);
    double mean = sum / v.size();

    std::vector<double> diff(v.size());
    std::transform(v.begin(), v.end(), diff.begin(),
                   std::bind2nd(std::minus<double>(), mean));
    double sq_sum = std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
    double stdev = std::sqrt(sq_sum / v.size());
    std::pair<double, double> res;
    res.first = mean;
    res.second = stdev;
    size_t n_of_largerthan1 = 0;
    double max = 0.0;
    for (auto k : v) {
        if (k > 1) {
            n_of_largerthan1++;
        }
        if (k > max) {
            max = k;
        }
    }
    if (print_res) {
        std::cout << "Size is " << v.size() << ", # of > 1 is " << n_of_largerthan1 <<
                  ", Sum is " << sum << ", Mean is " << mean << ", std is " << stdev << ", Max is " << max << std::endl;
    }
    return res;
}

std::pair<double, double> calculate_mean_std(const std::vector<size_t> v_data, bool print_res = true) {
    std::vector<double> v;
    for (auto v_i : v_data) { v.push_back(v_i); }
    double sum = std::accumulate(v.begin(), v.end(), 0.0);
    double mean = sum / v.size();

    std::vector<double> diff(v.size());
    std::transform(v.begin(), v.end(), diff.begin(),
                   std::bind2nd(std::minus<double>(), mean));
    double sq_sum = std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
    double stdev = std::sqrt(sq_sum / v.size());
    std::pair<double, double> res;
    res.first = mean;
    res.second = stdev;
    size_t n_of_largerthan1 = 0;
    for (auto k : v) {
        if (k > 1) {
            n_of_largerthan1++;
        }
    }
    if (print_res) {
        std::cout << "Size is " << v_data.size() << ", # of > 1 is " << n_of_largerthan1
                  << ", Sum is " << sum << ", Mean is " << mean << ", std is " <<
                  stdev << std::endl;
    }
    return res;
}

// estimate the gaussian_mean_std using Maximum Likelihood Estimation
std::pair<double, double> calculate_gaussian_mean_std(const std::vector<double> v) {
    double sum = std::accumulate(v.begin(), v.end(), 0.0);
    double mean = sum / v.size();

    double product_sum = std::inner_product(v.begin(), v.end(), v.begin(), 0.0);
    double stdev = std::sqrt(product_sum / v.size() - (mean * mean));
    std::pair<double, double> res;
    res.first = mean;
    res.second = stdev;
    // std::cout<< "Gaussian Mean is " << mean <<", Gaussian std is "<< stdev<<std::endl;
    return res;
}


/*
 * calculated the MAE between y1 and y2, for data_with_pos format
 */
template<typename KEY_TYPE>
void pair_MAE(std::vector<std::pair<KEY_TYPE, size_t>> &data_with_pos_y1,
              std::vector<std::pair<KEY_TYPE, size_t>> &data_with_pos_y2) {
    std::vector<size_t> y_2, y_1;
    for (size_t i = 0; i < data_with_pos_y1.size(); ++i) {
        y_1.emplace_back(data_with_pos_y1[i].second);
        y_2.emplace_back(data_with_pos_y2[i].second);
    }
    dlib::matrix<double, 1, 4> results;
    results = evaluate_regression_ML(y_1, y_2, 10000, false, "");
    std::cout << "ML oriented matricx for y1 and y2: " << results << std::endl;
}


/*
 * calculated the MAE between y1 and y2, y3 and y2, for data_with_pos format
 */
template<typename KEY_TYPE>
void triple_MAE(float sample_rate, std::vector<std::pair<KEY_TYPE, size_t>> &data_with_pos_y1,
                std::vector<std::pair<KEY_TYPE, size_t>> &data_with_pos_y2,
                std::vector<std::pair<KEY_TYPE, size_t>> &data_with_pos_y3) {
    std::vector<size_t> y_2, y_1, y_3, temp_y_inserted, temp_y_true;
    for (int i = 0; i < data_with_pos_y1.size(); ++i) {
        y_1.emplace_back(data_with_pos_y1[i].second);
        y_2.emplace_back(data_with_pos_y2[i].second);
        y_3.emplace_back(data_with_pos_y3[i].second);
    }
    dlib::matrix<double, 1, 4> results;
    results = evaluate_regression_ML(y_1, y_2, 10000, false, "");
    std::cout << "ML oriented matricx for y1 and y2: " << results << std::endl;

    results = evaluate_regression_ML(y_3, y_2, 10000, false, "");
    std::cout << "ML oriented matricx for y3 and y2: " << results << std::endl;


    bool analysis_different_ranges = false;
    bool save_y_res_to_binary = false;
    if (analysis_different_ranges) {
        int begin, end;
        begin = 0, end = int(float(y_2.size()) / 3);
        temp_y_inserted = std::vector<size_t>(y_3.begin() + begin, y_3.begin() + end);
        temp_y_true = std::vector<size_t>(y_2.begin() + begin, y_2.begin() + end);
        results = evaluate_regression_ML(temp_y_inserted, temp_y_true, 10000, false, "");
        std::cout << "ML oriented matricx for gap inseted res: " << results << ", range: " << begin << ": " << end
                  << std::endl;
        begin = int(float(y_2.size()) / 3), end = int(float(y_2.size()) / 3 * 2);
        temp_y_inserted = std::vector<size_t>(y_3.begin() + begin, y_3.begin() + end);
        temp_y_true = std::vector<size_t>(y_2.begin() + begin, y_2.begin() + end);
        results = evaluate_regression_ML(temp_y_inserted, temp_y_true, 10000, false, "");
        std::cout << "ML oriented matricx for gap inseted res: " << results << ", range: " << begin << ": " << end
                  << std::endl;
        begin = int(float(y_2.size()) / 3 * 2), end = int(y_2.size());
        temp_y_inserted = std::vector<size_t>(y_3.begin() + begin, y_3.begin() + end);
        temp_y_true = std::vector<size_t>(y_2.begin() + begin, y_2.begin() + end);
        results = evaluate_regression_ML(temp_y_inserted, temp_y_true, 10000, false, "");
        std::cout << "ML oriented matricx for gap inseted res: " << results << ", range: " << begin << ": " << end
                  << std::endl;
    }
    if (save_y_res_to_binary) {
        std::vector<size_t> delta_y;
        delta_y.reserve(y_3.size());
        for (unsigned long i = 0; i < y_3.size(); ++i) {
            size_t temp = (y_3[i] >= y_2[i]) ?
                          (y_3[i] - y_2[i]) : (y_2[i] - y_3[i]);
            delta_y.emplace_back(temp);
        }
        std::ostringstream delta_f;
        delta_f << "/home/xxx/work/learned_index/build-release/prediction_res/delta_inserted_y" <<
                sample_rate << ".int64";
        std::string delta_f_name(delta_f.str());
        write_vector_to_f<size_t>(delta_y, delta_f_name);

        std::ostringstream sampled_y_f;
        sampled_y_f << "/home/xxx/work/learned_index/build-release/prediction_res/sampled_y" <<
                    sample_rate << ".int64";
        std::string sampled_y_name(sampled_y_f.str());
        write_vector_to_f<size_t>(y_2, sampled_y_name);

        std::ostringstream inserted_y_f;
        inserted_y_f << "/home/xxx/work/learned_index/build-release/prediction_res/inserted_y" <<
                     sample_rate << ".int64";
        std::string inserted_y_name(inserted_y_f.str());
        write_vector_to_f<size_t>(y_3, inserted_y_name);
    }
}

std::vector<double> statistic_of_delta_y(std::vector<size_t> sampled_y, bool print_res = true, bool pair_wise = false) {
    std::vector<double> delta_y;
    size_t sample_size = sampled_y.size();
    std::vector<double> res = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    if (sample_size == 0 or sample_size == 1) {
        return res;
    }
    if (print_res) {
        std::cout << "Statistic of delta_y, " << std::endl;
    }
    std::sort(sampled_y.begin(), sampled_y.end());
    delta_y.reserve(sample_size);
    if (pair_wise) {
        assert(sample_size % 2 == 0);
        for (size_t i = 0; i < sample_size - 1; i += 2) {
            delta_y.emplace_back(double(sampled_y[i + 1] - sampled_y[i]));
            //delta_y.emplace_back(log(double(sampled_y[i+1] - sampled_y[i])));
        }
    } else {
        for (size_t i = 0; i < sample_size - 1; i++) {
            delta_y.emplace_back(double(sampled_y[i + 1] - sampled_y[i]));
            //delta_y.emplace_back(log(double(sampled_y[i+1] - sampled_y[i])));
        }
    }
    int number = delta_y[0];
    int mode = number, count = 1, countMode = 1;
    for (size_t i = 0; i < delta_y.size(); i++) {
        if (delta_y[i] == number) {
            count++;
        } else {
            if (count > countMode) {
                countMode = count;
                mode = number;
            }
            count = 1;
            number = delta_y[i];
        }
    }

    auto mean_std_res = calculate_mean_std(delta_y, print_res);
    size_t middle = int(sample_size * 0.5);
    std::sort(delta_y.begin(), delta_y.end());
    res.emplace_back(mean_std_res.first);
    res.emplace_back(mean_std_res.second);
    res.emplace_back(delta_y[0]);
    res.emplace_back(delta_y[sample_size - 2]);
    res.emplace_back(delta_y[middle]);
    res.emplace_back(mode);
    if (print_res) {
        std::cout << "Min, Max, Median, Mode are: " << delta_y[0] << ", " << delta_y[sample_size - 2] << ", "
                  << delta_y[middle] << ", " << mode << std::endl;
    }
    return res;
}




/***
 * sampling
 */

template<typename data_type>
std::vector<data_type> random_sample_without_replacement(const std::vector<data_type> &source,
                                                         int newpopsize, bool hold_head_and_tail = false) {
    assert(newpopsize > -1 && newpopsize <= source.size());
    std::vector<data_type> result;
    result.reserve(source.size());
    if (hold_head_and_tail) {
        result.assign(source.begin() + 1, source.end() - 1);
        // copy(source.begin()+1, source.end()-1, result.begin());
        newpopsize = newpopsize >= 3 ? (newpopsize - 2) : newpopsize; // exclude the head and tail items
    } else {
        result.assign(source.begin(), source.end());
        //copy(source.begin(), source.end(), result.begin());
    }
    random_shuffle(result.begin(), result.end());
    result.resize(newpopsize);
    return result;
}


/***
 * comparator
 */

template<typename KeyType>
struct CompareForDataPair {
    static KeyType as_key(std::pair<KeyType, size_t> data_pair) {
        return data_pair.first;  // (key, index_pos) pair
    }

    static KeyType as_key(KeyType key) {
        return key;
    }

    template<typename T1, typename T2>
    inline bool operator()(T1 const &t1, T2 const &t2) const {
        return as_key(t1) < as_key(t2);
    }
};

/*
 * Maintain a TOTAL ORDER for the gapped array
 * e.g., x = [(10,[0]),   (13, []),   (13, []),   (13, [0,1,2])]
 *   x[0] and x[3] are real data, x[1] and x[2] are gaps.
 */
template<typename KeyType>
struct LessComparatorForGappedArray {
    static inline KeyType as_key(std::pair<KeyType, std::vector<KeyType>> k_in_gapped_array) {
        return k_in_gapped_array.first;
    }

    static inline KeyType as_key(KeyType key) {
        return key;
    }

    template<typename T1, typename T2>
    inline bool operator()(T1 const &t1, T2 const &t2) const {
        // compared by their keys, we will not get_payload_given_key two gaps
        // comparsion relationship: 3 types;   gap combination: 2*2-1=3 types,  thus 3*3 = 9 cases.
        // (1)    t1=13 (1), t2=13 (0): true
        // (2)    t1=11 (1), t2=13 (0): true
        // (3)    t1=14 (1), t2=13 (0): false
        // (4)    t1=13 (0), t2=13 (1): true
        // (5)    t1=13 (0), t2=11 (1): false
        // (6)    t1=13 (0), t2=14 (1): true
        // (7)    t1=13 (1), t2=14 (1): true
        // (8)    t1=14 (1), t2=13 (1): false
        // (9)    t1=14 (1), t2=14 (1): false
        // the exp_search_function is called by std::upper_bound, where t1 must be the real key
        // thus we need only consider case 2, 3, 7, 8, 9
        // assert(t1.second.size() != 0);
        // since t1 must be not a gap, we can simplify the comparision as key_t1 < key_t2
        return as_key(t1) < as_key(t2);

    }
};

/*
 * Maintain a TOTAL ORDER for the gapped array
 * e.g., x = [(10,[0]),   (13, []),   (13, []),   (13, [0,1,2])]
 *   x[0] and x[3] are real data, x[1] and x[2] are gaps.
 */
template<typename KeyType>
struct GreaterComparatorForGappedArray {
    static inline KeyType as_key(std::pair<KeyType, std::vector<KeyType>> k_in_gapped_array) {
        return k_in_gapped_array.first;
    }

    static inline KeyType as_key(KeyType key) {
        return key;
    }

    template<typename T1, typename T2>
    inline bool operator()(T1 const &t1, T2 const &t2) const {
        // compared by their keys, we will not get_payload_given_key two gaps
        // comparsion relationship: 3 types;   gap combination: 2*2-1=3 types,  thus 3*3 = 9 cases.
        // (1)    t1=13 (1), t2=13 (0): true       if 4
        // (2)    t1=11 (1), t2=13 (0): false      if 2
        // (3)    t1=14 (1), t2=13 (0): true       if 1
        // (4)    t1=13 (0), t2=13 (1): false      if 3
        // (5)    t1=13 (0), t2=11 (1): true       if 1
        // (6)    t1=13 (0), t2=14 (1): false      if 2
        // (7)    t1=13 (1), t2=14 (1): false      if 2
        // (8)    t1=14 (1), t2=13 (1): true       if 1
        // (9)    t1=14 (1), t2=14 (1): false      if 5
        // bool t1_is_gap = (t1.second.size() == 0), t2_is_gap = (t2.second.size() == 0);
        // the exp_search_function is called by std::lower_bound, where t2 must be the real key
        //auto key_t1 = as_key(t1), key_t2 = as_key(t2);
        // assert(t2.second.size() != 0);
        // since t2 must be not a gap, we can simplify the comparision as key_t1 > key_t2
        return as_key(t1) > as_key(t2);
        // if ( key_t1 > key_t2) {
        //     return true;
        // } else if (key_t1 < key_t2) {
        //     return false;
        // } else if (t1_is_gap){ // t1=13 (0),  t2=13(1)
        //     return false;
        // } else if (t2_is_gap) { // t2=13 (0),  t1=13(1)
        //     return true;
        // } else {
        //     return false;
        // }
        /*
         * more readable implementation but with lower efficiency
         *
        // key_t1 != key_t2
        bool key_equal = isEqual(key_t1, key_t2); //
        if (not key_equal){
            return key_t1 > key_t2;
        } else if (t1_is_gap){ // t1=13 (0),  t2=13(1)
            return false;
        } else if (t2_is_gap){ // t2=13 (0),  t1=13(1)
            return true;
        } else if (key_equal){ // found the exact position. t0=13(1), t1=13(1)
            return false;
        } else {
            throw "Incomplete handling for gapped array comparator!";
        }
         */

    }
};


/***
 * search strategies
 */


template<class ForwardIt, class T_val, typename T_pos, typename _Compare>
inline static ForwardIt exponential_search(ForwardIt begin, ForwardIt end, const T_val &value,
                                           const T_pos &pos, _Compare __comp) {
    typename std::iterator_traits<ForwardIt>::difference_type count;
    count = std::distance(begin, end);
    if (count == 0) {
        return end;
    }

    long bound = 1, lower, upper;
    T_val lower_key, upper_key;
    while (1) {
        lower = std::max(long(pos) - bound, long(0));
        lower_key = (*(begin + lower)).first;
        if (lower_key > value) {
            bound = bound << 1;
        } else {
            break;
        }
    }
    bound = 1;
    while (1) {
        upper = std::min(long(pos) + bound, count - 1);
        upper_key = (*(begin + upper)).first;
        if (upper_key < value) {
            bound = bound << 1;
        } else {
            break;
        }
    }

    auto lower_bound = begin + lower;
    auto upper_bound = begin + upper;
    auto res = std::lower_bound(lower_bound, upper_bound, value, __comp);

    return res;
}


template<class ForwardIt, class T_val, typename T_pos, typename _Compare>
inline static ForwardIt exponential_search(ForwardIt begin, ForwardIt end, const T_val &value,
                                           const T_pos &pos, _Compare __comp,
                                           size_t &upper_iter_count, size_t &lower_iter_count,
                                           long &binary_search_length) {
    typename std::iterator_traits<ForwardIt>::difference_type count;
    count = std::distance(begin, end);
    if (count == 0) {
        return end;
    }

    long bound = 1, lower, upper;
    T_val lower_key, upper_key;
    while (1) {
        lower = std::max(long(pos) - bound, long(0));
        lower_key = (*(begin + lower)).first;
        if (lower_key > value) {
            bound = bound << 1;
            lower_iter_count++;
        } else {
            break;
        }
    }
    bound = 1;
    while (1) {
        upper = std::min(long(pos) + bound, count - 1);
        upper_key = (*(begin + upper)).first;
        if (upper_key < value) {
            bound = bound << 1;
            upper_iter_count++;
        } else {
            break;
        }
    }

    binary_search_length += (upper - lower);

    auto lower_bound = begin + lower;
    auto upper_bound = begin + upper;
    auto res = std::lower_bound(lower_bound, upper_bound, value, __comp);

    return res;
}

template<class ForwardIt, class T, class T_pos, typename _Compare>
inline static const ForwardIt upper_bound_exp_search_total_order(
        const ForwardIt &begin, const ForwardIt &end,
        const T &key, const T_pos pos, _Compare __comp,
        size_t &upper_iter_count, size_t &lower_iter_count) {
    typename std::iterator_traits<ForwardIt>::difference_type count;
    count = std::distance(begin, end);
    if (count == 0) {
        //return std::reverse_iterator<ForwardIt>end);
        return end;
    }

    long bound = 1, lower, upper;
    T lower_key, upper_key;
    while (1) {
        lower = long(pos) - bound;
        if (lower <= long(0)) {
            lower = long(0);
            break;
        }
        lower_key = (*(begin + lower));
        if (lower_key > key) {
            bound = bound << 1;
            lower_iter_count++;
        } else {
            break;
        }
    }
    bound = 1;
    while (1) {
        upper = long(pos) + bound;
        if (upper >= count - 1) {
            upper = count - 1;
            break;
        }
        upper_key = (*(begin + upper));
        if (upper_key < key or upper == (count - 1)) {
            bound = bound << 1;
            upper_iter_count++;
        } else {
            break;
        }
    }

    auto lower_bound = begin + lower;
    auto upper_bound =
            begin + upper + 1; // make the semantic consistent to standard iterator: the last real element + 1.
    //std::reverse_iterator<decltype(lower_bound)> r_lower_bound(upper_bound), r_upper_bound(lower_bound);
    //return std::lower_bound(r_lower_bound, r_upper_bound, key, __comp);
    return std::upper_bound(lower_bound, upper_bound, key, __comp);
}


template<class ForwardIt, class T, class T_pos, typename _Compare>
inline static const std::reverse_iterator<ForwardIt> &exponential_search_total_order(
        const ForwardIt &begin, const ForwardIt &end,
        const T &key_pair, const T_pos pos, _Compare __comp) {
    typename std::iterator_traits<ForwardIt>::difference_type count;
    count = std::distance(begin, end);
    if (count == 0) {
        return std::reverse_iterator<ForwardIt>(end);
    }

    long bound = 1, lower, upper;
    auto key = key_pair.first;
    decltype(key) lower_key, upper_key;
    while (1) {
        lower = std::max(long(pos) - bound, long(0));
        lower_key = (*(begin + lower)).first;
        if (lower_key > key_pair.first) {
            bound = bound << 1;
        } else {
            break;
        }
    }
    bound = 1;
    while (1) {
        upper = std::min(long(pos) + bound, count - 1);
        upper_key = (*(begin + upper)).first;
        if (upper_key < key_pair.first) {
            bound = bound << 1;
        } else {
            break;
        }
    }

    //while ((*(begin + std::min(long(pos) + bound, count - 1))).first < value or
    //       (*(begin + std::max(long(pos) - bound, long(0)))).first > value){
    //    bound << 1;
    //}
    auto lower_bound = begin + lower;
    auto upper_bound =
            begin + upper + 1; // make the semantic consistent to standard iterator: the last real element + 1.
    std::reverse_iterator<decltype(lower_bound)> r_lower_bound(upper_bound), r_upper_bound(lower_bound);
    // we return the last item <= value, since the gapped array store the head of each linking array
    // e.g, in [(1,[0]), (5,[6,7]), (9, [Null]), (9,[0])], get_payload_given_key '7',  then we return the position of '5'
    //auto res = std::lower_bound(lower_bound, upper_bound, value, __comp);
    // auto res = std::lower_bound(r_lower_bound, r_upper_bound, value, __comp);

    return std::lower_bound(r_lower_bound, r_upper_bound, key_pair, __comp);
}

template<class ForwardIt, class T, class T_pos, typename _Compare>
inline static void exponential_search_total_order(std::reverse_iterator<ForwardIt> &res_iter,
                                                  const ForwardIt &begin, const ForwardIt &end, const T &key_pair,
                                                  const T_pos pos, _Compare __comp) {
    typename std::iterator_traits<ForwardIt>::difference_type count;
    count = std::distance(begin, end);
    if (count == 0) {
        res_iter = std::reverse_iterator<ForwardIt>(end);
    }

    long bound = 1, lower, upper;
    auto key = key_pair.first;
    decltype(key) lower_key, upper_key;
    while (1) {
        lower = std::max(long(pos) - bound, long(0));
        lower_key = (*(begin + lower)).first;
        if (lower_key > key_pair.first) {
            bound = bound << 1;
        } else {
            break;
        }
    }
    bound = 1;
    while (1) {
        upper = std::min(long(pos) + bound, count - 1);
        upper_key = (*(begin + upper)).first;
        if (upper_key < key_pair.first) {
            bound = bound << 1;
        } else {
            break;
        }
    }


    auto lower_bound = begin + lower;
    auto upper_bound =
            begin + upper + 1; // make the semantic consistent to standard iterator: the last real element + 1.
    std::reverse_iterator<decltype(lower_bound)> r_lower_bound(upper_bound), r_upper_bound(lower_bound);

    res_iter = std::lower_bound(r_lower_bound, r_upper_bound, key_pair, __comp);
}

template<class ForwardIt, class T, class T_pos, typename _Compare>
inline static void exponential_search_total_order(std::reverse_iterator<ForwardIt> &res_iter,
                                                  const ForwardIt &begin, const ForwardIt &end, const T &key_pair,
                                                  const T_pos pos, _Compare __comp,
                                                  size_t &upper_iter_count, size_t &lower_iter_count,
                                                  long &binary_search_length) {
    typename std::iterator_traits<ForwardIt>::difference_type count;
    count = std::distance(begin, end);
    if (count == 0) {
        res_iter = std::reverse_iterator<ForwardIt>(end);
    }

    long bound = 1, lower, upper;
    auto key = key_pair.first;
    decltype(key) lower_key, upper_key;
    while (1) {
        lower = std::max(long(pos) - bound, long(0));
        lower_key = (*(begin + lower)).first;
        if (lower_key > key_pair.first) {
            bound = bound << 1;
            lower_iter_count++;
        } else {
            break;
        }
    }
    bound = 1;
    while (1) {
        upper = std::min(long(pos) + bound, count - 1);
        upper_key = (*(begin + upper)).first;
        if (upper_key < key_pair.first) {
            bound = bound << 1;
            upper_iter_count++;
        } else {
            break;
        }
    }

    binary_search_length += (upper - lower);

    auto lower_bound = begin + lower;
    auto upper_bound =
            begin + upper + 1; // make the semantic consistent to standard iterator: the last real element + 1.
    std::reverse_iterator<decltype(lower_bound)> r_lower_bound(upper_bound), r_upper_bound(lower_bound);

    res_iter = std::lower_bound(r_lower_bound, r_upper_bound, key_pair, __comp);
}

template<class ForwardIt, class T1, class T2>
inline static ForwardIt exponential_search_check_gap(ForwardIt begin, ForwardIt end, const T1 &value, const T2 &pos) {
    typename std::iterator_traits<ForwardIt>::difference_type count;
    count = std::distance(begin, end);
    if (count == 0) {
        return end;
    }

    long bound = 1, lower, upper;
    long lower_gaps_count = 0, upper_gaps_count = 0; //accumulated gaps around the 'pos'
    T1 lower_key, upper_key;
    //exponentially find non-blank lower bound
    while (1) {
        lower = std::max(long(pos) - bound - lower_gaps_count, long(0));
        lower_key = (*(begin + lower)).first;
        while ((lower != 0) and (lower_key == 0)) {
            lower_gaps_count++;
            lower = std::max(long(pos) - bound - lower_gaps_count, long(0));
            lower_key = (*(begin + lower)).first;
        }
        if (lower_key <= value or (lower == 0)) {
            bound = 1;
            break;
        } else {
            bound = bound << 1;
        }
    }
    //exponentially find non-blank upper bound
    while (1) {
        upper = std::min(long(pos) + bound + upper_gaps_count, count - 1);
        upper_key = (*(begin + upper)).first;
        while ((upper != (count - 1)) and upper_key == 0) {
            upper_gaps_count++;
            upper = std::min(long(pos) + bound + upper_gaps_count, count - 1);
            upper_key = (*(begin + upper)).first;
        }
        if (upper_key >= value or (upper == (count - 1))) {
            break;
        } else {
            bound = bound << 1;
        }
    }

    // early return if the bound element is our target
    if (isEqual(lower_key, value)) {
        return (begin + lower);
    }
    if (isEqual(upper_key, value) or (isEqual(upper, (count - 1)))) {
        // if upper == (count - 1), it means that the get_payload_given_key value is larger than all keys
        return (begin + upper);
    }

    //gapped binary search from [lower, upper]
    long lower_of_mid, upper_of_mid, mid, shift_of_mid;
    while (lower < upper) {
        mid = (lower + upper) / 2;
        shift_of_mid = 1;
        if (isEqual((*(begin + mid)).first, 0)) {
            while ((mid - shift_of_mid) >= lower or (mid + shift_of_mid) <= upper) {
                lower_of_mid = std::max(mid - shift_of_mid, lower);
                if ((*(begin + lower_of_mid)).first != 0) {
                    mid = lower_of_mid;
                    break;
                }
                upper_of_mid = std::min(mid + shift_of_mid, upper);
                if ((*(begin + upper_of_mid)).first != 0) {
                    mid = upper_of_mid;
                    break;
                }
                shift_of_mid++;
            }
        }
        if (value < (*(begin + mid)).first) {
            upper = mid;
        } else if (value > (*(begin + mid)).first) {
            lower = mid + 1;
        } else {
            return (begin + mid);
        }
    }
    return (begin + upper);
}




/***
 * experimental explorations, adaptive-sampling
 */

template<class key_type>
double estimate_sample_rate_time_slots(std::vector<std::pair<key_type, size_t>> data_with_pos, bool print = true) {
    double sample_rate_estimated, temp_rate, min_x_delta_mean = 1000000000, min_x_delta_std = 1000000000;
    // format of timestamp key: 20170325120000
    int year = int(data_with_pos[0].first / 10000000000);
    long base_year = year * 10000000000;
    bool is_leap_year = ((year % 4 == 0 && year % 100 != 0) || (year % 400 == 0));
    std::vector<int> accumulated_days_leap_year = {0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335, 366};
    std::vector<int> accumulated_days_nonleap_year = {0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365};
    std::vector<int> accumulated_days_of_one_year = is_leap_year ?
                                                    accumulated_days_leap_year : accumulated_days_nonleap_year;
    std::vector<std::vector<std::vector<key_type>>> x_in_each_hours;
    x_in_each_hours.reserve(366); // each year has at most 366 days
    for (int k = 0; k < 366; ++k) {
        std::vector<std::vector<key_type>> tmp_out;
        x_in_each_hours.emplace_back(tmp_out);
        for (int m = 0; m < 24; ++m) {
            std::vector<key_type> tmp;
            x_in_each_hours[k].emplace_back(tmp); // each day has at most 24 hours;
        }
    }
    for (int j = 0; j < data_with_pos.size(); ++j) {
        size_t key = data_with_pos[j].first - base_year;
        int month = floor(key / 100000000) - 1;
        int day = int(floor(key / 1000000)) % 100;
        int day_idx = accumulated_days_of_one_year[month] + day - 1;
        int hour = int(floor(key / 10000)) % 100;
        x_in_each_hours[day_idx][hour].emplace_back(data_with_pos[j].first);
    }

    std::vector<int> time_slot_ranges = {1, 2, 3, 4, 5};
    long search_counter(0);
    // auto search a suitable time slot, where the delta_x of all days obey the same normal distribution
    for (auto slot_range : time_slot_ranges) {
        for (int hour = 0; (hour + slot_range - 1) < 24; hour++) {
            std::vector<double> x_delta_mean_of_specific_time_slot, x_delta_std_of_specific_time_slot;
            std::vector<long> number_of_keys_of_a_day;
            x_delta_mean_of_specific_time_slot.reserve(366);
            x_delta_std_of_specific_time_slot.reserve(366);
            number_of_keys_of_a_day.reserve(366);
            int day_count = 0;
            std::set<double> existed_keys_in_each_day;
            for (int day = 0; day < 366; day++) {
                std::vector<double> x_in_a_specific_time_slot;
                std::vector<double> delta_in_a_specific_time_slot;
                if (x_in_each_hours[day].size() == 0) {
                    continue;
                }
                for (int i = 0; (hour + i < slot_range) and x_in_each_hours[day][hour + i].size() > 0; i++) {
                    for (auto x : x_in_each_hours[day][hour + i]) {
                        double hour_minute_seconds = double((x - base_year) % 1000000);
                        x_in_a_specific_time_slot.emplace_back(hour_minute_seconds);
                        existed_keys_in_each_day.insert(hour_minute_seconds);
                    }
                }
                if (x_in_a_specific_time_slot.size() > 0) {
                    for (int i = 0; i < x_in_a_specific_time_slot.size() - 1; i++) {
                        delta_in_a_specific_time_slot.emplace_back(
                                (abs(x_in_a_specific_time_slot[i + 1] - x_in_a_specific_time_slot[i])));
                    }
                    std::pair<double, double> x_delta_res = calculate_gaussian_mean_std(delta_in_a_specific_time_slot);
                    day_count++;
                    x_delta_mean_of_specific_time_slot.emplace_back(x_delta_res.first);
                    x_delta_std_of_specific_time_slot.emplace_back(x_delta_res.second);
                    number_of_keys_of_a_day.emplace_back(x_in_a_specific_time_slot.size());
                }
            }
            search_counter++;
            if (day_count == 0) {
                continue;
            }
            double avg_number_of_keys_of_a_day = std::accumulate(number_of_keys_of_a_day.begin(),
                                                                 number_of_keys_of_a_day.end(), 0.0) /
                                                 number_of_keys_of_a_day.size();

            assert(x_delta_mean_of_specific_time_slot.size() != 0);
            double std_of_x_delta_mean_all_days, std_of_x_delta_std_all_days;
            std_of_x_delta_mean_all_days = calculate_mean_std(x_delta_mean_of_specific_time_slot, false).second;
            std_of_x_delta_std_all_days = calculate_mean_std(x_delta_std_of_specific_time_slot, false).second;

            assert(avg_number_of_keys_of_a_day <= existed_keys_in_each_day.size());
            temp_rate = avg_number_of_keys_of_a_day / existed_keys_in_each_day.size();
            if (std_of_x_delta_mean_all_days < min_x_delta_mean and std_of_x_delta_std_all_days < min_x_delta_std) {
                min_x_delta_mean = std_of_x_delta_mean_all_days;
                min_x_delta_std = std_of_x_delta_std_all_days;
                sample_rate_estimated = temp_rate;
            }
            if (print) {
                std::cout << "search times: " << search_counter << std::endl;
                std::cout << "estimated sample rate and temp_rate are: " << sample_rate_estimated << ", " << temp_rate
                          << std::endl;
            }
            if (std_of_x_delta_mean_all_days <= 1.0 and std_of_x_delta_std_all_days < 1.0) {
                return sample_rate_estimated;
            }
        }
    }
    return sample_rate_estimated;

}

template<class key_type, typename segment_type>
double
estimate_sample_rate_auto(std::vector<std::pair<key_type, size_t>> data_with_pos, std::vector<segment_type> segments,
                          int top_n = 10) {
    top_n = (top_n < segments.size()) ? top_n : segments.size();
    std::nth_element(segments.begin(), (segments.begin() + top_n - 1), segments.end(),
                     [](auto &l, auto &r) { return l.seg_slope > r.seg_slope; });

    std::vector<long> number_of_keys_of_a_seg;
    std::set<size_t> keys_of_all_segments;
    // TODO only applicable for time-slot key, to be modified for other formats
    // format of timestamp key: 20170325120000
    int year = int(data_with_pos[0].first / 10000000000);
    long base_year = year * 10000000000;

    std::vector<std::pair<key_type, size_t>> data_with_pos_in_top_segs;


    for (int i = 0; i < top_n; i++) {
        std::set<size_t> keys_of_a_segment;
        size_t left = segments[i].seg_intercept;
        size_t seg_end = segments[i].seg_last_y;
        // TODO: check when intercept or last_Y > size
        if (left > data_with_pos.size() or seg_end > data_with_pos.size()) {
            continue;
        }
        while (left <= seg_end) {
            data_with_pos_in_top_segs.emplace_back(data_with_pos[left]);
            left++;
        }
    }
    assert (data_with_pos_in_top_segs.size() < data_with_pos.size());
    std::cout << "the size of data_with_pos_in_top_segs: " << data_with_pos_in_top_segs.size() << std::endl;
    double estimate_rate = estimate_sample_rate_time_slots(data_with_pos_in_top_segs, false);
    return estimate_rate;
}


struct KeyGroup {
    size_t begin_pos;
    size_t end_pos;
    double mean;
    double std_dev;
    size_t gap_quota;

    KeyGroup(size_t begin_pos, size_t end_pos, double mean, double std_dev) :
            begin_pos(begin_pos), end_pos(end_pos), mean(mean), std_dev(std_dev), gap_quota(0) {};
};

struct MetaGroup {
    std::vector<KeyGroup> groups;
    double groups_mean;
    double groups_std_dev;
    double allocate_factor;
    size_t number_of_keys, gap_quota;

    MetaGroup(KeyGroup first_group) {
        groups.emplace_back(first_group);
        groups_mean = first_group.mean;
        groups_std_dev = first_group.std_dev;
        number_of_keys = first_group.end_pos - first_group.begin_pos;
        gap_quota = 0;
        allocate_factor = 0.0;
    }

    inline double distance_from_groups(KeyGroup group) {
        //Manhattan distance for quick calculate
        double dist = abs((group.mean - groups_mean)) + abs((group.std_dev - groups_std_dev));
        return dist;
    }

    inline void add_group(KeyGroup group) {
        size_t N = groups.size();
        groups_mean = (groups_mean * N + group.mean) / (N + 1);
        groups_std_dev = (groups_std_dev * N + group.std_dev) / (N + 1);
        number_of_keys += group.end_pos - group.begin_pos;
        groups.emplace_back(group);
    }
};

struct less_than_group_by_mean {
    inline bool operator()(const KeyGroup &group1, const KeyGroup &group2) {
        return group1.mean < group2.mean;
    }
};

struct less_than_group_by_key {
    inline bool operator()(const KeyGroup &group1, const KeyGroup &group2) {
        return group1.begin_pos < group2.begin_pos;
    }
};

struct less_than_meta_group {
    inline bool operator()(const MetaGroup &group1, const MetaGroup &group2) {
        return group1.number_of_keys < group2.number_of_keys;
    }
};


void slice_data_by_T(const std::vector<std::pair<key_type, size_t>> &data_with_pos, key_type T, size_t data_size,
                     std::vector<KeyGroup> &groups);

void cluster_groups(double epsilon, std::vector<KeyGroup> &groups, std::vector<MetaGroup> &meta_groups);

std::vector<key_type, size_t> &
insert_gaps(const std::vector<std::pair<key_type, size_t>> &data_with_pos, const KeyGroup &group);

template<class key_type>
std::vector<std::pair<key_type, size_t>> estimate_and_allocate(
        std::vector<std::pair<key_type, size_t>> data_with_pos, key_type T, double epsilon) {
    size_t data_size = data_with_pos.size();
    if (data_size <= 2) { return data_with_pos; }

    // 1: estimate density by bottom_top grouping
    // 1.1: x -> g_i,  slicing by T.
    std::vector<KeyGroup> groups;
    slice_data_by_T(data_with_pos, T, data_size, groups);
    // 1.2: g_i -> G_i, clustering by density similarity
    std::vector<MetaGroup> meta_groups;
    cluster_groups(epsilon, groups, meta_groups);
    // 1.3: G_i -> "estimated density", choose the meta-group with the largest number of consistent groups
    size_t most_consistent_idx = 0;
    size_t largest_number_of_groups = 0.0;
    for (int i = 0; i < meta_groups.size(); i++) {
        // double avg_number_of_keys = double(meta_groups[i].number_of_keys) / meta_groups[i].groups.size();
        if (meta_groups[i].groups.size() > largest_number_of_groups) {
            most_consistent_idx = i;
            largest_number_of_groups = meta_groups[i].groups.size();
        }
    }
    double density = estimate_density_from_meta_group(meta_groups[most_consistent_idx], data_with_pos, T);
    std::cout << "the estimated density is: " << density << std::endl;

    // 2. insert gaps by top_bottom multi-level allocation
    size_t total_gap_quota = round(data_size / density - data_size);
    // 2.1 Q_all -> G_i: globally: inspired by Neyman allocation
    //          Q_i = Q_all * (W_i * C_i) / sum(W_i*C_i), W_i = N_i / N_all,
    //          C_i is adjust factor, C_i = Mean_i / Std_dev_i, mean_i indicates density, std_dev_i indicates reliablity
    double normorlization_factor = 0.0;
    for (auto meta_group : meta_groups) {
        double W_i = double(meta_group.number_of_keys) / data_size;
        double C_i = meta_group.groups_mean / meta_group.groups_std_dev;
        meta_group.allocate_factor = W_i * C_i;
        normorlization_factor += meta_group.allocate_factor;
    }
    for (auto meta_group : meta_groups) {
        meta_group.gap_quota = round(meta_group.allocate_factor / normorlization_factor * total_gap_quota);
    }
    // 2.2 G_i -> g_i: locally: q_i = Q_i * w_i, w_i = n_i / N_i
    for (auto meta_group : meta_groups) {
        for (auto group : meta_group.groups) {
            double w_i = double(meta_group.groups_mean / group.mean);
            group.gap_quota = round(w_i * group.gap_quota);
        }
    }
    // 2.3 g_i -> gap:  generate gaps from normal(alpha, alpha), alpha = g_i / n;
    std::sort(groups.begin(), groups.end(), less_than_group_by_key());
    for (auto group : groups) {
        insert_gaps(data_with_pos, group);
    }

    return data_with_pos;

    // // online one pass clustering
    // meta_groups.emplace_back(MetaGroup(groups[0]));
    // double min_dist = std::numeric_limits<double>::max();
    // size_t closest_meta_group_idx = 0;
    // for (int i = 1; i < groups.size(); i ++) {
    //     for (int j = 0; j < meta_groups.size(); j++){
    //         double dist = meta_groups[j].distance_from_groups(groups[i]);
    //         if (dist < min_dist){
    //             min_dist = dist;
    //             closest_meta_group_idx = j;
    //         }
    //     }
    //     if (min_dist < epsilon){
    //         meta_groups[closest_meta_group_idx].
    //     }

    // }

}

void insert_gaps(std::vector<std::pair<key_type, size_t>> &data_with_pos, const KeyGroup &group) {
    size_t delta_y_quota = group.gap_quota;
    size_t gap_number_of_the_group = group.end_pos - group.begin_pos + 1;
    std::normal_distribution<double> distribution(group.mean, group.std_dev);
    std::default_random_engine generator;
    std::vector<size_t> gaps_in_the_group;
    gaps_in_the_group.reserve(gap_number_of_the_group);
    for (size_t l = 0; l < gap_number_of_the_group; l++) {
        gaps_in_the_group.emplace_back(1);
    }
    // allocate the generated gaps
    for (size_t l = 0; l < gap_number_of_the_group and delta_y_quota > 0; l++) {
        size_t gap = round(distribution(generator));
        gap = gap > delta_y_quota ? delta_y_quota : gap;
        delta_y_quota -= gap;
        gaps_in_the_group[l] = gap + 1;
    }
    // allocate the remain quota
    size_t baseline = round(group.mean);
    while (delta_y_quota > 0) {
        for (size_t l = 0; l < gap_number_of_the_group; l++) {
            if (delta_y_quota == 0) { break; }
            if (gaps_in_the_group[l] <= baseline) {
                gaps_in_the_group[l]++;
                delta_y_quota--;
            }
        }
        baseline++;
    }
    // allocate the real gaps
    for (size_t i = group.begin_pos, j = 0; i <= group.end_pos; i++, j++) {
        size_t gap = gaps_in_the_group[j];
        data_with_pos[i + 1].second = data_with_pos[i].second + gap;
    }
}

void cluster_groups(double epsilon, std::vector<KeyGroup> &groups, std::vector<MetaGroup> &meta_groups) {
    std::sort(groups.begin(), groups.end(), less_than_group_by_mean());
    MetaGroup meta_group(groups[0]);
    size_t meta_idx = 0;
    meta_groups.emplace_back(meta_group);
    for (size_t i = 1; i < groups.size(); i++) {
        double dist = meta_groups[meta_idx].distance_from_groups(groups[i]);
        if (dist < epsilon) {
            meta_groups[meta_idx].add_group(groups[i]);
        } else {
            MetaGroup meta_group(groups[i]);
            meta_groups.emplace_back(meta_group);
            meta_idx++;
        }
    }
}

void slice_data_by_T(const std::vector<std::pair<key_type, size_t>> &data_with_pos, key_type T, size_t data_size,
                     std::vector<KeyGroup> &groups) {
    key_type group_key_end = data_with_pos[0].first + T;
    size_t group_begin_idx(0), group_end_idx(0);
    double delta_sum(0.0), delta_square_sum(0.0), mean(0.0), std_dev(0.0);
    for (size_t i = 1; i < data_size; i++) {
        double delta_x = data_with_pos[i].first - data_with_pos[i - 1].first;
        if (data_with_pos[i].first <= group_key_end) {
            delta_sum += delta_x;
            delta_square_sum += delta_x * delta_x;
        } else {
            group_end_idx = i - 1;
            size_t delta_count = group_end_idx - group_begin_idx;
            mean = delta_sum / delta_count;
            // one pass std_dev calculation: s_x^2 = (sum(x_i^2)/N - mean_x^2)
            std_dev = sqrt(delta_square_sum / delta_count - mean * mean);
            if (group_end_idx != (group_begin_idx + 1)) {
                // skip the slots that have only one get_payload_given_key
                groups.emplace_back(KeyGroup(group_begin_idx, group_end_idx, mean, std_dev));
            }
            group_begin_idx = group_end_idx;
            group_key_end += T;
            while (data_with_pos[group_begin_idx].first > group_key_end) {
                // skip blank slot since we slicing by fixed T
                group_key_end += T;
            }
            delta_sum = 0.0;
            delta_square_sum = 0.0;
        }
    }
}

template<class key_type>
double estimate_density_from_meta_group(MetaGroup meta_group,
                                        std::vector<std::pair<key_type, size_t>> data_with_pos, key_type T) {
    std::unordered_set<key_type> distinct_relative_keys;
    std::vector<size_t> number_of_keys_of_groups;
    for (auto group: meta_group.groups) {
        // size_t base_x = data_with_pos[group.begin_pos].first;
        size_t number_of_keys_in_group = group.end_pos - group.begin_pos + 1;
        for (int i = group.begin_pos; i <= group.end_pos; i++) {
            // distinct_relative_keys.insert(data_with_pos[i].first - base_x);
            distinct_relative_keys.insert(data_with_pos[i].first % T);
        }
        number_of_keys_of_groups.emplace_back(number_of_keys_in_group);
    }
    double avg_number_of_keys_of_groups = std::accumulate(number_of_keys_of_groups.begin(),
                                                          number_of_keys_of_groups.end(), 0.0) /
                                          number_of_keys_of_groups.size();
    double estimate_rate = avg_number_of_keys_of_groups / distinct_relative_keys.size();
    return estimate_rate;
}

template<typename T>
std::string type_name();

template<class T>
std::string
type_name() {
    typedef typename std::remove_reference<T>::type TR;
    std::unique_ptr<char, void (*)(void *)> own
            (
#ifndef _MSC_VER
            abi::__cxa_demangle(typeid(TR).name(), nullptr,
                                nullptr, nullptr),
#else
            nullptr,
#endif
            std::free
    );
    std::string r = own != nullptr ? own.get() : typeid(TR).name();
    if (std::is_const<TR>::value)
        r += " const";
    if (std::is_volatile<TR>::value)
        r += " volatile";
    if (std::is_lvalue_reference<T>::value)
        r += "&";
    else if (std::is_rvalue_reference<T>::value)
        r += "&&";
    return r;
}

/*
 * Insert gaps in a result_driven manner, allocating gaps to make the inserted data more linear
 */
template<typename segment_type>
std::vector<std::pair<key_type_transformed, size_t>> insert_gaps_result_driven(
        std::vector<std::pair<key_type_transformed, size_t>> data, double gap_rate, std::vector<segment_type> segments,
        bool cliping_by_delta_x = false) {
    std::vector<std::pair<key_type_transformed, size_t>> data_inserted;
    for (auto key : data) {
        data_inserted.emplace_back(key);
    }
    float_t seg_slope_ave(0.0);
    for (auto seg : segments) {
        seg_slope_ave += seg.seg_slope;
    }
    seg_slope_ave = seg_slope_ave / segments.size();
    int count = 0, gap_number_of_a_seg = 0, cur_seg_idx = 1;
    key_type_transformed delta_x_of_a_seg = 0;
    int seg_begin_y(0), seg_end_y(0);
    while (seg_end_y < (data.size() - 1) and seg_begin_y < (data.size() - 1)) {
        // find the keys belonging to a certain segment
        if (cur_seg_idx == segments.size() - 1) {
            seg_end_y = data.size() - 1;
            cur_seg_idx++; // to keep consistent with the 'segments[cur_seg_idx-1]'
        } else {
            while (data[seg_end_y].first < segments[cur_seg_idx].seg_start) {
                seg_end_y++;
            }
        }
        delta_x_of_a_seg = data[seg_end_y].first - data[seg_begin_y].first;
        gap_number_of_a_seg = data[seg_end_y].second - data[seg_begin_y].second - 1;
        assert(seg_end_y > seg_begin_y);
        size_t delta_y_quota = round(gap_rate * (gap_number_of_a_seg + 1) *
                                     (seg_slope_ave / segments[cur_seg_idx - 1].seg_slope));
        //size_t delta_y_quota = round(gap_rate * (gap_number_of_a_seg + 1));
        if (seg_slope_ave / segments[cur_seg_idx - 1].seg_slope > 1000) {
            double big_quota_occured = seg_slope_ave / segments[cur_seg_idx - 1].seg_slope;
            int i = 0 + big_quota_occured;
        }
        std::vector<size_t> gaps_in_a_seg;
        size_t number_of_keys = seg_end_y - seg_begin_y;
        for (int l = 0; l < number_of_keys - 1; l++) {
            gaps_in_a_seg.emplace_back(0);
        }
        size_t delta_y_remain = delta_y_quota;
        for (int l = 0; l < number_of_keys - 1 and delta_y_remain > 0; l++) {
            key_type_transformed delta_x_l = data[seg_begin_y + l + 1].first - data[seg_begin_y].first;
            size_t old_delta_y = data[seg_begin_y + l + 1].second - data[seg_begin_y].second; // when sampling case
            int gap = ceil(double(delta_x_l) / delta_x_of_a_seg * (gap_number_of_a_seg + delta_y_quota)) - old_delta_y;
            gap = gap < 0 ? 0 : gap;
            gap = gap > delta_y_remain ? delta_y_remain : gap;
            if (cliping_by_delta_x) {
                gap = gap > delta_x_l ? delta_x_l : gap;
            }
            if (gap > 500000) {
                int error_occured = 0;
            }
            delta_y_remain -= gap;
            gaps_in_a_seg[l + 1] = gap;
        }
        if (seg_begin_y != 0) {
            // int gap_between_seg = floor((gap_rate * (seg_end_y - seg_begin_y) - 1 ) / gap_number_of_a_seg);
            int gap_between_seg = 0;
            size_t old_delta_y = data[seg_begin_y].second - data[seg_begin_y - 1].second;
            data_inserted[seg_begin_y].second = data_inserted[seg_begin_y - 1].second + old_delta_y + gap_between_seg;
            if (data_inserted[seg_begin_y].second > 10000000) {
                int error_occured = 0;
            }
        }
        std::vector<size_t> old_delta_ys;  // used for sampled case, e.g.  x[1].second = 3, x[2].second = 11
        old_delta_ys.reserve(number_of_keys);
        for (int k = seg_begin_y; k < seg_end_y - 1; k++) {
            old_delta_ys.emplace_back(data[k + 1].second - data[k].second);
        }
        for (int k = seg_begin_y, l = 0; k < seg_end_y - 1; k++, l++) {
            size_t gap = gaps_in_a_seg[l];
            data_inserted[k + 1].second = data_inserted[k].second + gap + old_delta_ys[l];
            if (data_inserted[k + 1].second > 100000000) {
                int error_occured = 0;
            }
        }
        seg_begin_y = seg_end_y;
        cur_seg_idx++;
    }
    size_t old_delta_y = data[seg_begin_y].second - data[seg_begin_y - 1].second;
    data_inserted[seg_begin_y].second = data_inserted[seg_begin_y - 1].second + old_delta_y;
    //std::cout<<"Inserted end, the count of gap > delta is: "<< count <<std::endl;
    return data_inserted;
}


/*
 * Insert gaps in a result_driven manner, allocating gaps to make the inserted data more linear
 * The sequential placement strategy will return data including no conflicting positions, i.e., no y'_i == y'_j
 */
template<typename segment_type>
std::vector<std::pair<key_type_transformed, size_t>> insert_gaps_result_driven_sequential_place(
        std::vector<std::pair<key_type_transformed, size_t>> data, double gap_rate, std::vector<segment_type> segments,
        bool cliping_by_delta_x = false, std::string save_dir = "") {
    std::vector<std::pair<key_type_transformed, size_t>> data_inserted;
    for (auto key : data) {
        data_inserted.emplace_back(key);
    }
    float_t seg_slope_ave(0.0);
    std::vector<size_t> conflicts_of_sequential_placement, conflicts_of_linking_array;
    for (auto seg : segments) {
        seg_slope_ave += seg.seg_slope;
        conflicts_of_sequential_placement.emplace_back(0);
        conflicts_of_linking_array.emplace_back(0);
    }
    seg_slope_ave = seg_slope_ave / segments.size();

    key_type_transformed delta_x_of_a_seg = 0;
    size_t seg_begin_i(0), seg_end_i(0), cur_seg_idx(0);


    // each loop indicates a segment
    while (seg_begin_i < (data.size() - 1)) {
        std::vector<size_t> round_ys_of_a_seg; // used for do statistics of the conflicts_of_linking_array in a seg
        round_ys_of_a_seg.reserve(data.size());

        // find the keys belonging to a certain segment
        while (data[seg_begin_i].first < segments[cur_seg_idx].seg_start) {
            seg_begin_i++;
        }
        if (cur_seg_idx == segments.size() - 1) {
            seg_end_i = data.size() - 1;
        } else {
            while (data[seg_end_i].first < segments[cur_seg_idx + 1].seg_start) {
                seg_end_i++;
            }
        }
        assert(seg_end_i > seg_begin_i);
        // for the complement segments including only 2 nodes, we do not insert un-necessary gaps
        if ((seg_end_i - seg_begin_i) == 1) {
            data_inserted[seg_begin_i].second = data_inserted[seg_begin_i - 1].second + 1;
            data_inserted[seg_end_i].second = data_inserted[seg_begin_i - 1].second + 2;
            seg_begin_i = seg_end_i;
            cur_seg_idx++;
            continue;
        }

#ifdef Debug
        // (debug) checking big quota
        if (seg_slope_ave / segments[cur_seg_idx].seg_slope > 1000){
            double big_quota_occured = seg_slope_ave / segments[cur_seg_idx].seg_slope;
            int i = 0 + big_quota_occured;
            i++;
        }
#endif

        // the deltas used in the two similar triangles
        delta_x_of_a_seg = data[seg_end_i].first - data[seg_begin_i].first;
        size_t delta_y_of_a_seg = data[seg_end_i].second - data[seg_begin_i].second;
        //size_t gap_quota = round(gap_rate * delta_y_of_a_seg *
        //                         (seg_slope_ave / segments[cur_seg_idx].seg_slope));
        size_t gap_quota = round(gap_rate * delta_y_of_a_seg);
        size_t gapped_delta_y_of_a_seg = delta_y_of_a_seg + gap_quota;

        // the first node of a segment
        size_t base_pos_of_seg = data[seg_begin_i].second; // the position of the first point of the segment
        if (cur_seg_idx > 0) {
            base_pos_of_seg += data_inserted[seg_begin_i - 1].second - data[seg_begin_i - 1].second;
        }
        data_inserted[seg_begin_i].second = base_pos_of_seg;
        round_ys_of_a_seg.emplace_back(base_pos_of_seg);

        // the middle nodes of a segment
        for (int l = seg_begin_i + 1; l < seg_end_i; l++) {
            key_type_transformed delta_x_l = data[l].first - data[seg_begin_i].first;
            size_t y_gapped = round(double(delta_x_l) / delta_x_of_a_seg * gapped_delta_y_of_a_seg);
            y_gapped = y_gapped + base_pos_of_seg;
            round_ys_of_a_seg.emplace_back(y_gapped);
            // since the round operation makes the y_gapped_i may == y_(i-1), we need to guarantee the order
            // i.e., the so-called sequential placement
            auto last_y_gapped = data_inserted[l - 1].second;
            if (y_gapped <= last_y_gapped) {
                y_gapped = last_y_gapped + 1;
                conflicts_of_sequential_placement[cur_seg_idx]++;
            }
            data_inserted[l].second = y_gapped;
        }

        // the last node of a segment
        size_t last_y_of_the_seg = data[seg_end_i].second + gap_quota + base_pos_of_seg;
        round_ys_of_a_seg.emplace_back(last_y_of_the_seg);
        data_inserted[seg_end_i].second = last_y_of_the_seg;
        if (last_y_of_the_seg <= data_inserted[seg_end_i - 1].second) {
            data_inserted[seg_end_i].second = data_inserted[seg_end_i - 1].second + 1;
            conflicts_of_sequential_placement[cur_seg_idx]++;
        }
        seg_begin_i = seg_end_i;
        cur_seg_idx++;

#ifdef Debug
        // statistic for the conflicts_of_linking_array
        std::map<size_t, size_t > counter_of_round_ys;
        for (auto ys : round_ys_of_a_seg) {
            counter_of_round_ys[ys]++;
        }
        for (auto ys : counter_of_round_ys) {
            if (ys.second > 1){
                conflicts_of_linking_array[cur_seg_idx] += (ys.second - 1);
            }
        }
#endif
    }
    // // compress the gaps for last point of the whole dataset
    // data_inserted[data_inserted.size() - 1].second = data_inserted[data_inserted.size() - 2].second + 1;
    auto ori_last_pos = data[data_inserted.size() - 1].second;
    auto pre_gapped_pos = data_inserted[data_inserted.size() - 2].second;
    data_inserted[data_inserted.size() - 1].second =
            ori_last_pos > (pre_gapped_pos + 1) ? ori_last_pos : (pre_gapped_pos + 1);

#ifdef Debug
    //write_vector_to_f(conflicts_of_sequential_placement, "conflicts_of_sequential_placement");
    //write_vector_to_f(conflicts_of_linking_array, "conflicts_of_linking_array");
    std::cout << "In gap insertion stage, statistics for conflicts of sequential-placement and linking-array:" << std::endl;
    calculate_mean_std(conflicts_of_sequential_placement);
    calculate_mean_std(conflicts_of_linking_array);
#endif

    return data_inserted;
}


/*
 * Insert gaps in a result_driven manner, allocating gaps to make the inserted data more linear
 * the linking array strategy will return data that including duplicated postions
 */
template<typename segment_type>
std::vector<std::pair<key_type_transformed, size_t>> insert_gaps_result_driven_linking_array(
        std::vector<std::pair<key_type_transformed, size_t>> data, double gap_rate, std::vector<segment_type> segments,
        bool cliping_by_delta_x = false, std::string save_dir = "") {
    std::vector<std::pair<key_type_transformed, size_t>> data_inserted;
    for (auto key : data) {
        data_inserted.emplace_back(key);
    }
    float_t seg_slope_ave(0.0);
    for (auto seg : segments) {
        seg_slope_ave += seg.seg_slope;
    }
    seg_slope_ave = seg_slope_ave / segments.size();

    key_type_transformed delta_x_of_a_seg = 0;
    int seg_begin_i(0), seg_end_i(0), cur_seg_idx(0);

    std::vector<size_t> round_ys; // used for do statistics of the conflicts_of_linking_array strategy

    // each loop indicates a segment
    while (seg_begin_i < (data.size() - 1)) {

        // find the keys belonging to a certain segment
        while (data[seg_begin_i].first < segments[cur_seg_idx].seg_start) {
            seg_begin_i++;
        }
        if (cur_seg_idx == segments.size() - 1) {
            seg_end_i = data.size() - 1;
        } else {
            while (data[seg_end_i].first < segments[cur_seg_idx + 1].seg_start) {
                seg_end_i++;
            }
        }
        assert(seg_end_i > seg_begin_i);
        // for the complement segments including only 2 nodes, we do not insert un-necessary gaps
        if ((seg_end_i - seg_begin_i) == 1) {
            data_inserted[seg_begin_i].second = data_inserted[seg_begin_i - 1].second + 1;
            data_inserted[seg_end_i].second = data_inserted[seg_begin_i - 1].second + 2;
            seg_begin_i = seg_end_i;
            cur_seg_idx++;
            continue;
        }
#ifdef Debug
        // (debug) checking big quota
        if (seg_slope_ave / segments[cur_seg_idx].seg_slope > 1000){
            double big_quota_occured = seg_slope_ave / segments[cur_seg_idx].seg_slope;
            int i = 0 + big_quota_occured;
        }
#endif
        // the deltas used in the two similar triangles
        delta_x_of_a_seg = data[seg_end_i].first - data[seg_begin_i].first;
        size_t delta_y_of_a_seg = data[seg_end_i].second - data[seg_begin_i].second;
        //size_t gap_quota = round(gap_rate * delta_y_of_a_seg *
        //                         (seg_slope_ave / segments[cur_seg_idx].seg_slope));
        size_t gap_quota = round(gap_rate * delta_y_of_a_seg);
        size_t gapped_delta_y_of_a_seg = delta_y_of_a_seg + gap_quota;

        // the first node of a segment
        size_t base_pos_of_seg = data[seg_begin_i].second; // the position of the first point of the segment
        if (cur_seg_idx > 0) {
            base_pos_of_seg += data_inserted[seg_begin_i - 1].second - data[seg_begin_i - 1].second;
        }
        data_inserted[seg_begin_i].second = base_pos_of_seg;
        round_ys.emplace_back(base_pos_of_seg);

        // the middle nodes of a segment
        for (int l = seg_begin_i + 1; l < seg_end_i; l++) {
            key_type_transformed delta_x_l = data[l].first - data[seg_begin_i].first;
            size_t y_gapped = round(double(delta_x_l) / delta_x_of_a_seg * gapped_delta_y_of_a_seg);
            y_gapped = y_gapped + base_pos_of_seg;
            round_ys.emplace_back(y_gapped);
            data_inserted[l].second = y_gapped;
        }

        // the last node of a segment
        size_t last_y_of_the_seg = data[seg_end_i].second + gap_quota + base_pos_of_seg;
        round_ys.emplace_back(last_y_of_the_seg);
        data_inserted[seg_end_i].second = last_y_of_the_seg;
        seg_begin_i = seg_end_i;
        cur_seg_idx++;
    }
    // compress the gaps for last point of the whole dataset
    data_inserted[data_inserted.size() - 1].second = data_inserted[data_inserted.size() - 2].second + 1;

    //return std::make_pair(data_inserted, counter_of_round_ys);
    return data_inserted;
}

/*
 *  according to the gap-inserted data, update the positions of original whole data,
 *  the sequential strategy place the un-sampled key in the position nearest to its left sampled-key
 */
template<typename key_type_transformed, typename index_type>
void update_pos_by_gaps_sequential(std::vector<std::pair<key_type_transformed, size_t>> &original_data,
                                   std::vector<std::pair<key_type_transformed, size_t>> const &gapped_sampled_data,
                                   index_type &learned_index = NULL) {
    size_t idx_of_gapped_data = 0;
    size_t original_size = original_data.size();
    size_t sampled_size = gapped_sampled_data.size();
    assert(original_size >= sampled_size); // the gapped data is sampled from the original data
    // each loop places one key from original data
    for (size_t i = 0; i < original_size; ++i) {
        auto &cur_ori_data = original_data[i];
        auto &pre_ori_data = i > 0 ? original_data[i - 1] : original_data[0];
        auto &cur_gap_sample_data = idx_of_gapped_data < sampled_size ?
                                    gapped_sampled_data[idx_of_gapped_data] : gapped_sampled_data[sampled_size - 1];
        // if the point is sampled, i.e., x_i == x'_i
        if (cur_ori_data.first == cur_gap_sample_data.first) {
            // due to the combination of sampling and gap,
            // there may be gapped keys whose positions are less than the keys of un-sampled
            if (i > 0 and cur_gap_sample_data.second < (pre_ori_data.second + 1)) {
                cur_ori_data.second = pre_ori_data.second + 1;
            } else {
                cur_ori_data.second = cur_gap_sample_data.second;
            }
            idx_of_gapped_data++;
        } else {
            cur_ori_data.second = pre_ori_data.second + 1;
        }
    }
    assert(idx_of_gapped_data == gapped_sampled_data.size()); // faced ``idx_of_gapped_data'' times for sampled keys
}


/*
 *  according to the gap-inserted data, update the positions of original whole data,
 *  the sequential-with-linking-array strategy places the un-sampled key in the predicted positions first,
 *  if face conflict, then place them into sequentially, and finally link them on the rightest sampled keys.
 */
template<typename key_type_transformed, typename index_type>
void update_pos_by_gaps_sequential_with_linking(std::vector<std::pair<key_type_transformed, size_t>> &original_data,
                                                std::vector<std::pair<key_type_transformed, size_t>> const &gapped_sampled_data,
                                                index_type &learned_index) {
    size_t original_size = original_data.size();
    size_t sampled_size = gapped_sampled_data.size();
    assert(original_size >= sampled_size); // the gapped data is sampled from the original data
    size_t idx_of_gapped_data = 0;
    // each loop places one key from original data
    for (size_t i = 0; i < original_size; ++i) {
        auto &cur_ori_data = original_data[i];
        auto &pre_ori_data = i > 0 ? original_data[i - 1] : original_data[0];
        auto &cur_gap_sample_data = idx_of_gapped_data < sampled_size ?
                                    gapped_sampled_data[idx_of_gapped_data] : gapped_sampled_data[sampled_size - 1];
        // if the point is sampled, i.e., x_i == x'_i
        if (cur_ori_data.first == cur_gap_sample_data.first) {
            cur_ori_data.second = cur_gap_sample_data.second;
            idx_of_gapped_data++;
            assert(cur_ori_data.second >= pre_ori_data.second);
            //} else if (cur_ori_data.first < cur_gap_sample_data.first){
        } else {
            auto pos_predicted = learned_index.predict_position(cur_ori_data.first).pos;
            if (pos_predicted <= pre_ori_data.second) {
                // to maintain the monotony, i.e., the ``sequential'' placement
                cur_ori_data.second = pre_ori_data.second + 1;
            } else {
                // place in the position based on model predictions
                cur_ori_data.second = pos_predicted;
            }
            if (cur_ori_data.second >= cur_gap_sample_data.second) {
                // bounded in the tail of gaps between two sampled keys, i.e., tailed in the ``linking'' array
                cur_ori_data.second = cur_gap_sample_data.second - 1;
            }
            assert(cur_ori_data.second >= pre_ori_data.second);
        }
        // else {
        //     assert(idx_of_gapped_data == sampled_size-1);
        //     //cur_ori_data.first > cur_gap_sample_data.first, the keys after the last sampled key
        //     cur_ori_data.second = pre_ori_data.second + 1; // sequentially insert in the tail of the whole data
        // }
    }
    assert(idx_of_gapped_data == gapped_sampled_data.size());
}

/*
 *  according to the gap-inserted data, update the positions of original whole data,
 */
template<typename key_type_transformed, typename index_type>
void update_pos_linking_array(std::vector<std::pair<key_type_transformed, size_t>> &original_data,
                              std::vector<std::pair<key_type_transformed, size_t>> const &gapped_sampled_data,
                              index_type &learned_index) {

    size_t original_size = original_data.size();
    size_t sampled_size = gapped_sampled_data.size();
    assert(original_size >= sampled_size); // the gapped data is sampled from the original data
    size_t idx_of_gapped_data = 0;
    for (size_t i = 0; i < original_size; ++i) {
        auto &cur_ori_data = original_data[i];
        auto &pre_ori_data = i > 0 ? original_data[i - 1] : original_data[0];
        auto &cur_gap_sample_data = idx_of_gapped_data < sampled_size ?
                                    gapped_sampled_data[idx_of_gapped_data] : gapped_sampled_data[sampled_size - 1];

        // if the point is sampled
        if (cur_ori_data.first == cur_gap_sample_data.first) {
            cur_ori_data.second = cur_gap_sample_data.second;
            idx_of_gapped_data++;
            assert(cur_ori_data.second >= (pre_ori_data.second));
        } else if (cur_ori_data.first < cur_gap_sample_data.first) {
            auto pos_predicted = learned_index.predict_position(cur_ori_data.first).pos;
            // to maintain the monotonicity of the (x, y) sequence,
            // bound the pos_predicted in [pre_ori_data, cur_gap_sample_data]
            if (pos_predicted <= pre_ori_data.second) {
                pos_predicted = pre_ori_data.second;
            }
            if (pos_predicted >= cur_gap_sample_data.second) {
                pos_predicted = cur_gap_sample_data.second;
            }
            original_data[i].second = pos_predicted;
            assert(cur_ori_data.second >= pre_ori_data.second);
        } else {
            assert(idx_of_gapped_data == sampled_size - 1);
            //cur_ori_data.first > cur_gap_sample_data.first, the keys after the last sampled key
            cur_ori_data.second = pre_ori_data.second + 1; // sequentially insert in the tail of the whole data
        }
    }
    assert(idx_of_gapped_data == gapped_sampled_data.size());
}

/*
 *  according to the gap-inserted data, update the positions of original whole data,
 */
template<typename key_type_transformed, typename index_type>
std::vector<std::pair<key_type_transformed, size_t>>
update_pos_by_gaps_sequential_with_linking_array(std::vector<std::pair<key_type_transformed, size_t>> original_data,
                                                 std::vector<std::pair<key_type_transformed, size_t>> gapped_data,
                                                 index_type learned_index) {
    assert(original_data.size() >= gapped_data.size()); // the gapped data is sampled from the original data
    size_t idx_of_gapped_data = 0;
    size_t original_size = original_data.size();
    for (size_t i = 0; i < original_size; ++i) {
        // for debug
        if (i == 15190090) {
            bool stop_to_debug = 0;
        }
        // if the point is sampled
        if (original_data[i].first == gapped_data[idx_of_gapped_data].first) {
            if (i > 0 and gapped_data[idx_of_gapped_data].second < (original_data[i - 1].second + 1)) {
                original_data[i].second = original_data[i - 1].second + 1;
            } else {
                original_data[i].second = gapped_data[idx_of_gapped_data].second;
            }
            idx_of_gapped_data++;
        }
            // handling the un-sampled keys from original data.
        else if (original_data[i].first < gapped_data[idx_of_gapped_data].first) {
            auto pos_predicted = learned_index.predict_position(original_data[i].first).pos;
            // facing prediction conflict, we put it in the same pos, i.e., the linking array strategy
            if (pos_predicted <= original_data[i - 1].second) {
                original_data[i].second = original_data[i - 1].second;
            } else {
                original_data[i].second = pos_predicted;
            }
        }
    }
    assert(idx_of_gapped_data == gapped_data.size());
    return original_data;
}



/***
 * MISC
 */


template<typename Iterator>
std::pair<double_t, double_t> train_linear_model(Iterator first, Iterator end, bool compress_key = false,
                                                 bool sequential_y = false) {
    double dx, dx2, mean_x_ (0.0), mean_y_ (0.0), c_ (0.0), m2_ (0.0);
    double alpha_ (0.0), beta_;
    size_t data_size = 0;
    auto seg_start = (*first).first;
    int i = 0, n_ = 0;
    for (Iterator it = first; it != end; it++, i++) {
        double x = (*it).first;
        if (compress_key) {
            x = log(x);
        }
        x = x - seg_start;
        size_t y = i;
        if (not sequential_y) {
            y = (*it).second;
        }
        n_ += 1;
        dx = x - mean_x_;
        mean_x_ += dx / double(n_);
        mean_y_ += (y - mean_y_) / double(n_);
        c_ += dx * (y - mean_y_);

        dx2 = x - mean_x_;
        m2_ += dx * dx2;
        data_size++;
    }
    if (data_size == 0) {
        alpha_ = 0;
        beta_ = 0;
        return std::make_pair(alpha_, beta_);
    } else if (data_size == 1 and isEqual(alpha_, 0.0)) {
        alpha_ = mean_y_;
        beta_ = 0;
        return std::make_pair(alpha_, beta_);
    }

    double cov = c_ / double(n_ - 1), var = m2_ / double(n_ - 1);
    assert(var >= 0.0);
    if (isEqual(var, 0.0)) {
        alpha_ = mean_y_;
        beta_ = 0;
        return std::make_pair(alpha_, beta_);
    }

    beta_ = cov / var;
    alpha_ = mean_y_ - beta_ * mean_x_;
    return std::make_pair(alpha_, beta_);
}


// variadic templates
#include <tuple>
#include <utility>

template<std::size_t I = 0, typename FuncT, typename... Tp>
inline typename std::enable_if<I == sizeof...(Tp), void>::type
for_each(const std::tuple<Tp...> &, FuncT) // Unused arguments are given no names.
{}

template<std::size_t I = 0, typename FuncT, typename... Tp>
inline typename std::enable_if<I < sizeof...(Tp), void>::type
for_each(const std::tuple<Tp...> &t, FuncT f) {
    f(std::get<I>(t));
    for_each<I + 1, FuncT, Tp...>(t, f);
}

template<std::size_t I = 0, typename FuncT, typename... Tp>
inline typename std::enable_if<I == sizeof...(Tp), void>::type
for_each(std::tuple<Tp...> &&, FuncT) // Unused arguments are given no names.
{}

template<std::size_t I = 0, typename FuncT, typename... Tp>
inline typename std::enable_if<I < sizeof...(Tp), void>::type
for_each(std::tuple<Tp...> &&t, FuncT f) {
    f(std::get<I>(t));
    for_each<I + 1, FuncT, Tp...>(std::move(t), f);
}


#endif //LEARNED_INDEX_UTILITIES_HPP
