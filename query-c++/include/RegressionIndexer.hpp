//
//

#ifndef LEARNED_INDEX_RegressionIndexer_HPP
#define LEARNED_INDEX_RegressionIndexer_HPP


#include <vector>
#include <list>
#include <chrono>
#include <cstddef>
#include <dlib/matrix.h>
#include <dlib/statistics.h>
#include "cstddef"
#include "cstdio"
#include "cstdlib"
#include "cstring"
#include "PgmIndexer.hpp"
#include <algorithm>
#include "ctime"
#include "Utilities.hpp"
#include <dlib/svm.h>
#include <dlib/dnn.h>



/*
 * biased binary search, the first middle point is set to be the predicted pos
 * revised from std::lower_bound
 */
template<class ForwardIt, class T, typename _Compare >
static ForwardIt biased_lower_bound(ForwardIt first, ForwardIt last, const T& value, const T& pos,  _Compare __comp)
{
    ForwardIt it;
    typename std::iterator_traits<ForwardIt>::difference_type count, step;
    count = std::distance(first, last);
    //count = pos * 2;
    //assert((pos == 0) or (count > pos));
    int search_count = 0;
    while (count > 0) {
        it = first;
        step = search_count == 0 ? pos : (count / 2) ; // the first search point (mid_point) is pos
        if (__comp(*it, value)) {
        //if (*it < value) {
            first = ++it;
            count -= step + 1;
        }
        else
            count = step;
        search_count ++;
    }
    return first;
}



template <typename KeyType, typename Iterator>
class RegressionIndexer {
public:
    // 1 dimensional column for the key type
    //typedef dlib::matrix<float, 1, 1> sample_type;
    typedef dlib::matrix<double> sample_type;

    Iterator first_key_iter_, last_key_iter_;
    size_t data_size_, first_key_, max_x_;

     using net_type = dlib::loss_mean_squared<
             dlib::fc<1, dlib::relu< dlib::fc<10000,
             dlib::input<sample_type>
             >>>>;

    net_type regression_net;
    using trainer_type = dlib::dnn_trainer<net_type>;
    trainer_type trainer_;

    RegressionIndexer(Iterator first_key_iter, Iterator last_key_iter, size_t data_size, dlib::sgd defsolver) :
            first_key_iter_(first_key_iter), last_key_iter_(std::prev(last_key_iter)),
            data_size_(data_size), regression_net(), trainer_(regression_net, defsolver){
        max_x_ = *std::prev(last_key_iter_);
    };

    void set_optimizer(float learning_rate, float min_lr, int batch_size, int iter_num){

        trainer_.set_learning_rate(learning_rate);
        trainer_.set_min_learning_rate(min_lr);
        trainer_.set_mini_batch_size(batch_size);
        trainer_.set_max_num_epochs(iter_num);
        trainer_.be_verbose();
        //trainer_.set_synchronization_file("nn_regression_sync_file", std::chrono::seconds(100));
    }


   void learn_index(Iterator first_iter, Iterator last_iter) try{
        std::vector<sample_type> samples;
        std::vector<float> targets;
        float i = 0;
        double max = 0;
        first_key_ = (*first_iter);
        dlib::matrix<double> tmp(1,1);
        for (Iterator it = first_iter; it != last_iter; it++, i++) {
            tmp = double (*it - first_key_);
            max = tmp > max ? tmp :max;
            //sample_type x;
            //x(0) = float (*it - first_key_);
            //samples.emplace_back(x);
            //samples.emplace_back(tmp);
            samples.emplace_back(tmp/ max_x_);
            targets.emplace_back(double_t(i) / data_size_);
        }
        trainer_.train(samples, targets);
    }
   catch(std::exception& e)
   {
       std::cout << e.what() << std::endl;
   }

    KeyType query(KeyType key) {
        std::vector<sample_type> samples;
        samples.emplace_back(sample_type(key));
        return regression_net(samples);
    }

    void evaluate_indexer() {
        std::vector<float> all_predicted_pos;
        auto t0 = std::chrono::high_resolution_clock::now();
        for(Iterator it = first_key_iter_; it != last_key_iter_; it++){
            float pos = predict_position(*it);
            all_predicted_pos.emplace_back(pos);
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        int i = 0;
        for(Iterator it = first_key_iter_; it != last_key_iter_; it++, i++){
            correct_position(all_predicted_pos[i], *it);
        }
        auto t2 = std::chrono::high_resolution_clock::now();
        //lookup-time per query
        auto predict_time = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count() / data_size_;
        auto correct_time = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() / data_size_;
        //size
        size_t size_in_bytes = this->calculate_size_in_bytes();

        // ML-oriented metric
        // regress_metric = ["RMSE", "MAE", "R2"]
        std::vector<size_t> all_true_pos;
        for (size_t i=0; i < data_size_; i++){
            all_true_pos.push_back(i);
        }
        dlib::matrix<double, 1, 4> results;
        results =  evaluate_regression_ML(all_predicted_pos, all_true_pos);

        std::cout<< "predict time: " << predict_time << std::endl;
        std::cout<< "correct time: " << correct_time << std::endl;
        std::cout << "size in bytes: " << size_in_bytes << std::endl;
        std::cout<< "ML oriented matricx: " << results<< std::endl;

    }

    float predict_position(KeyType key) {
        std::vector<sample_type> samples;
        dlib::matrix<double> x(1,1);
        x(0) = double(key) / max_x_;
        samples.emplace_back(x);
        std::vector<float> predicted_labels = regression_net(samples);
        //std::vector<float> predicted_labels;
        //std::regression_net(samples.begin(), samples.end(), predicted_labels.begin());
        double_t predicted_label = predicted_labels.front() * data_size_;
        return round(predicted_label);
        //return predicted_labels;
    }


    KeyType correct_position(float predicted_pos, KeyType key){
         Iterator true_pos  = biased_lower_bound(first_key_iter_, last_key_iter_, key,
                KeyType(predicted_pos), CompareForDataPair<KeyType>());
        //Iterator true_pos  = exponential_search(first_key_iter_, last_key_iter_, key,
                                                //KeyType(predicted_pos), CompareForDataPair<KeyType>());
        return true_pos - first_key_iter_;

    };

/**
* Returns the size in bytes of the learned index of regression indexer, i.e., the machine learning parameters
* @return the size in bytes of the index.
*/
    size_t calculate_size_in_bytes() const {
        // for SVR, the parameters are support vectors, the alpha and the bias term, b.
        //size_t para_size = df.alpha.NC * df.alpha.NR  + df.basis_vectors.NC + df.basis_vectors.NR + 1;
        size_t para_size = 1 * 100 + 100 * 1;
        size_t size_in_bytes = para_size * sizeof(float);

        return size_in_bytes;
    }
};


template <typename KeyType, typename Iterator>
class KRR_SVR_RegressionIndexer {
public:
    // 1 dimensional column for the key type
    typedef dlib::matrix<float,1,1> sample_type;
    typedef dlib::radial_basis_kernel<sample_type> default_kernel_type;
    dlib::svr_trainer<default_kernel_type> trainer;
    //dlib::krr_trainer<default_kernel_type> trainer;
    dlib::decision_function<default_kernel_type> df;

    Iterator first_key_iter_, last_key_iter_;
    size_t data_size_;

    KRR_SVR_RegressionIndexer(Iterator first_key_iter, Iterator last_key_iter, size_t data_size) :
        first_key_iter_(first_key_iter), last_key_iter_(std::prev(last_key_iter)), data_size_(data_size) {};

    // epsilon-insensitive support vector regression, ref. http://dlib.net/svr_ex.cpp.html
    void set_SVR_parameters(float kernel_gamma) {
        trainer.set_kernel(default_kernel_type(kernel_gamma));
    }
    void set_KRR_parameters() {
    }
    void learn_index(Iterator first_iter, Iterator last_iter) {
        std::vector<sample_type> samples;
        std::vector<float> targets;
        size_t i = 0;
        for (Iterator it = first_iter; it != last_iter; it++, i++) {
            samples.emplace_back(float (*it));
            targets.emplace_back(float (i));
        }
        df = trainer.train(samples, targets);
    }

    KeyType query(KeyType key) {
        sample_type m;
        m(0) = float(key);
        return df(m);
    }

    void evaluate_indexer() {
        std::vector<float> all_predicted_pos;
        auto t0 = std::chrono::high_resolution_clock::now();
        for(Iterator it = first_key_iter_; it != last_key_iter_; it++){
            float pos = predict_position(*it);
            all_predicted_pos.emplace_back(pos);
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        int i = 0;
        for(Iterator it = first_key_iter_; it != last_key_iter_; it++, i++){
            correct_position(all_predicted_pos[i], *it);
        }
        auto t2 = std::chrono::high_resolution_clock::now();
        //lookup-time per query
        auto predict_time = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count() / data_size_;
        auto correct_time = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() / data_size_;
        //size
        size_t size_in_bytes = this->calculate_size_in_bytes();

        // ML-oriented metric
        // regress_metric = ["RMSE", "MAE", "R2"]
        std::vector<size_t> all_true_pos;
        for (size_t i=0; i < data_size_; i++){
            all_true_pos.push_back(i);
        }
        dlib::matrix<float, 1, 4> results;
        results =  evaluate_regression_ML(all_predicted_pos, all_true_pos);

        std::cout<< "predict time: " << predict_time << std::endl;
        std::cout<< "correct time: " << correct_time << std::endl;
        std::cout << "size in bytes: " << size_in_bytes << std::endl;
        std::cout<< "ML oriented matricx: " << results<< std::endl;


    }

    float predict_position(KeyType key) const{
        sample_type m;
        m(0) = float(key);
        return df(m);
    }


    KeyType correct_position(float predicted_pos, KeyType key){
        Iterator true_pos  = biased_lower_bound(first_key_iter_, last_key_iter_, key, KeyType(predicted_pos));
        return true_pos - first_key_iter_;

    };

/**
* Returns the size in bytes of the learned index of regression indexer, i.e., the machine learning parameters
* @return the size in bytes of the index.
*/
size_t calculate_size_in_bytes() const {
    // for SVR, the parameters are support vectors, the alpha and the bias term, b.
    size_t para_size = df.alpha.NC * df.alpha.NR  + df.basis_vectors.NC + df.basis_vectors.NR + 1;
    size_t size_in_bytes = para_size * sizeof(float);

    return size_in_bytes;
}
};





#endif //LEARNED_INDEX_BTREE_INDEXER_HPP
