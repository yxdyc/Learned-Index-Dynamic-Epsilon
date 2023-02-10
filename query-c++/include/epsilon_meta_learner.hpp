//
// Created by daoyuan on 2020/10/26.
//

#ifndef LEARNEDINDEXONLINELEARNING_EPSILON_META_LEARNER_HPP
#define LEARNEDINDEXONLINELEARNING_EPSILON_META_LEARNER_HPP

using namespace std;

template<typename KeyType, typename Iterator>
class EpsilonMetaLearner {

    struct LinearModel{
        double alpha_, beta_;
        size_t lower_error_, upper_error_; // both are positive numbers
        // the "mean_x, mean_y, c, m2, n" are stored to support online model update (re-training)
        double mean_x_, mean_y_, c_, m2_;
        size_t n_;

        LinearModel(){
            alpha_ = 0;
            beta_ = 0;
            n_ = 0;
            mean_x_ = 0;
            mean_y_ = 0;
            c_= 0.0;
            m2_ = 0.0;
            lower_error_ = std::numeric_limits<unsigned long>::min();
            upper_error_ = std::numeric_limits<unsigned long>::min();
        }
        LinearModel(double alpha, double beta, size_t n):
                alpha_(alpha), beta_(beta), n_(n){
            n_ = 0;
            mean_x_ = 0;
            mean_y_ = 0;
            c_= 0.0;
            m2_ = 0.0;
        };

        inline double linear(double inp) {
            return alpha_ + beta_ * inp;
        }

        /*
         * train the linear model
         *
         * @params compress_key weather transform the keys using log()
         * @params second_model_size, all_data_size used to re-scale the model
         */
        void train(Iterator first, Iterator end, bool compress_key = false,
                   size_t second_model_size=0, size_t all_data_size = 0){
            double dx , dx2;
            size_t data_size = 0;
            for (Iterator it = first; it != end; it ++){
                double x = (*it).first;
                if (compress_key){
                    //x = log(x - 20170117000000 + 1);
                    x = log(x);
                }
                size_t y = (*it).second;
                if (second_model_size > 0){
                    // if second_model_size > 0, indicating the non-last layer model, rescale the y
                    y = floor(double (y) * second_model_size / all_data_size);
                }
                n_ += 1;
                dx = x - mean_x_;
                mean_x_ += dx / double(n_);
                mean_y_ += (y - mean_y_) / double(n_);
                c_ += dx * (y - mean_y_);

                dx2 = x - mean_x_;
                m2_ += dx * dx2;
                data_size ++;
            }
            if (data_size == 0){
                alpha_ = 0;
                beta_ = 0;
                return;
            } else if (data_size == 1 and alpha_ == 0){
                alpha_ = mean_y_;
                beta_ = 0;
                return;
            }

            double cov = c_ / double(n_ - 1), var = m2_ / double (n_ - 1);
            assert(var >= 0.0);
            if (var == 0.0){
                alpha_ = mean_y_;
                beta_ = 0;
                return;
            }

            beta_ = cov / var;
            alpha_ = mean_y_ - beta_ * mean_x_;
            return;
        }

        /*
         * the standard RMI stores the min-error and max-error for every model on the last stage, such that:
         * each key can be searched by binary search with respective lower and upper bound for every model
         */
        void track_lower_upper_bound(Iterator begin, Iterator end){
            for (auto it = begin; it != end; it ++){
                long pred_error = long(round(linear((*it).first))) - long((*it).second);
                if (pred_error < 0) {
                    pred_error = abs(pred_error);
                    //lower_error_ = pred_error > lower_error_ ? pred_error : lower_error_;
                    upper_error_ = pred_error > upper_error_ ? pred_error : upper_error_;
                } else {
                    //upper_error_ = pred_error > upper_error_ ? pred_error : upper_error_;
                    lower_error_ = pred_error > lower_error_ ? pred_error : lower_error_;
                }
            }
        }

    };

    LinearModel linearApproximator; // approximate the relationship between density & golden epsilon

    inline double get_density(long num_data, KeyType key_x1, KeyType key_x2){
        assert(key_x2 > key_x1);
        return double (num_data) / (key_x2-key_x1);
    }

    inline double inference (std::vector<KeyType> meta_data){
        auto density = get_density(meta_data.size(), meta_data.first(), meta_data.last());
        return density;
    }


};


#endif //LEARNEDINDEXONLINELEARNING_EPSILON_META_LEARNER_HPP
