//
// Created by daoyuan on 2020/12/26.
//

#ifndef LEARNED_INDEX_SAMPLE_ALGO_HPP
#define LEARNED_INDEX_SAMPLE_ALGO_HPP

#include <algorithm>
#include <dlib/geometry/vector.h>
#include <vector>

template<typename _PopulationIterator, typename _SampleContainer,
        typename _SampleRate, typename _UniformRandomNumberGenerator>
std::vector<unsigned int> *random_sample(_PopulationIterator __first, _PopulationIterator __last,
                                         _SampleContainer &__out, _SampleRate s,
                                         _UniformRandomNumberGenerator &&__g) {
    /*
     * randomly sample a order-preserving subset with sample rate s.
     */
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    if (s < 0.0 or s > 1.0) {
        std::cout << "You specified a illegal sample rate: " << s << std::endl;
        return nullptr;
    }

    //analysis
    auto cur_x = (*(__first)).first;
    auto last_x = (*(__first)).first;
    auto cur_y = (*(__first)).second;
    auto last_y = (*(__first)).second;
    double cur_delta_x;
    long cur_delta_y;
    double cur_slope;
    std::vector<double> all_slopes;

    double rand_number;
    long input_size = std::distance(__first, __last);
    auto sampled_pos = new std::vector<unsigned int>();
    sampled_pos->reserve(input_size * s);
    for (unsigned int i = 0; i < input_size; i++) {
        rand_number = distribution(__g);
        if (rand_number <= s) {
            __out.emplace_back(*(__first + i));
            sampled_pos->emplace_back(i);

            cur_x = (*(__first + i)).first;
            cur_y = (*(__first + i)).second;
            cur_delta_x = cur_x - last_x;
            cur_delta_y = cur_y - last_y;
            cur_slope = double(cur_delta_y) / cur_delta_x;
            all_slopes.emplace_back(cur_slope);
        }
        last_x = cur_x;
        last_y = cur_y;
    }

    //analysis
    std::cout << "statistic of all slopes: ";
    basic_statistic(all_slopes);

    return sampled_pos;
}


template<typename _PopulationIterator, typename _SampleContainer,
        typename _SampleRate, typename _UniformRandomNumberGenerator>
void fix_first_rand_step_sample(_PopulationIterator __first, _PopulationIterator __last,
                                _SampleContainer &__out, _SampleRate s,
                                _UniformRandomNumberGenerator &&__g, int first_at_M = 0) {
    /*
     * fixed sample the M-th data as first data, then randomly sample the remaining data with sample rate s.
     */
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    if (s < 0.0 or s > 1.0) {
        std::cout << "You specified a illegal sample rate: " << s << std::endl;
        return;
    }

    //analysis
    auto cur_x = (*(__first)).first;
    auto last_x = (*(__first)).first;
    auto cur_y = (*(__first)).second;
    auto last_y = (*(__first)).second;
    double cur_delta_x;
    long cur_delta_y;
    double cur_slope;
    std::vector<double> all_slopes;

    long i = 0;

    __out.emplace_back(*(__first + first_at_M));
    i += first_at_M;

    double rand_number;
    long input_size = std::distance(__first, __last);
    while (i < input_size) {
        rand_number = distribution(__g);
        if (rand_number <= s) {
            __out.emplace_back(*(__first + i));

            cur_x = (*(__first + i)).first;
            cur_y = (*(__first + i)).second;
            cur_delta_x = cur_x - last_x;
            cur_delta_y = cur_y - last_y;
            cur_slope = double(cur_delta_y) / cur_delta_x;
            all_slopes.emplace_back(cur_slope);
        }

        last_x = cur_x;
        last_y = cur_y;
        i++;
    }

    //analysis
    std::cout << "statistic of all slopes: ";
    basic_statistic(all_slopes);

    return;
}


/*
 * Given an initial s, which is very small
 * adaptive sampling with a sliding window, adjust the sample step
 */
template<typename _PopulationIterator, typename _SampleContainer,
        typename _SampleRate>
void sliding_window_by_step(_PopulationIterator __first, _PopulationIterator __last,
                            _SampleContainer &__out, _SampleRate s,
                            int window_size = 5, double error_bound = 1.5) {
    if (s < 0.0 or s > 1.0) {
        std::cout << "You specified an illegal sample rate: " << s << std::endl;
        return;
    }
    long input_size = std::distance(__first, __last);

    long i = 0;
    __out.emplace_back(*(__first + i));    // always select the first element

    int fix_step = ceil(1 / s);
    int max_step = round(double(fix_step) * error_bound);

    auto cur_x = (*(__first)).first;
    auto last_x = (*(__first)).first;
    std::vector<double> slopes_over_windows; // a circular buffer to store the slopes of past W windows
    for (int i = 0; i < window_size; i++) {
        slopes_over_windows.emplace_back(0.0);
    }

    double cur_mean_slope_of_window = 0.0;
    double last_mean_slope_of_window = 0.0;

    double slope_variance_of_window = 0.0;
    double var_sum = 0.0;


    long number_sampled_size = 0;
    long number_increase_step = 0;
    long number_decrease_step = 0;

    double cur_delta_x;
    double cur_slope;

    double rel_diff_baseline = 0.0; // auto-regressive baseline used to wave the steps, auto-regressive
    long rel_diff_n = 1;

    while (i < (input_size - fix_step)) {

        i += fix_step;
        __out.emplace_back(*(__first + i));
        number_sampled_size++;

        cur_x = (*(__first + i)).first;
        cur_delta_x = cur_x - last_x;
        cur_slope = double(fix_step) / cur_delta_x;
        auto next_window_i = (number_sampled_size + 1) % window_size;
        // calculate density variance using Welford's online algorithm
        cur_mean_slope_of_window =
                last_mean_slope_of_window +
                (cur_slope - slopes_over_windows[next_window_i]) / double(window_size);

        var_sum = var_sum +
                  (cur_slope - last_mean_slope_of_window) * (cur_slope - cur_mean_slope_of_window) -
                  (slopes_over_windows[next_window_i] - last_mean_slope_of_window) *
                  (slopes_over_windows[next_window_i] - cur_mean_slope_of_window);

        // std::cout << "Current var_sum: " << var_sum << std::endl;

        slopes_over_windows[next_window_i] = cur_slope;
        last_x = cur_x;
        last_mean_slope_of_window = cur_mean_slope_of_window;

        if (number_sampled_size % window_size == 0) {
            // adjust step every window_size keys
            slope_variance_of_window = var_sum / double(number_sampled_size);
            if (slope_variance_of_window <
                0) { slope_variance_of_window = 0.0; } // underflow due to numerical instability
            double standard_deviation = sqrt(slope_variance_of_window);
            double relative_diff = (standard_deviation) / last_mean_slope_of_window; // [0%, 100%]

            // calculate the baseline as "running mean of the relative_diff"
            rel_diff_baseline = rel_diff_baseline + (relative_diff - rel_diff_baseline) / rel_diff_n;
            rel_diff_n++;

            double relative_diff_minus_basline = relative_diff - rel_diff_baseline;
#ifdef Debug
            std::cout << "[Before adjust] Current relative_diff of densities within windows, baseline, step are: "
                      << relative_diff << ", " << rel_diff_baseline << ", " << fix_step << std::endl;
#endif
            // too small variance, thus we can increase step to skip redundant sampling
            if (relative_diff_minus_basline < 0) {
                //if (relative_diff_minus_basline < 0 and fix_step < max_fix_step) {
                fix_step = ceil(fix_step * (1 - relative_diff_minus_basline));
                if (fix_step > max_step) { fix_step = max_step; }
                number_increase_step++;
            }
            // too large variance, thus we can reduce step to add more fine-grained points
            if (relative_diff_minus_basline > 0) {
                //if (relative_diff_minus_basline > 0 and fix_step > min_fix_step) {
                fix_step = floor(fix_step * (1 - relative_diff_minus_basline));
                number_decrease_step++;
            }
#ifdef Debug
            std::cout << "[After adjust] Current relative_diff of densities within windows, baseline, step are: "
                      << relative_diff << ", " << rel_diff_baseline << ", " << fix_step << std::endl;
#endif
            if (rel_diff_n % (window_size + 1) == 0) { // the plus 1 is because that rel_diff_n begins from 1
                // // clear the baseline every W times, for numerical stability
                rel_diff_baseline = 0.0;
                rel_diff_n = 1;
                // re-set the step as 1/s to make it wave as a cycling manner
                fix_step = ceil(1 / s);
            }
        }
    }
    std::cout << "Window size, number of increse step, decrese step, ori s, varied s are: " <<
              window_size << ", " << number_increase_step << ", "
              << number_decrease_step << ", " << s << ", "
              << (double(number_sampled_size) / double(input_size)) << std::endl;
}


/*
 * Given an initial s
 * adaptive sampling with a sliding window, adjust the sample rate, then sample with the fix_step strategy
 */
template<typename _PopulationIterator, typename _SampleContainer,
        typename _SampleRate>
void sliding_window_by_s_fix(_PopulationIterator __first, _PopulationIterator __last,
                             _SampleContainer &__out, _SampleRate s,
                             int window_size = 5) {
    if (s < 0.0 or s > 1.0) {
        std::cout << "You specified an illegal sample rate: " << s << std::endl;
        return;
    }
    long input_size = std::distance(__first, __last);

    long i = 0;
    __out.emplace_back(*(__first + i));    // always select the first element for the fix_first_fix_step strategy

    int fix_step = floor(1.0 / s);
    double adjusted_s = s;

    auto cur_x = (*(__first)).first;
    auto last_x = (*(__first)).first;
    std::vector<double> slopes_over_windows; // a circular buffer to store the slopes of past W windows
    for (int i = 0; i < window_size; i++) {
        slopes_over_windows.emplace_back(0.0);
    }

    double cur_mean_slope_of_window = 0.0;
    double last_mean_slope_of_window = 0.0;

    double slope_variance_of_window = 0.0;
    double var_sum = 0.0;
    double decay_factor_sum = 0.0;


    long number_sampled_size = 0;
    long number_increase_step = 1;
    long number_decrease_step = 1;
    double degree_increase_step = 0.0;
    double degree_decrease_step = 0.0;
    double s_rel_diff;

    double cur_delta_x;
    double cur_slope;

    double rel_diff_baseline = 0.0; // auto-regressive baseline used to wave the steps, auto-regressive
    long rel_diff_n = 1;

    double standard_deviation;
    double relative_diff;
    double relative_diff_minus_basline;


    // for analysis
    std::vector<double> all_adjusted_s;
    std::vector<double> all_slopes;
    std::vector<double> all_slope_stardand_variances;
    all_adjusted_s.emplace_back(s);
    dlib::running_scalar_covariance<double> rc;


    while (i < (input_size - fix_step)) {

        i += fix_step;
        __out.emplace_back(*(__first + i));
        number_sampled_size++;

        cur_x = (*(__first + i)).first;
        cur_delta_x = cur_x - last_x;
        cur_slope = double(fix_step) / cur_delta_x;
        auto next_window_i = (number_sampled_size + 1) % window_size;
        // calculate density variance using Welford's online algorithm
        cur_mean_slope_of_window =
                last_mean_slope_of_window +
                (cur_slope - slopes_over_windows[next_window_i]) / double(window_size);

        var_sum = var_sum +
                  (cur_slope - last_mean_slope_of_window) * (cur_slope - cur_mean_slope_of_window) -
                  (slopes_over_windows[next_window_i] - last_mean_slope_of_window) *
                  (slopes_over_windows[next_window_i] - cur_mean_slope_of_window);
        // a negative variance is mathematically impossible, when the variance very small compared to the square of the
        // mean and leads to negative number due to the numerical instability, we patch it as 0.0 manually
        if (var_sum < 0) { var_sum = 0.0; }

        // std::cout << "Current var_sum: " << var_sum << std::endl;

        slopes_over_windows[next_window_i] = cur_slope;
        last_x = cur_x;
        last_mean_slope_of_window = cur_mean_slope_of_window;

        if (number_sampled_size % window_size == 0) {
            // adjust step every window_size keys
            slope_variance_of_window = var_sum / double(number_sampled_size);
            if (slope_variance_of_window <
                0) { slope_variance_of_window = 0.0; } // underflow due to numerical instability
            standard_deviation = sqrt(slope_variance_of_window);
            relative_diff = (standard_deviation) / last_mean_slope_of_window; // [0%, 100%]

            // calculate the baseline as "running mean of the relative_diff"
            rel_diff_baseline = rel_diff_baseline + (relative_diff - rel_diff_baseline) / rel_diff_n;
            rel_diff_n++;

            relative_diff_minus_basline = relative_diff - rel_diff_baseline;

            if (rel_diff_n % (window_size + 1) == 0) { // the plus 1 is because that rel_diff_n begins from 1
                // // clear the baseline every W times adjustment, for numerical stability
                rel_diff_baseline = 0.0;
                rel_diff_n = 1;
            }
#ifdef Debug
            std::cout << "[Before adjust] Current relative_diff of densities within windows, baseline, step are: "
                      << relative_diff << ", " << rel_diff_baseline << ", " << fix_step << std::endl;
#endif
            s_rel_diff = std::abs((adjusted_s - s) / s);

            auto decay_factor = 0.0;
            auto diff_degree_increase_minus_decrease = (degree_increase_step - degree_decrease_step);
            if (degree_decrease_step > 0 and degree_increase_step > 0 and
                (diff_degree_increase_minus_decrease * relative_diff_minus_basline > 0)) {
                //acce_factor = std::abs(diff_degree_increase_minus_decrease / degree_decrease_step);
                decay_factor = s_rel_diff;
            }
            // too large variance, thus we can increase s to add more fine-grained points
            if (relative_diff_minus_basline > 0) {
                //degree_increase_step += relative_diff_minus_basline;
                //degree_increase_step += (1 + relative_diff_minus_basline * (1 + s_rel_diff));
                degree_increase_step += relative_diff_minus_basline * (1 - decay_factor);
                number_increase_step++;
            }
            // too small variance, thus we can reduce s to skip redundant sampling
            if (relative_diff_minus_basline < 0) {
                //degree_decrease_step -= relative_diff_minus_basline;
                //degree_decrease_step += (1 + relative_diff_minus_basline * (1 + s_rel_diff));
                degree_decrease_step -= relative_diff_minus_basline * (1 - decay_factor);
                number_decrease_step++;
            }
            decay_factor_sum += decay_factor;
            adjusted_s = adjusted_s * (1 + relative_diff_minus_basline * (1 - decay_factor));

            //adjusted_s = adjusted_s * (1 + relative_diff_minus_basline);
            //adjusted_s = s * (1 + relative_diff_minus_basline);


            //adjusted_s = s * (1 + relative_diff_minus_basline) * (1 + s_rel_diff);
            //adjusted_s = adjusted_s * (1 + relative_diff_minus_basline);
            //adjusted_s = adjusted_s * (1 + relative_diff_minus_basline * (1 + s_rel_diff));
            //adjusted_s = adjusted_s * (1 + relative_diff_minus_basline * (1 + acce_factor));
            //adjusted_s = adjusted_s * (1 + relative_diff_minus_basline);
            //adjusted_s = s * (1 + relative_diff_minus_basline * (1 + s_rel_diff));

            fix_step = round(1.0 / adjusted_s);
            if (fix_step < 1 or adjusted_s > 1.0) {
                fix_step = 1;
                adjusted_s = 1.0;
            }


            all_adjusted_s.emplace_back(adjusted_s);
            all_slopes.emplace_back(last_mean_slope_of_window);
            all_slope_stardand_variances.emplace_back(relative_diff);
            rc.add(adjusted_s, cur_slope);

#ifdef Debug

            std::cout
                    << "[After adjust] Current relative_diff of densities within windows, baseline, adjusted_s, step are: "
                    << relative_diff << ", " << rel_diff_baseline << ", " << (1.0 / adjusted_s) << ", " << fix_step
                    << std::endl;
#endif
        }
    }

    //analysis
    std::cout << "statistic of all adjusted s:\n";
    basic_statistic(all_adjusted_s);
    std::cout << "statistic of all slopes:\n";
    basic_statistic(all_slopes);
    std::cout << "correlation of adjusted_s and all slopes: " << rc.correlation() << std::endl;
    write_vector_to_f(all_adjusted_s,
                      "/home/xxx/work/learned_index/build-release/cmake-build-release-remote-gcc5/results/adaptive_study/fixall_all_s.double");
    write_vector_to_f(all_slopes,
                      "/home/xxx/work/learned_index/build-release/cmake-build-release-remote-gcc5/results/adaptive_study/fixall_all_slopes.double");
    write_vector_to_f(all_slope_stardand_variances,
                      "/home/xxx/work/learned_index/build-release/cmake-build-release-remote-gcc5/results/adaptive_study/fixall_all_slope_sigmas.double");

    std::cout
            << "Window size, number of increase and decrease, degree of increase and decrease, ori s, varied s, decay-sum are: "
            <<
            window_size << ", " << number_increase_step << ", "
            << number_decrease_step << ", " << degree_increase_step << ", " << degree_decrease_step << ", " << s
            << ", " << (double(number_sampled_size) / double(input_size)) << ", "
            << decay_factor_sum << std::endl;
}


/*
 * Given an initial s
 * adaptive sampling with a sliding window, adjust the sample rate, then sample with the random strategy
 */
template<typename _PopulationIterator, typename _SampleContainer,
        typename _SampleRate, typename _UniformRandomNumberGenerator>
void sliding_window_by_s_random(_PopulationIterator __first, _PopulationIterator __last,
                                _SampleContainer &__out, _SampleRate s,
                                _UniformRandomNumberGenerator &&__g, int window_size = 5) {
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    if (s < 0.0 or s > 1.0) {
        std::cout << "You specified an illegal sample rate: " << s << std::endl;
        return;
    }
    long input_size = std::distance(__first, __last);

    long i = 0;
    double rand_number;

    double adjusted_s = s;

    auto cur_x = (*(__first)).first;
    auto last_x = (*(__first)).first;
    auto cur_y = (*(__first)).second;
    auto last_y = (*(__first)).second;


    std::vector<double> slopes_over_windows; // a circular buffer to store the slopes of past W windows
    for (int i = 0; i < window_size; i++) {
        slopes_over_windows.emplace_back(0.0);
    }

    double cur_delta_x;
    long cur_delta_y;
    double cur_slope;
    double cur_mean_slope_of_window = 0.0;
    double last_mean_slope_of_window = 0.0;
    double slope_variance_of_window = 0.0;
    double var_sum = 0.0;

    long number_sampled_size = 0;
    long number_increase_step = 0;
    long number_decrease_step = 0;


    double degree_increase = 0.0;
    double degree_decrease = 0.0;
    double s_rel_diff;

    double rel_diff_baseline = 0.0; // auto-regressive baseline used to wave the steps
    long rel_diff_n = 1;


    // for analysis
    std::vector<double> all_adjusted_s;
    std::vector<double> all_slopes;
    all_adjusted_s.emplace_back(s);
    dlib::running_scalar_covariance<double> rc;


    while (i < input_size) {
        rand_number = distribution(__g);
        if (rand_number <= adjusted_s) {
            __out.emplace_back(*(__first + i));
            number_sampled_size++;

            if (i == 0) {
                i++; // sampled first key, calculate the slopes of window from last
                continue;
            }
            cur_x = (*(__first + i)).first;
            cur_y = (*(__first + i)).second;
            cur_delta_x = cur_x - last_x;
            cur_delta_y = cur_y - last_y;
            cur_slope = double(cur_delta_y) / cur_delta_x;
            auto next_window_i = (number_sampled_size + 1) % window_size;
            // calculate density variance using Welford's online algorithm
            cur_mean_slope_of_window =
                    last_mean_slope_of_window +
                    (cur_slope - slopes_over_windows[next_window_i]) / double(window_size);

            var_sum = var_sum +
                      (cur_slope - last_mean_slope_of_window) * (cur_slope - cur_mean_slope_of_window) -
                      (slopes_over_windows[next_window_i] - last_mean_slope_of_window) *
                      (slopes_over_windows[next_window_i] - cur_mean_slope_of_window);

            // std::cout << "Current var_sum: " << var_sum << std::endl;

            slopes_over_windows[next_window_i] = cur_slope;
            last_x = cur_x;
            last_y = cur_y;
            last_mean_slope_of_window = cur_mean_slope_of_window;

            if (number_sampled_size % window_size == 0) {
                // adjust step every window_size keys
                slope_variance_of_window = var_sum / double(number_sampled_size);
                if (slope_variance_of_window <
                    0) { slope_variance_of_window = 0.0; } // underflow due to numerical instability
                double standard_deviation = sqrt(slope_variance_of_window);
                double relative_diff = (standard_deviation) / last_mean_slope_of_window; // [0%, 100%]

                // calculate the baseline as "running mean of the relative_diff"
                rel_diff_baseline = rel_diff_baseline + (relative_diff - rel_diff_baseline) / rel_diff_n;
                rel_diff_n++;

                double relative_diff_minus_basline = relative_diff - rel_diff_baseline;
#ifdef Debug
                std::cout
                        << "[Before adjust] Current relative_diff of densities within windows, baseline, s, "
                           "slope_variance_of_window are: " << relative_diff << ", " << rel_diff_baseline << ", "
                        << adjusted_s << ", " << slope_variance_of_window << ", " << std::endl;
#endif

                s_rel_diff = abs((adjusted_s - s) / s);


                auto decay_factor = 0.0;
                auto diff_degree_increase_minus_decrease = (degree_increase - degree_decrease);
                if (degree_decrease > 0 and degree_increase > 0 and
                    (diff_degree_increase_minus_decrease * relative_diff_minus_basline > 0)) {
                    decay_factor = s_rel_diff;
                }
                // too large variance, thus we can increase s to add more fine-grained points
                if (relative_diff_minus_basline > 0) {
                    degree_increase += relative_diff_minus_basline * (1 - decay_factor);
                    number_increase_step++;
                }
                // too small variance, thus we can reduce s to skip redundant sampling
                if (relative_diff_minus_basline < 0) {
                    degree_decrease -= relative_diff_minus_basline * (1 - decay_factor);
                    number_decrease_step++;
                }

                adjusted_s = s * (1 + relative_diff_minus_basline * (1 - decay_factor));

                all_adjusted_s.emplace_back(adjusted_s);
                all_slopes.emplace_back(last_mean_slope_of_window);
                rc.add(adjusted_s, cur_slope);


#ifdef Debug
                std::cout << "[After adjust] Current relative_diff of densities within windows, baseline, s are: "
                          << relative_diff << ", " << rel_diff_baseline << ", " << adjusted_s << ", " << std::endl;
#endif
                if (rel_diff_n % (window_size + 1) == 0) { // the plus 1 is because that rel_diff_n begins from 1
                    // clear the baseline every W times adjustment, for numerical stability
                    rel_diff_baseline = 0.0;
                    rel_diff_n = 1;
                }
            }


        }
        i++;

    }

    //analysis
    std::cout << "statistic of all adjusted s: ";
    basic_statistic(all_adjusted_s);
    std::cout << "statistic of all slopes: ";
    basic_statistic(all_slopes);
    std::cout << "correlation of adjusted_s and all slopes: " << rc.correlation() << std::endl;

    std::cout
            << "Window size, number of increase and decrease, degree of increase and decrease, ori s, varied s are: "
            <<
            window_size << ", " << number_increase_step << ", "
            << number_decrease_step << ", " << degree_increase << ", " << degree_decrease << ", " << s
            << ", "
            << (double(number_sampled_size) / double(input_size)) << std::endl;
}


/*
 * Given an initial s, which is very small
 * coarse to fine adjusting
 */
template<typename _PopulationIterator, typename _SampleContainer,
        typename _SampleRate>
void coarse_to_fine_sample(_PopulationIterator __first, _PopulationIterator __last,
                           _SampleContainer &__out, _SampleRate s) {
    if (s < 0.0 or s > 1.0) {
        std::cout << "You specified a illegal sample rate: " << s << std::endl;
        return;
    }
    long input_size = std::distance(__first, __last);

    long i = 0;
    __out.emplace_back(*(__first + i));    // always select the first element

    int fix_step = ceil(1 / s);
    auto last_x = (*(__first)).first;
    auto cur_x = (*(__first)).first;

    long course_sampled_size = 0;
    long fined_sampled_size = 0;

    double last_delta_x = 0.0;
    double cur_delta_x;
    while (i < (input_size - fix_step)) {
        i += fix_step;
        cur_x = (*(__first + i)).first;
        cur_delta_x = cur_x - last_x;
        __out.emplace_back(*(__first + i));
        course_sampled_size++;

        double rel_diff = std::abs(cur_delta_x - last_delta_x) / last_delta_x;
        if (rel_diff < 0.3) {
            __out.emplace_back(*(__first + i - round(double(fix_step) / 2)));
            fined_sampled_size++;
        }

        last_x = cur_x;
        last_delta_x = cur_delta_x;
    }
    std::cout << "Course sampled size, fined sampled size, ori s, varied s are: " << course_sampled_size << ", "
              << fined_sampled_size << ", " << s << ", "
              << (double(course_sampled_size + fined_sampled_size) / double(input_size)) << std::endl;
}


template<typename _PopulationIterator, typename _SampleContainer,
        typename _SampleRate, typename _UniformRandomNumberGenerator>
void rand_first_fix_step_sample(_PopulationIterator __first, _PopulationIterator __last,
                                _SampleContainer &__out, _SampleRate s,
                                _UniformRandomNumberGenerator &&__g) {
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    if (s < 0.0 or s > 1.0) {
        std::cout << "You specified a illegal sample rate: " << s << std::endl;
        return;
    }
    double rand_number;
    long input_size = std::distance(__first, __last);
    long i = 0;
    while (i < input_size) {
        rand_number = distribution(__g);
        if (rand_number <= s) {
            __out.emplace_back(*(__first + i));
            break; // randomly select first
        }
        i++;
    }
    int fix_step = ceil(1 / s);
    while (i < (input_size - fix_step)) {
        i += fix_step;
        __out.emplace_back(*(__first + i));
    }
    return;
}

template<typename _PopulationIterator, typename _SampleContainer,
        typename _SampleRate>
std::vector<unsigned int> *fix_first_fix_step_sample(_PopulationIterator __first, _PopulationIterator __last,
                                                     _SampleContainer &__out, _SampleRate s,
                                                     int first_at_M = 0) {
    if (s < 0.0 or s > 1.0) {
        std::cout << "You specified a illegal sample rate: " << s << std::endl;
        return nullptr;
    }

    //analysis
    auto cur_x = (*(__first)).first;
    auto last_x = (*(__first)).first;
    auto cur_y = (*(__first)).second;
    auto last_y = (*(__first)).second;
    double cur_delta_x;
    long cur_delta_y;
    double cur_slope;
    std::vector<double> all_slopes;

    long input_size = std::distance(__first, __last);

    auto sampled_pos = new std::vector<unsigned int>();
    sampled_pos->reserve(input_size * s);

    long i = 0;
    __out.emplace_back(*(__first + first_at_M));
    sampled_pos->emplace_back(i);
    i += first_at_M;

    int fix_step = ceil(1 / s);
    while (i < (input_size - fix_step)) {
        i += fix_step;
        sampled_pos->emplace_back(i);
        __out.emplace_back(*(__first + i));

        cur_x = (*(__first + i)).first;
        cur_y = (*(__first + i)).second;
        cur_delta_x = cur_x - last_x;
        cur_delta_y = cur_y - last_y;
        cur_slope = double(cur_delta_y) / cur_delta_x;
        all_slopes.emplace_back(cur_slope);

        last_x = cur_x;
        last_y = cur_y;
    }

    //analysis
    std::cout << "statistic of all slopes: ";
    basic_statistic(all_slopes);

    return sampled_pos;
}


template<typename _PopulationIterator, typename _SampleContainer,
        typename _SampleRate, typename _UniformRandomNumberGenerator>
void fix_first_at_k_fix_step_sample(_PopulationIterator __first, _PopulationIterator __last,
                                    _SampleContainer &__out, _SampleRate s,
                                    _UniformRandomNumberGenerator &&__g, long k) {
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    if (s < 0.0 or s > 1.0) {
        std::cout << "You specified a illegal sample rate: " << s << std::endl;
        return;
    }
    long input_size = std::distance(__first, __last);
    long i = k;  // always select the first element at position k
    __out.emplace_back(*(__first + i));
    int fix_step = ceil(1 / s);
    while (i < (input_size - fix_step)) {
        i += fix_step;
        __out.emplace_back(*(__first + i));
    }
    return;
}


template<typename _PopulationIterator, typename _SampleContainer,
        typename _SampleRate, typename _UniformRandomNumberGenerator>
void random_truncated_sample(_PopulationIterator __first, _PopulationIterator __last,
                             _SampleContainer &__out, _SampleRate s,
                             _UniformRandomNumberGenerator &&__g) {
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    if (s < 0.0 or s > 1.0) {
        std::cout << "You specified a illegal sample rate: " << s << std::endl;
        return;
    }
    double rand_number;
    long input_size = std::distance(__first, __last);
    long i = 0, selected_last = 0;
    long truncted_step = 3 / s;
    while (i < input_size) {
        rand_number = distribution(__g);
        if (rand_number <= s) {
            if (i - selected_last > truncted_step) {
                i = selected_last + truncted_step;
                if (i > (input_size - 1)) { return; }
            }
            __out.emplace_back(*(__first + i));
            selected_last = i;
        }
        i++;
    }
    return;
}

#include <string>
#include <sstream>
#include <vector>

std::vector<std::string> str_split(const std::string &s, char delim) {
    std::stringstream ss(s);
    std::string item;
    std::vector<std::string> elems;
    while (std::getline(ss, item, delim)) {
        //elems.push_back(item);
        elems.push_back(std::move(item)); // if C++11
    }
    return elems;
}


template<typename _PopulationIterator, typename _SampleContainer,
        typename _SampleRate, typename _UniformRandomNumberGenerator>
std::vector<unsigned int> *sample_on_strategy(_PopulationIterator __first, _PopulationIterator __last,
                                              _SampleContainer &__out, _SampleRate s,
                                              _UniformRandomNumberGenerator &&__g, std::string strategy = "random") {
    std::vector<std::string> splited_strategy; // some strategy may include parameters
    splited_strategy = str_split(strategy, ':');

    std::vector<unsigned int> *sampled_pos = nullptr;
    if (splited_strategy[0] == "random") {
        sampled_pos = random_sample(__first, __last, __out, s, __g);
    } else if (splited_strategy[0] == "rand_first_fix_step") {
        rand_first_fix_step_sample(__first, __last, __out, s, __g);
    } else if (splited_strategy[0] == "fix_first_fix_step") {
        if (splited_strategy.size() == 2) {
            fix_first_fix_step_sample(__first, __last, __out, s, std::stoi(splited_strategy[1]));
        } else {
            sampled_pos = fix_first_fix_step_sample(__first, __last, __out, s);
        }
    } else if (splited_strategy[0] == "random_truncated") {
        random_truncated_sample(__first, __last, __out, s, __g);
    } else if (splited_strategy[0] == "adaptive_k_step") { //deprecated adaptive implementation
        coarse_to_fine_sample(__first, __last, __out, s);
    } else if (splited_strategy[0] == "sliding_window") { // "sliding_window:5
        if (splited_strategy.size() == 2) {
            sliding_window_by_step(__first, __last, __out, s, std::stoi(splited_strategy[1]));
        } else if (splited_strategy.size() == 3) { // "sliding_window:5:fix
            if (splited_strategy[2] == "fix") {
                sliding_window_by_s_fix(__first, __last, __out, s, std::stoi(splited_strategy[1]));
            } else if (splited_strategy[2] == "random") {
                sliding_window_by_s_random(__first, __last, __out, s, __g, std::stoi(splited_strategy[1]));
            } else {
                throw std::runtime_error("Un-supported adaptive sample setting: " + strategy);
            }
        } else {
            throw std::runtime_error("Un-supported adaptive sample setting: " + strategy +
                                     "\nYou should specify as sliding_window:5 or sliding_window:5:fix");
        }
    } else if (splited_strategy[0] == "fix_first_rand_step") {
        fix_first_rand_step_sample(__first, __last, __out, s, __g, std::stoi(splited_strategy[1]));
    } else {
        throw std::runtime_error("Un-supported sample strategy: " + strategy);
    }
    return sampled_pos;
}


#endif //LEARNED_INDEX_SAMPLE_ALGO_HPP
