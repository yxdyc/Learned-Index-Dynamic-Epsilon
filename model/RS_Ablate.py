import numpy as np
import scipy
from scipy.optimize import curve_fit

from model.RS_Meta import RS_Meta, compute_orientation, calc_seg_err


class RS_Random(RS_Meta):
    def choose_epsilon(self, mu, sigma):
        epsilon = np.random.randint(self.epsilon_low, self.epsilon_high)
        return epsilon

    def meta_learn(self):
        pass


class RS_LS(RS_Meta):
    def meta_learn(self):
        def func(X, w1, w2, w3):
            return w1 * X[0] ** w2 * X[1] ** w3

        valid = np.array(self.seg_sigma) * np.array(self.seg_err) != 0
        x1 = np.array(self.seg_mu)[valid] / np.array(self.seg_sigma)[valid]
        x2 = np.array(self.seg_epsilon)[valid]
        Y = np.array(self.seg_err)[valid]
        X = scipy.array([x1, x2])
        if self.withBound:
            popt, _ = curve_fit(func, X, Y, bounds=([0.56, 1, 2], [0.78, 2, 3]))
        else:
            popt, _ = curve_fit(func, X, Y)
        self.w1, self.w2, self.w3 = popt
        self.w1s.append(self.w1)
        self.w2s.append(self.w2)
        self.w3s.append(self.w3)
        return


class RS_Poly(RS_Meta):
    def choose_epsilon(self, mu, sigma):
        epsilon = int((self.expected_seg_err / self.w1) ** (1 / self.w3))
        if epsilon < self.epsilon_low:
            epsilon = self.epsilon_low
        elif epsilon > self.epsilon_high:
            epsilon = self.epsilon_high
        return epsilon

    def meta_init(self, data, gaps):
        data_len = len(data)
        seg_start = 0
        init_seg = 0
        k = 5
        self.init_epsilon = list(self.init_epsilon)
        self.init_epsilon.extend([self.expected_epsilon for i in range(k)])
        self.epsilon = self.init_epsilon[init_seg]
        init_seg += 1
        for cur_i in range(data_len):
            if self.curr_max_position_ == 0:
                self.spline_points.append((data[cur_i], cur_i))
                self.prev_point_ = (data[cur_i], cur_i)
                self.curr_max_position_ += 1
                continue

            if data[cur_i] == self.prev_point_[0]:
                self.curr_max_position_ += 1
                continue

            if self.curr_max_position_ == 1:
                upper_node = (data[cur_i], cur_i + self.epsilon)
                lower_node = (data[cur_i], 0 if cur_i < self.epsilon else cur_i - self.epsilon)
                self.prev_point_ = (data[cur_i], cur_i)
                self.curr_max_position_ += 1
                continue

            last_spline_point = self.spline_points[-1]
            upper_y = self.epsilon + cur_i
            lower_y = 0 if cur_i < self.epsilon else cur_i - self.epsilon

            upper_limit_x_diff = upper_node[0] - last_spline_point[0]
            lower_limit_x_diff = lower_node[0] - last_spline_point[0]
            x_diff = data[cur_i] - last_spline_point[0]
            upper_limit_y_diff = upper_node[1] - last_spline_point[1]
            lower_limit_y_diff = lower_node[1] - last_spline_point[1]
            y_diff = cur_i - last_spline_point[1]

            if compute_orientation(upper_limit_x_diff, upper_limit_y_diff, x_diff, y_diff) != "CW" or \
                    compute_orientation(lower_limit_x_diff, lower_limit_y_diff, x_diff, y_diff) != "CCW":
                self.spline_points.append((self.prev_point_[0], self.prev_point_[1]))
                upper_node = (data[cur_i], upper_y)
                lower_node = (data[cur_i], lower_y)

                # update seg and statistics
                cur_spline_point = self.spline_points[-1]
                last_spline_point = self.spline_points[-2]
                intercept = seg_start_y = last_spline_point[1]
                slope = (last_spline_point[1] - cur_spline_point[1]) / (last_spline_point[0] - cur_spline_point[0])
                # segment format: [begin_key(x0), \hat{y0}, slope], prediction can be (x-x0)*slope+\hat{y0}
                seg = [last_spline_point[0], intercept, slope]
                self.seg_len.append(cur_i - seg_start_y)
                self.seg_mu.append(gaps[seg_start_y:cur_i - 1].mean())
                self.seg_sigma.append(gaps[seg_start_y:cur_i - 1].std())
                self.seg_err.append(calc_seg_err(data[seg_start_y:cur_i], seg, seg_start_y))
                self.seg_epsilon.append(self.epsilon)
                if init_seg < len(self.init_epsilon):
                    self.epsilon = self.init_epsilon[init_seg]
                    init_seg += 1
                else:
                    break
            else:
                upper_y_diff = upper_y - last_spline_point[1]
                if compute_orientation(upper_limit_x_diff, upper_limit_y_diff, x_diff, upper_y_diff) == "CW":
                    upper_node = (data[cur_i], upper_y)
                lower_y_diff = lower_y - last_spline_point[1]
                if compute_orientation(lower_limit_x_diff, lower_limit_y_diff, x_diff, lower_y_diff) == "CCW":
                    lower_node = (data[cur_i], lower_y)

            self.prev_point_ = (data[cur_i], cur_i)
            self.curr_max_position_ += 1
        self.mean_len = np.mean(self.seg_len[-k:])
        self.expected_seg_err = np.mean(self.seg_err[-k:])

        def func(X, w1, w3):
            return w1 * X ** w3

        valid = np.array(self.seg_err) != 0
        X = np.array(self.seg_epsilon)[valid]
        Y = np.array(self.seg_err)[valid]
        try:
            popt, _ = curve_fit(func, X, Y)
            self.w1, self.w3 = popt
        except RuntimeError:
            pass
        return

    def meta_learn(self):
        def func(X, w1, w3):
            return w1 * X ** w3

        valid = np.array(self.seg_err) != 0
        X = np.array(self.seg_epsilon)[valid]
        Y = np.array(self.seg_err)[valid]
        try:
            popt, _ = curve_fit(func, X, Y)
            self.w1, self.w3 = popt
        except RuntimeError:
            pass
        self.w1s.append(self.w1)
        self.w3s.append(self.w3)