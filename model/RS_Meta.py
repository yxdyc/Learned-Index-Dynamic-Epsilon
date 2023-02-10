import numpy as np
from tqdm import tqdm
from scipy.optimize import curve_fit
import scipy


def calc_seg_err(data, seg, seg_start_y):
    all_predicted_pos = np.array([int(seg[2] * (x - seg[0]) + seg[1]) for x in data])
    all_true_pos = np.arange(seg_start_y, len(data) + seg_start_y)
    err = np.abs(all_predicted_pos - all_true_pos)

    return np.sum(err), err


def compute_orientation(dx1, dy1, dx2, dy2):
    res_1 = dy1 * dx2
    res_2 = dy2 * dx1
    if res_1 > res_2:
        return "CW"
    elif res_1 < res_2:
        return "CCW"
    else:
        return "Collinear"


class RS_Meta:
    def __init__(self, expected_epsilon, lr=1e-8, low=1, high=1000, init_epsilon=[50, 100, 150], withBound=True):
        self.epsilon_low = low
        self.epsilon_high = high
        self.init_epsilon = init_epsilon
        self.segments = []
        self.seg_num = 0
        self.seg_mu = []
        self.seg_sigma = []
        self.mae = 0
        self.seg_epsilon = []
        self.seg_err = []
        self.seg_len = []
        self.curr_max_position_ = 0
        self.spline_points = []
        self.prev_point_ = None
        self.w1 = 0
        self.w2 = 0
        self.w3 = 0
        self.w1s = []
        self.w2s = []
        self.w3s = []
        self.all_err = 0
        self.withBound = withBound
        self.lr = lr
        self.w1_delta = 0
        self.w2_delta = 0
        self.w3_delta = 0
        self.expected_epsilon = expected_epsilon
        self.expected_seg_err = 0
        self.mean_data_feature = 0
        self.mean_len = 0
        self.err_all = []  # for statistic, the shape is the same as the len(data)

    def get_err_std_p99(self):
        self.all_err = np.array(self.err_all)

        mean = np.mean(self.all_err)
        self.all_err = np.sort(self.all_err)
        std = np.std(self.all_err)
        p99 = self.all_err[int(len(self.all_err) / 100 * 99)]

        log2_err = np.log2(self.all_err + 1) + 1
        std_log = np.std(log2_err)
        mean_log = np.mean(log2_err)
        p99_log = log2_err[int(len(self.all_err) / 100 * 99)]

        return std / mean, p99 / mean, std_log / mean_log, p99_log / mean_log


    def choose_epsilon(self, mu, sigma):
        if sigma == 0:
            return self.epsilon_low
        epsilon = int((self.expected_seg_err / (self.w1 * (mu / sigma) ** self.w2)) ** (1 / self.w3))
        if epsilon < self.epsilon_low:
            epsilon = self.epsilon_low
        elif epsilon > self.epsilon_high:
            epsilon = self.epsilon_high
        return epsilon

    def meta_init(self, data, gaps):
        upper_node = None
        lower_node = None
        data_len = len(data)
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
                cur_seg_err, errors = calc_seg_err(data[seg_start_y:cur_i], seg, seg_start_y)
                self.seg_err.append(cur_seg_err)
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

        def func(X, w1, w2, w3):
            return w1 * X[0] ** w2 * X[1] ** w3

        valid = np.array(self.seg_sigma) * np.array(self.seg_err) != 0
        x1 = np.array(self.seg_mu)[valid] / np.array(self.seg_sigma)[valid]
        x2 = np.array(self.seg_epsilon)[valid]
        Y = np.array(self.seg_err)[valid]
        X = scipy.array([x1, x2])
        if self.withBound:
            popt, _ = curve_fit(func, X, Y, bounds=([0.564, 1, 2], [0.78, 2, 3]))
        else:
            popt, _ = curve_fit(func, X, Y)
        self.w1, self.w2, self.w3 = popt
        self.w1s.append(self.w1)
        self.w2s.append(self.w2)
        self.w3s.append(self.w3)

        self.mean_data_feature = x1.mean()
        expected_seg_error = self.w1 * self.mean_data_feature ** self.w2 * self.expected_epsilon ** self.w3
        self.expected_seg_err = expected_seg_error
        return

    def meta_learn(self):
        if self.seg_sigma[-1] == 0:
            return
        w1 = self.w1
        w2 = self.w2
        w3 = self.w3
        x1 = self.seg_mu[-1] / self.seg_sigma[-1]
        x2 = self.seg_epsilon[-1]
        y = self.seg_err[-1]
        w1_delta = self.lr * 2 * (w1 * x1 ** w2 * x2 ** w3 - y) * x1 ** w2 * x2 ** w3
        w2_delta = self.lr * 2 * (w1 * x1 ** w2 * x2 ** w3 - y) * w1 * x1 ** w2 * x2 ** w3 * np.log(x1)
        w3_delta = self.lr * 2 * (w1 * x1 ** w2 * x2 ** w3 - y) * w1 * x1 ** w2 * x2 ** w3 * np.log(x2)
        self.w1_delta += w1_delta
        self.w2_delta += w2_delta
        self.w3_delta += w3_delta
        if len(self.segments) % 1 == 0:
            self.w1_delta = self.w1_delta / 1
            self.w2_delta = self.w2_delta / 1
            self.w3_delta = self.w3_delta / 1
            delta_norm = np.linalg.norm([self.w1_delta, self.w2_delta, self.w3_delta])
            if delta_norm > 0.1:
                self.w1_delta = self.w1_delta / delta_norm / 10
                self.w2_delta = self.w2_delta / delta_norm / 10
                self.w3_delta = self.w3_delta / delta_norm / 10
            if self.w1_delta > w1:
                return
            self.w1 = w1 - self.w1_delta
            self.w2 = w2 - self.w2_delta
            self.w3 = w3 - self.w3_delta
            if self.withBound:
                self.w1 = np.clip(self.w1, 0.564, 0.78)
                self.w2 = np.clip(self.w2, 1, 2)
                self.w3 = np.clip(self.w3, 2, 3)
            self.w1s.append(self.w1)
            self.w2s.append(self.w2)
            self.w3s.append(self.w3)
            self.w1_delta = 0
            self.w2_delta = 0
            self.w3_delta = 0
            if self.seg_sigma[-1] != 0:
                self.mean_data_feature = (self.mean_data_feature * (len(self.seg_sigma) - 1) + self.seg_mu[-1] /
                                          self.seg_sigma[-1]) / len(self.seg_sigma)
            # valid = np.array(self.seg_sigma[-20:])!=0
            # self.mean_data_feature = (np.array(self.seg_mu[-20:])[valid]/np.array(self.seg_sigma[-20:])[valid]).mean()
            self.expected_seg_err = self.w1 * self.mean_data_feature ** self.w2 * self.expected_epsilon ** self.w3

    def learn_index_lookahead(self, data, lookn=0.4):
        data_len = len(data)
        err_sum = 0
        gaps = np.diff(data)

        upper_node = None
        lower_node = None

        self.meta_init(data, gaps)
        look_mu = gaps[0:int(lookn * self.mean_len)].mean()
        look_sigma = gaps[0:int(lookn * self.mean_len)].std()
        self.epsilon = self.choose_epsilon(look_mu, look_sigma)

        self.curr_max_position_ = 0
        self.spline_points = []
        self.prev_point_ = None

        for cur_i in tqdm(range(data_len)):
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
                self.segments.append(seg)
                self.seg_num += 1
                self.seg_len.append(cur_i - seg_start_y)
                self.seg_mu.append(gaps[seg_start_y:cur_i - 1].mean())
                self.seg_sigma.append(gaps[seg_start_y:cur_i - 1].std())
                cur_seg_err, errors = calc_seg_err(data[seg_start_y:cur_i], seg, seg_start_y)
                self.err_all.extend(errors.tolist())  # for statistic: to check the std and P99 of seg error
                self.seg_err.append(cur_seg_err)
                err_sum += self.seg_err[-1]
                self.seg_epsilon.append(self.epsilon)
                self.meta_learn()
                self.mean_len = (self.mean_len * (self.seg_num - 1) + self.seg_len[-1]) / self.seg_num
                look_list = gaps[cur_i:cur_i + int(lookn * self.mean_len)]
                if len(look_list) > 1:
                    look_mu = look_list.mean()
                    look_sigma = look_list.std()
                    self.epsilon = self.choose_epsilon(look_mu, look_sigma)
            else:
                upper_y_diff = upper_y - last_spline_point[1]
                if compute_orientation(upper_limit_x_diff, upper_limit_y_diff, x_diff, upper_y_diff) == "CW":
                    upper_node = (data[cur_i], upper_y)
                lower_y_diff = lower_y - last_spline_point[1]
                if compute_orientation(lower_limit_x_diff, lower_limit_y_diff, x_diff, lower_y_diff) == "CCW":
                    lower_node = (data[cur_i], lower_y)

            self.prev_point_ = (data[cur_i], cur_i)
            self.curr_max_position_ += 1

        # last seg
        self.spline_points.append((self.prev_point_[0], self.prev_point_[1]))
        cur_spline_point = self.spline_points[-1]
        last_spline_point = self.spline_points[-2]
        intercept = seg_start_y = last_spline_point[1]
        slope = (last_spline_point[1] - cur_spline_point[1]) / (last_spline_point[0] - cur_spline_point[0])
        # segment format: [begin_key(x0), \hat{y0}, slope], prediction can be (x-x0)*slope+\hat{y0}
        seg = [last_spline_point[0], intercept, slope]
        self.segments.append(seg)
        self.seg_num += 1
        self.seg_len.append(data_len - seg_start_y)
        self.seg_mu.append(gaps[seg_start_y:].mean())
        self.seg_sigma.append(gaps[seg_start_y:].std())
        cur_seg_err, errors = calc_seg_err(data[seg_start_y:], seg, seg_start_y)
        self.err_all.extend(errors.tolist())  # for statistic: to check the std and P99 of seg error
        self.seg_err.append(cur_seg_err)
        self.seg_epsilon.append(self.epsilon)
        err_sum += self.seg_err[-1]
        self.mae = err_sum / data_len
        print(self.expected_epsilon, self.seg_num, self.mae)
        return


