import numpy as np
from tqdm import tqdm
from scipy.optimize import curve_fit


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


class RadixSpline:
    def __init__(self, init_epsilon):
        self.epsilon = init_epsilon
        self.segments = []
        self.seg_num = 0
        self.seg_mu = []
        self.seg_sigma = []
        self.mae = 0
        self.seg_err = []
        self.seg_len = []
        self.err_all = []  # for statistic, the shape is the same as the len(data)

        self.curr_max_position_ = 0
        self.spline_points = []
        self.prev_point_ = None

    def get_err_std_p99(self):
        self.all_err = np.array(self.err_all) + 1e-7
        mean = np.mean(self.all_err)
        self.all_err = np.sort(self.all_err)
        std = np.std(self.all_err)
        p99 = self.all_err[int(len(self.all_err)/100 * 99)]
        mean_log = np.log2(mean)
        std_log = np.std(np.log2(self.all_err))
        p99_log = np.log2(self.all_err)[int(len(self.all_err)/100 * 99)]

        return std/mean, p99/mean, std_log/mean_log, p99_log/mean_log


    def learn_index(self, data):
        data_len = len(data)
        err_sum = 0
        gaps = np.diff(data)

        upper_node = None
        lower_node = None

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
        err_sum += self.seg_err[-1]
        self.mae = err_sum / data_len
        print(self.epsilon, self.seg_num, self.mae)
        return

    def predict_position(self, key):
        # binary search the segment responsible for the given key
        l = 0
        r = len(self.segments) - 1
        while l < r:
            m = int((l + r + 1) / 2)
            if self.segments[m][0] <= key:
                l = m
            else:
                r = m - 1
        seg = self.segments[l]
        # assert self.segments[l][0] <= key and (l == len(self.segments)-1 or self.segments[l+1][0] > key)  
        pred_pos = int(seg[2] * (key - seg[0]) + seg[1])
        return pred_pos

    def evaluate_indexer(self, data):
        data_len = len(data)
        true_pos = np.array([i for i in range(data_len)])
        pred_pos = np.array([self.predict_position(data[i]) for i in tqdm(range(data_len))])
        pos_err = np.abs(pred_pos - true_pos)
        self.mae = pos_err.mean()
        print(self.seg_num, self.mae)

