# The original PGM class is implemented according to the official C++ codes released by the authors of PGM
# reference: https://github.com/gvinciguerra/PGM-index

import numpy as np
from tqdm import tqdm
from scipy.optimize import curve_fit
import scipy

USER_DEFINED_INIT = False  # flag indicating whether we use epsilon_list that is pre-defined by users
EXPLORE_EPOCH = 5


def calc_pgm_seg_err(data, seg, seg_start_y):
    def predict(x, seg):
        res = int(seg[2] * (x - seg[0]) + seg[1])
        if res < 0:
            return 0
        else:
            return res

    data_len = len(data)
    all_predicted_pos = np.array([predict(x, seg) for x in data])
    all_true_pos = np.arange(seg_start_y, data_len + seg_start_y)
    err = np.abs(all_predicted_pos - all_true_pos)

    return np.sum(err)


class Point:
    def __init__(self):
        self.x = 0
        self.y = 0

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __sub__(self, other):
        return Point(self.x - other.x, self.y - other.y)

    def __lt__(self, other):
        return self.y * other.x < self.x * other.y

    def __gt__(self, other):
        return self.y * other.x > self.x * other.y

    def __eq__(self, other):
        return self.y * other.x == self.x * other.y


class OptimalPiecewiseLinearModel:

    def __init__(self, error_fwd, error_bwd):
        self.error_fwd = error_fwd
        self.error_bwd = error_bwd
        self.upper = []
        self.lower = []
        self.lower_start = 0
        self.upper_strat = 0
        self.points_in_hull = 0
        self.rectangle = [Point(0, 0)] * 4

    def cross(self, O, A, B):
        return (A.x - O.x) * (B.y - O.y) - (A.y - O.y) * (B.x - O.x)

    def add_point(self, x, y):
        if x < self.rectangle[2].x or x < self.rectangle[3].x:
            raise Exception("Points must be increasing by x.")

        xx = x
        yy = y

        if self.points_in_hull == 0:
            self.rectangle[0] = Point(xx, yy + self.error_fwd)
            self.rectangle[1] = Point(xx, yy - self.error_bwd)
            self.points_in_hull += 1
            return True

        if self.points_in_hull == 1:
            self.rectangle[2] = Point(xx, yy - self.error_bwd)
            self.rectangle[3] = Point(xx, yy + self.error_fwd)
            self.upper = []
            self.upper.append(self.rectangle[0])
            self.upper.append(self.rectangle[3])
            self.lower = []
            self.lower.append(self.rectangle[1])
            self.lower.append(self.rectangle[2])
            self.lower_start = self.upper_strat = 0
            self.points_in_hull += 1
            return True

        p1 = Point(xx, yy + self.error_fwd)
        p2 = Point(xx, yy - self.error_bwd)
        slope1 = self.rectangle[2] - self.rectangle[0]
        slope2 = self.rectangle[3] - self.rectangle[1]
        outside_line1 = (p1 - self.rectangle[2]) < slope1
        outside_line2 = (p2 - self.rectangle[3]) > slope2

        if outside_line1 or outside_line2:
            self.points_in_hull = 0
            return False

        if p1 - self.rectangle[1] < slope2:
            # Find extreme slope
            min = self.lower[self.lower_start] - p1
            min_i = self.lower_start
            for i in range(min_i + 1, len(self.lower)):
                val = self.lower[i] - p1
                if val > min:
                    break
                else:
                    min = val
                    min_i = i

            self.rectangle[1] = self.lower[min_i]
            self.rectangle[3] = p1
            self.lower_start = min_i

            # Hull update
            end = len(self.upper)
            while (end >= self.upper_strat + 2) and (self.cross(self.upper[end - 2], self.upper[end - 1], p1) <= 0):
                end -= 1
            self.upper = self.upper[:end + 1]
            self.upper.append(p1)

        if p2 - self.rectangle[0] > slope1:
            # Find extreme slope
            max = self.upper[self.upper_strat] - p2
            max_i = self.upper_strat
            for i in range(max_i + 1, len(self.upper)):
                val = self.upper[i] - p2
                if val < max:
                    break
                else:
                    max = val
                    max_i = i

            self.rectangle[0] = self.upper[max_i]
            self.rectangle[2] = p2
            self.upper_strat = max_i

            # Hull update
            end = len(self.lower)
            while (end >= self.lower_start + 2) and (self.cross(self.lower[end - 2], self.lower[end - 1], p2) >= 0):
                end -= 1
            self.lower = self.lower[:end + 1]
            self.lower.append(p2)

        self.points_in_hull += 1
        return True

    def get_intersection(self):
        slope1 = self.rectangle[2] - self.rectangle[0]
        slope2 = self.rectangle[3] - self.rectangle[1]

        if self.points_in_hull == 1 or slope1 == slope2:
            return self.rectangle[0].x, self.rectangle[0].y

        a = slope1.x * slope2.y - slope1.y * slope2.x
        b = ((self.rectangle[1].x - self.rectangle[0].x) * (self.rectangle[3].y - self.rectangle[1].y) -
             (self.rectangle[1].y - self.rectangle[0].y) * (self.rectangle[3].x - self.rectangle[1].x)) / a
        i_x = self.rectangle[0].x + b * slope1.x
        i_y = self.rectangle[0].y + b * slope1.y
        return i_x, i_y

    def get_slope_range(self):
        if self.points_in_hull == 1:
            return 0, 1

        min_slope = (self.rectangle[2].y - self.rectangle[0].y) / (self.rectangle[2].x - self.rectangle[0].x)
        max_slope = (self.rectangle[3].y - self.rectangle[1].y) / (self.rectangle[3].x - self.rectangle[1].x)
        return min_slope, max_slope

    def get_intercept(self, key):
        i_x, i_y = self.get_intersection()
        min_slope, max_slope = self.get_slope_range()
        slope = 0.5 * (min_slope + max_slope)
        return i_y - (i_x - key) * slope


class Pgm:
    def __init__(self, init_epsilon):
        self.epsilon = init_epsilon
        self.segments = []
        self.seg_num = 0
        self.mae = 0
        self.mechanism = OptimalPiecewiseLinearModel(self.epsilon, self.epsilon)
        # self.seg_err = []
        # self.seg_mu = []

    def learn_index(self, data):
        data_len = len(data)
        seg_start_y = 0
        seg_start = data[seg_start_y]
        err_sum = 0
        gaps = np.diff(data)

        for cur_i in tqdm(range(data_len)):
            i = cur_i
            add_success = False
            while not add_success:
                if i > 0 and data[i] == data[i - 1]:  # skip duplicated keys
                    add_success = True
                    continue

                add_success = self.mechanism.add_point(data[i], i)
                if not add_success:
                    intercept = self.mechanism.get_intercept(seg_start)
                    slope_low, slope_high = self.mechanism.get_slope_range()
                    slope = 0.5 * (slope_low + slope_high)
                    # segment format: [begin_key(x0), y0, slope], prediction can be (x-x0)*slope+y0
                    seg = [seg_start, intercept, slope]
                    self.segments.append(seg)
                    self.seg_num += 1
                    # self.seg_mu.append(gaps[seg_start_y:i - 1].mean())
                    cur_seg_err = calc_pgm_seg_err(data[seg_start_y:i], seg, seg_start_y)
                    # self.seg_err.append(cur_seg_err)
                    err_sum += cur_seg_err

                    seg_start_y = i
                    seg_start = data[seg_start_y]
                    i -= 1

        # last seg
        intercept = self.mechanism.get_intercept(seg_start)
        slope_low, slope_high = self.mechanism.get_slope_range()
        slope = 0.5 * (slope_low + slope_high)
        seg = [seg_start, intercept, slope]
        self.segments.append(seg)
        self.seg_num += 1
        # self.seg_mu.append(gaps[seg_start_y:i - 1].mean())
        cur_seg_err = calc_pgm_seg_err(data[seg_start_y:i], seg, seg_start_y)
        # self.seg_err.append(cur_seg_err)
        err_sum += cur_seg_err
        self.mae = err_sum / data_len
        print(self.epsilon, self.seg_num, self.mae)
        return


class PgmDynamic:
    def __init__(self, expected_epsilon, lr=1e-8, low=1, high=1000, init_epsilon=[50, 100, 150], withBound=True):
        self.epsilon_low = low
        self.epsilon_high = high
        if USER_DEFINED_INIT:
            self.init_epsilon = init_epsilon
        else:
            self.init_epsilon = [0.25 * expected_epsilon, 0.5 * expected_epsilon, expected_epsilon,
                                 2 * expected_epsilon, 4 * expected_epsilon] * EXPLORE_EPOCH
        self.segments = []
        self.seg_num = 0
        self.mae = 0
        self.seg_epsilon = []
        self.w1 = 0
        self.w2 = 0
        self.w3 = 0
        self.all_err = 0
        self.withBound = withBound
        self.lr = lr
        self.expected_epsilon = expected_epsilon
        self.expected_seg_err = 0
        self.mean_data_feature = 0
        self.mechanism = OptimalPiecewiseLinearModel(self.expected_epsilon, self.expected_epsilon)
        self.mean_len = 0

    def choose_epsilon(self, mu, sigma):
        if sigma == 0:
            return self.epsilon_low
        epsilon = int((self.expected_seg_err / (self.w1 * (mu / sigma) ** self.w2)) ** (1 / self.w3))
        if epsilon < self.epsilon_low:
            epsilon = self.epsilon_low
        elif epsilon > self.epsilon_high:
            epsilon = self.epsilon_high
        return epsilon

    def learner_init(self, data, gaps):
        init_seg_len = []
        init_seg_err = []
        init_seg_mu = []
        init_seg_epsilon = []
        init_seg_sigma = []
        seg_start_y = 0
        seg_start = data[seg_start_y]
        data_len = len(data)
        init_seg = 0
        if USER_DEFINED_INIT:
            self.init_epsilon = list(self.init_epsilon)
            # ensure that the expected_epsilon is tried EPOCH times
            self.init_epsilon.extend([self.expected_epsilon for i in range(EXPLORE_EPOCH)])
        self.mechanism.error_bwd = self.mechanism.error_fwd = self.init_epsilon[init_seg]
        init_seg += 1
        for cur_i in range(data_len):
            i = cur_i
            add_success = False
            stop_init = False
            while not add_success:
                if i > 0 and data[i] == data[i - 1]:  # skip duplicated keys
                    add_success = True
                    continue

                add_success = self.mechanism.add_point(data[i], i)
                if not add_success:
                    intercept = self.mechanism.get_intercept(seg_start)
                    slope_low, slope_high = self.mechanism.get_slope_range()
                    slope = 0.5 * (slope_low + slope_high)
                    # segment format: [begin_key(x0), y0, slope], prediction can be (x-x0)*slope+y0
                    seg = [seg_start, intercept, slope]
                    init_seg_len.append(i - seg_start_y)
                    init_seg_mu.append(gaps[seg_start_y:i - 1].mean())
                    init_seg_sigma.append(gaps[seg_start_y:i - 1].std())
                    init_seg_err.append(calc_pgm_seg_err(data[seg_start_y:i], seg, seg_start_y))
                    init_seg_epsilon.append(self.mechanism.error_bwd)
                    seg_start_y = i
                    seg_start = data[seg_start_y]
                    i -= 1
                    if init_seg < len(self.init_epsilon):
                        self.mechanism.error_bwd = self.mechanism.error_fwd = self.init_epsilon[init_seg]
                        init_seg += 1
                    else:
                        stop_init = True
                        break
            if stop_init:
                break

        # self.mean_len = np.mean(init_seg_len[-EXPLORE_EPOCH:])

        def func(X, w1, w2, w3):
            return w1 * X[0] ** w2 * X[1] ** w3

        valid = np.array(init_seg_sigma) * np.array(init_seg_err) != 0
        x1 = np.array(init_seg_mu)[valid] / np.array(init_seg_sigma)[valid]
        x2 = np.array(init_seg_epsilon)[valid]
        Y = np.array(init_seg_err)[valid]
        X = scipy.array([x1, x2])
        if self.withBound:
            popt, _ = curve_fit(func, X, Y, bounds=([0.564, 1, 2], [0.78, 2, 3]))
        else:
            popt, _ = curve_fit(func, X, Y)
        self.w1, self.w2, self.w3 = popt

        self.mean_data_feature = x1.mean()
        expected_seg_error = self.w1 * self.mean_data_feature ** self.w2 * self.expected_epsilon ** self.w3
        self.expected_seg_err = expected_seg_error
        return

    def learner_update(self, seg_mu, seg_sigma, seg_epsilon, seg_err):
        if seg_sigma == 0:
            return
        w1 = self.w1
        w2 = self.w2
        w3 = self.w3
        x1 = seg_mu / seg_sigma
        x2 = seg_epsilon
        y = seg_err
        w1_delta = self.lr * 2 * (w1 * x1 ** w2 * x2 ** w3 - y) * x1 ** w2 * x2 ** w3
        w2_delta = self.lr * 2 * (w1 * x1 ** w2 * x2 ** w3 - y) * w1 * x1 ** w2 * x2 ** w3 * np.log(x1)
        w3_delta = self.lr * 2 * (w1 * x1 ** w2 * x2 ** w3 - y) * w1 * x1 ** w2 * x2 ** w3 * np.log(x2)
        delta_norm = np.linalg.norm([w1_delta, w2_delta, w3_delta])
        if delta_norm > 0.1:
            w1_delta = w1_delta / delta_norm / 10
            w2_delta = w2_delta / delta_norm / 10
            w3_delta = w3_delta / delta_norm / 10
        self.w1 = w1 - w1_delta
        self.w2 = w2 - w2_delta
        self.w3 = w3 - w3_delta
        if self.withBound:
            self.w1 = np.clip(self.w1, 0.564, 0.78)
            self.w2 = np.clip(self.w2, 1, 2)
            self.w3 = np.clip(self.w3, 2, 3)
        self.mean_data_feature = (self.mean_data_feature * (self.seg_num - 1) + seg_mu / seg_sigma) / self.seg_num
        self.expected_seg_err = self.w1 * self.mean_data_feature ** self.w2 * self.expected_epsilon ** self.w3

    def learn_index_lookahead(self, data, lookn=0.4):
        data_len = len(data)
        seg_start_y = 0
        seg_start = data[seg_start_y]
        err_sum = 0
        gaps = np.diff(data)

        self.learner_init(data, gaps)
        look_mu = gaps[0:int(lookn * self.mean_len)].mean()
        look_sigma = gaps[0:int(lookn * self.mean_len)].std()
        first_epsilon = self.choose_epsilon(look_mu, look_sigma)
        self.mechanism = OptimalPiecewiseLinearModel(first_epsilon, first_epsilon)
        for cur_i in tqdm(range(data_len)):
            i = cur_i
            add_success = False
            while not add_success:
                if i > 0 and data[i] == data[i - 1]:  # skip duplicated keys
                    add_success = True
                    continue

                add_success = self.mechanism.add_point(data[i], i)
                if not add_success:
                    intercept = self.mechanism.get_intercept(seg_start)
                    slope_low, slope_high = self.mechanism.get_slope_range()
                    slope = 0.5 * (slope_low + slope_high)
                    # segment format: [begin_key(x0), y0, slope], prediction can be (x-x0)*slope+y0
                    seg = [seg_start, intercept, slope]
                    self.segments.append(seg)
                    self.seg_num += 1
                    self.seg_epsilon.append(self.mechanism.error_bwd)
                    seg_len = i - seg_start_y
                    seg_mu = gaps[seg_start_y:i - 1].mean()
                    seg_sigma = gaps[seg_start_y:i - 1].std()
                    seg_err = calc_pgm_seg_err(data[seg_start_y:i], seg, seg_start_y)
                    err_sum += seg_err
                    self.learner_update(seg_mu, seg_sigma, self.mechanism.error_bwd, seg_err)
                    self.mean_len = (self.mean_len * (self.seg_num - 1) + seg_len) / self.seg_num
                    look_list = gaps[cur_i:cur_i + int(lookn * self.mean_len)]
                    look_mu = look_list.mean()
                    look_sigma = look_list.std()
                    self.mechanism.error_bwd = self.mechanism.error_fwd = self.choose_epsilon(look_mu, look_sigma)

                    seg_start_y = i
                    seg_start = data[seg_start_y]
                    i -= 1

        # last seg
        intercept = self.mechanism.get_intercept(seg_start)
        slope_low, slope_high = self.mechanism.get_slope_range()
        slope = 0.5 * (slope_low + slope_high)
        seg = [seg_start, intercept, slope]
        self.segments.append(seg)
        self.seg_num += 1
        self.seg_epsilon.append(self.mechanism.error_bwd)
        seg_err = calc_pgm_seg_err(data[seg_start_y:i], seg, seg_start_y)
        err_sum += seg_err
        self.mae = err_sum / data_len
        print(self.expected_epsilon, self.seg_num, self.mae)
        return
