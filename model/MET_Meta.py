import numpy as np
from tqdm import tqdm
import scipy
from scipy.optimize import curve_fit


class MET_Meta:
    """
    Variables
    ---------
    epsilon_low: minimal max-error-bound
    epsilon_high: maximal max-error-bound
    rescale: hyper-parameter to re-scale the two terms
    seg_num: the number of segments
    mae: mean absolute error over the entire dataset
    seg_mu: mean of the gaps of data coverd by each segment
    seg_sigma: std of the gaps of data coverd by each segment
    seg_len: the number of data covered by each segment
    seg_mae: mean absolute error over each segment
    seg_epsilon: epsilon of each segment
    """

    def __init__(self, expected_epsilon, lr=1e-8, low=1, high=1000, init_epsilon=[50, 100, 150], withBound=True):
        self.epsilon_low = low
        self.epsilon_high = high
        self.init_epsilon = init_epsilon
        self.seg_num = 0
        self.mae = 0
        self.seg_mu = []
        self.seg_sigma = []
        self.seg_len = []
        self.seg_mae = []
        self.seg_err = []
        self.seg_epsilon = []
        self.segments = []
        self.err_all = []  # for statistic, the shape is the same as the len(data)
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
        # print(self.expected_seg_err, self.w1, self.w2, self.w3, mu, sigma)
        if epsilon < self.epsilon_low:
            epsilon = self.epsilon_low
        elif epsilon > self.epsilon_high:
            epsilon = self.epsilon_high
        return epsilon

    def meta_init(self, data, gaps):
        data_len = len(data)
        seg_start = 0
        mu = gaps[0:100].mean()
        cur_err = 0
        k = 5
        self.init_epsilon = list(self.init_epsilon)
        self.init_epsilon.extend([self.expected_epsilon for i in range(k)])
        for epsilon in self.init_epsilon:
            for i in range(seg_start, data_len):
                x = data[i] - data[seg_start]
                y = i - seg_start
                if (x / mu > (y + epsilon) or x / mu < (y - epsilon)):
                    self.seg_mu.append(gaps[seg_start:i].mean())
                    self.seg_sigma.append(gaps[seg_start:i].std())
                    self.seg_epsilon.append(epsilon)
                    self.seg_err.append(cur_err)
                    self.seg_len.append(y)
                    cur_err = 0
                    seg_start = i
                    mu = gaps[i:i + 100].mean()
                    break
                else:
                    res = abs(int(x / mu - y))
                    cur_err += res
                    self.err_all.append(res)
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

            valid = np.array(self.seg_sigma[-20:]) != 0
            # self.mean_data_feature = (self.mean_data_feature * (len(self.seg_sigma) - 1) + self.seg_mu[-1]/self.seg_sigma[-1]) / len(self.seg_sigma)
            self.mean_data_feature = (np.array(self.seg_mu[-20:])[valid] / np.array(self.seg_sigma[-20:])[valid]).mean()
            self.expected_seg_err = self.w1 * self.mean_data_feature ** self.w2 * self.expected_epsilon ** self.w3

    def learn_index_lookahead(self, data, lookn=0.4):
        self.seg_num = 0
        self.mae = 0
        self.seg_mu = []
        self.seg_sigma = []
        self.seg_len = []
        self.seg_mae = []
        self.seg_epsilon = []
        data_len = len(data)
        left_len = data_len
        gaps = np.diff(data)

        self.meta_init(data, gaps)
        look_mu = gaps[0:int(lookn * self.mean_len) + 1].mean()
        look_sigma = gaps[0:int(lookn * self.mean_len) + 1].std()
        epsilon = self.choose_epsilon(look_mu, look_sigma)
        cur_err = 0
        seg_start = 0
        mu = look_mu
        gaps = np.append(gaps, gaps.mean())

        for i in tqdm(range(data_len)):
            x = data[i] - data[seg_start]
            y = i - seg_start
            if (x / mu > (y + epsilon) or x / mu < (y - epsilon)):
                self.seg_mu.append(gaps[seg_start:i].mean())
                self.seg_sigma.append(gaps[seg_start:i].std())
                self.seg_num += 1
                self.seg_epsilon.append(epsilon)
                self.seg_err.append(cur_err)
                self.seg_len.append(y)
                left_len = data_len - i
                seg_start = i
                self.meta_learn()
                self.mean_len = (self.mean_len * (self.seg_num - 1) + self.seg_len[-1]) / self.seg_num
                look_list = gaps[seg_start:seg_start + int(lookn * self.mean_len) + 1]
                look_mu = look_list.mean()
                look_sigma = look_list.std()
                epsilon = self.choose_epsilon(look_mu, look_sigma)
                cur_err = 0
                mu = look_mu
            else:
                res = abs(int(x / mu - y))
                cur_err += res
                self.err_all.append(res)
                self.all_err += res

        # last seg       
        self.mae = self.all_err / data_len
        self.seg_num += 1
        self.seg_epsilon.append(epsilon)
        self.seg_err.append(cur_err)
        self.seg_mu.append(gaps[seg_start:].mean())
        self.seg_sigma.append(gaps[seg_start:].std())
        print(self.seg_num, self.mae)

        return
