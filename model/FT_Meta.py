import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import struct
import sys
from sklearn.linear_model import LinearRegression
import scipy
from scipy.optimize import curve_fit


class FT_Meta:

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
        self.err_all = np.array([])  # for statistic, the shape is the same as the len(data)
        self.segments = []
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
        data_len = len(data)
        seg_start = 0
        k = 5
        self.init_epsilon = list(self.init_epsilon)
        self.init_epsilon.extend([self.expected_epsilon for i in range(k)])
        for epsilon in self.init_epsilon:
            slope_high = sys.float_info.max
            slope_low = 0
            for i in range(seg_start, len(data)):
                delta_y = i - seg_start
                delta_x = data[i] - data[seg_start]
                slope = 0 if delta_x == 0 else delta_y / delta_x
                if slope <= slope_high and slope >= slope_low:
                    if delta_x == 0:
                        continue
                    max_slope = (delta_y + epsilon) / delta_x
                    min_slope = ((delta_y - epsilon) / delta_x) if delta_y >= epsilon else 0
                    slope_high = min(slope_high, max_slope)
                    slope_low = max(slope_low, min_slope)
                else:
                    self.seg_mu.append(gaps[seg_start:i - 1].mean())
                    self.seg_sigma.append(gaps[seg_start:i - 1].std())
                    self.seg_len.append(i - seg_start)
                    self.seg_epsilon.append(epsilon)
                    seg_err, errors = self.calc_seg_err(data[seg_start:i],
                                                        [data[seg_start], seg_start, 0.5 * (slope_high + slope_low)])
                    self.seg_err.append(seg_err)
                    self.err_all = np.append(self.err_all,
                                             errors)  # for statistic: to check the std and P99 of seg error

                    self.seg_mae.append(self.seg_err[-1] / self.seg_len[-1])
                    seg_start = i
                    slope_high = sys.float_info.max
                    slope_low = 0
                    break
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
        '''
        valid = np.array(self.seg_sigma) * np.array(self.seg_err) != 0
        x1 = np.log(np.array(self.seg_mu)[valid] / np.array(self.seg_sigma)[valid])
        x2 = np.log(np.array(self.seg_epsilon)[valid])
        y = np.log(np.array(self.seg_err)[valid])
        x1 = x1.reshape((-1,1))
        x2 = x2.reshape((-1,1))
        X = np.concatenate((x1,x2), axis = 1)
        Y = y.reshape((-1,1))
    
        model = LinearRegression()
        model.fit(X, Y)
        self.w1 = np.e ** model.intercept_[0]
        self.w2 = model.coef_[0][0]
        self.w3 = model.coef_[0][1]
        self.w1s.append(self.w1)
        self.w2s.append(self.w2)
        self.w3s.append(self.w3)
        '''
        '''
        def func(X,w1,w2,w3):
            return w1 *  X[0] ** w2 * X[1] ** w3
        valid = np.array(self.seg_sigma) * np.array(self.seg_err) != 0
        x1 = np.array(self.seg_mu)[valid] / np.array(self.seg_sigma)[valid]
        x2 = np.array(self.seg_epsilon)[valid]
        Y = np.array(self.seg_err)[valid]
        X = scipy.array([x1,x2])
        if self.withBound:
            popt, _ = curve_fit(func,X,Y,bounds=(0,[0.53,2,3]))
        else:
            popt, _ = curve_fit(func,X,Y)
        self.w1,self.w2,self.w3 = popt
        self.w1s.append(self.w1)
        self.w2s.append(self.w2)
        self.w3s.append(self.w3)
        return
        '''
        # '''
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

    def calc_seg_err(self, data, seg):
        data_len = len(data)
        err = 0
        all_errors = []
        for i in range(data_len):
            pred_pos = int(seg[2] * (data[i] - seg[0]))
            cur_err = abs(pred_pos - i)
            err += cur_err
            all_errors.append(cur_err)
        return err, np.array(all_errors)

    def learn_index_lookahead(self, data, lookn=0.4):
        segments = []
        seg_num = 0
        mae = 0
        data_len = len(data)
        segments = []
        seg_start = 0
        slope_high = sys.float_info.max
        slope_low = 0
        err_sum = 0
        gaps = np.diff(data)
        self.meta_init(data, gaps)
        # self.meta_learn()
        look_mu = gaps[0:int(lookn * self.mean_len)].mean()
        look_sigma = gaps[0:int(lookn * self.mean_len)].std()
        epsilon = self.choose_epsilon(look_mu, look_sigma)
        for i in tqdm(range(data_len)):
            delta_y = i - seg_start
            delta_x = data[i] - data[seg_start]
            slope = 0 if delta_x == 0 else delta_y / delta_x
            if slope <= slope_high and slope >= slope_low:
                if delta_x == 0:
                    continue
                max_slope = (delta_y + epsilon) / delta_x
                min_slope = ((delta_y - epsilon) / delta_x) if delta_y >= epsilon else 0
                slope_high = min(slope_high, max_slope)
                slope_low = max(slope_low, min_slope)
            else:
                self.seg_num += 1
                self.seg_mu.append(gaps[seg_start:i - 1].mean())
                self.seg_sigma.append(gaps[seg_start:i - 1].std())
                self.seg_len.append(i - seg_start)
                self.seg_epsilon.append(epsilon)
                seg_err, errors = self.calc_seg_err(data[seg_start:i],
                                                    [data[seg_start], seg_start, 0.5 * (slope_high + slope_low)])
                self.seg_err.append(seg_err)
                self.err_all = np.append(self.err_all, errors)  # for statistic: to check the std and P99 of seg error

                self.seg_mae.append(self.seg_err[-1] / self.seg_len[-1])
                self.all_err += self.seg_err[-1]
                seg_end = i if i == 0 else i - 1
                self.segments.append([data[seg_start], seg_start, 0.5 * (slope_high + slope_low)])
                slope_high = sys.float_info.max
                slope_low = 0
                seg_start = i
                if i == data_len - 1:
                    slope_high = 0
                    break
                self.meta_learn()
                self.mean_len = (self.mean_len * (self.seg_num - 1) + self.seg_len[-1]) / self.seg_num
                look_list = gaps[seg_start:seg_start + int(lookn * self.mean_len)]
                look_mu = look_list.mean()
                look_sigma = look_list.std()
                # print(look_mu/look_sigma,gaps[seg_start:seg_start+100000].mean()/gaps[seg_start:seg_start+100000].std(),self.seg_len[-1],np.mean(self.seg_len[-5:]))
                epsilon = self.choose_epsilon(look_mu, look_sigma)
        # last seg
        self.segments.append([data[seg_start], seg_start, 0.5 * (slope_high + slope_low)])
        self.seg_num += 1
        self.seg_mu.append(gaps[seg_start:].mean())
        self.seg_sigma.append(gaps[seg_start:].std())
        self.seg_len.append(data_len - seg_start)
        self.seg_epsilon.append(epsilon)

        seg_err, errors = self.calc_seg_err(data[seg_start:i],
                                            [data[seg_start], seg_start, 0.5 * (slope_high + slope_low)])
        self.seg_err.append(seg_err)
        self.err_all = np.append(self.err_all, errors)  # for statistic: to check the std and P99 of seg error

        self.seg_mae.append(self.seg_err[-1] / self.seg_len[-1])
        self.all_err += self.seg_err[-1]
        self.mae = self.all_err / data_len
        print(self.seg_num, self.mae)
        return

    def predict_position(self, key):
        # binary search
        l = 0
        r = len(self.segments) - 1
        while l < r:
            m = int((l + r + 1) / 2)
            if (self.segments[m][0] <= key):
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


class FT_Random(FT_Meta):
    def choose_epsilon(self, mu, sigma):
        epsilon = np.random.randint(self.epsilon_low, self.epsilon_high)
        return epsilon

    def meta_learn(self):
        pass


class FT_LS(FT_Meta):
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


class FT_Poly(FT_Meta):
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
        k = 5
        self.init_epsilon = list(self.init_epsilon)
        self.init_epsilon.extend([self.expected_epsilon for i in range(k)])
        for epsilon in self.init_epsilon:
            slope_high = sys.float_info.max
            slope_low = 0
            for i in range(seg_start, len(data)):
                delta_y = i - seg_start
                delta_x = data[i] - data[seg_start]
                slope = 0 if delta_x == 0 else delta_y / delta_x
                if slope <= slope_high and slope >= slope_low:
                    if delta_x == 0:
                        continue
                    max_slope = (delta_y + epsilon) / delta_x
                    min_slope = ((delta_y - epsilon) / delta_x) if delta_y >= epsilon else 0
                    slope_high = min(slope_high, max_slope)
                    slope_low = max(slope_low, min_slope)
                else:
                    self.seg_mu.append(gaps[seg_start:i - 1].mean())
                    self.seg_sigma.append(gaps[seg_start:i - 1].std())
                    self.seg_len.append(i - seg_start)
                    self.seg_epsilon.append(epsilon)
                    seg_err, errors = self.calc_seg_err(data[seg_start:i],
                                                        [data[seg_start], seg_start, 0.5 * (slope_high + slope_low)])
                    self.seg_err.append(seg_err)
                    self.err_all = np.append(self.err_all,
                                             errors)  # for statistic: to check the std and P99 of seg error

                    self.seg_mae.append(self.seg_err[-1] / self.seg_len[-1])
                    seg_start = i
                    slope_high = sys.float_info.max
                    slope_low = 0
                    break
        self.mean_len = np.mean(self.seg_len[-k:])

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

        tmp_seg_err = []
        seg_start = 0
        epsilon = self.expected_epsilon
        for j in range(30):
            slope_high = sys.float_info.max
            slope_low = 0
            for i in range(seg_start, len(data)):
                delta_y = i - seg_start
                delta_x = data[i] - data[seg_start]
                slope = 0 if delta_x == 0 else delta_y / delta_x
                if slope <= slope_high and slope >= slope_low:
                    if delta_x == 0:
                        continue
                    max_slope = (delta_y + epsilon) / delta_x
                    min_slope = ((delta_y - epsilon) / delta_x) if delta_y >= epsilon else 0
                    slope_high = min(slope_high, max_slope)
                    slope_low = max(slope_low, min_slope)
                else:
                    seg_err, errors = self.calc_seg_err(data[seg_start:i],
                                                        [data[seg_start], seg_start, 0.5 * (slope_high + slope_low)])
                    tmp_seg_err.append(seg_err)
                    self.err_all = np.append(self.err_all,
                                             errors)  # for statistic: to check the std and P99 of seg error

                    seg_start = i
                    slope_high = sys.float_info.max
                    slope_low = 0
                    break
        self.expected_seg_err = np.mean(tmp_seg_err)
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
