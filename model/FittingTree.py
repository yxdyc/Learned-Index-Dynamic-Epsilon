import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import struct
import sys
from scipy.optimize import curve_fit


class FT:

    def __init__(self, init_epsilon):
        self.epsilon = init_epsilon
        self.segments = []
        self.seg_num = 0
        self.seg_mu = []
        self.seg_sigma = []
        self.mae = 0
        self.epsilons = []
        self.seg_nums = []
        self.seg_err = []
        self.seg_len = []
        self.err_all = np.array([])  # for statistic, the shape is the same as the len(data)

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


    def learn_index(self, data):
        segments = []
        seg_num = 0
        mae = 0
        epsilon = self.epsilon
        data_len = len(data)
        segments = []
        seg_start = 0
        slope_high = sys.float_info.max
        slope_low = 0
        gaps = []
        err_sum = 0
        gaps = np.diff(data)
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
                self.seg_len.append(i - seg_start)
                self.seg_mu.append(gaps[seg_start:i - 1].mean())
                self.seg_sigma.append(gaps[seg_start:i - 1].std())
                seg_err, errors = self.calc_seg_err(data[seg_start:i], [data[seg_start], seg_start, 0.5 * (slope_high + slope_low)])
                self.seg_err.append(seg_err)
                self.err_all = np.append(self.err_all, errors)  # for statistic: to check the std and P99 of seg error

                err_sum += self.seg_err[-1]
                seg_end = i if i == 0 else i - 1
                segments.append([data[seg_start], seg_start, 0.5 * (slope_high + slope_low)])
                slope_high = sys.float_info.max
                slope_low = 0
                seg_start = i
                if i == data_len - 1:
                    slope_high = 0
        # last seg
        segments.append([data[seg_start], seg_start, 0.5 * (slope_high + slope_low)])
        self.segments = segments
        self.seg_num += 1
        self.seg_len.append(data_len - seg_start)
        self.seg_mu.append(gaps[seg_start:].mean())
        self.seg_sigma.append(gaps[seg_start:].std())
        seg_err, errors = self.calc_seg_err(data[seg_start:i], [data[seg_start], seg_start, 0.5 * (slope_high + slope_low)])
        self.seg_err.append(seg_err)
        self.err_all = np.append(self.err_all, errors)  # for statistic: to check the std and P99 of seg error

        err_sum += self.seg_err[-1]
        self.mae = err_sum / data_len
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
