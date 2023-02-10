import numpy as np
from tqdm import tqdm


class MET:
    """
    Variables
    ---------
    epsilon: max error bound
    seg_num: the number of segments
    mae: mean absolute error over the entire dataset
    seg_mu: mean of the gaps of data coverd by each segment
    seg_sigma: std of the gaps of data coverd by each segment
    seg_len: the number of data covered by each segment
    seg_mae: mean absolute error over each segment
    """

    def __init__(self, init_epsilon):
        self.epsilon = init_epsilon
        self.seg_num = 0
        self.mae = 0
        self.seg_mu = []
        self.seg_sigma = []
        self.seg_len = []
        self.seg_mae = []
        self.seg_err = []
        self.err_all = []  # for statistic, the shape is the same as the len(data)
        self.mean_len = 0

    def learn_index(self, data):
        self.seg_num = 0
        self.mae = 0
        self.seg_mu = []
        self.seg_sigma = []
        self.seg_len = []
        epsilon = self.epsilon
        data_len = len(data)
        gaps = np.diff(data)

        mu = gaps.mean()
        cur_err = 0
        seg_start = 0
        all_err = 0

        for i in tqdm(range(data_len)):
            x = data[i] - data[seg_start]
            y = i - seg_start
            if (x / mu > (y + epsilon) or x / mu < (y - epsilon)):
                self.seg_len.append(y)
                self.seg_err.append(cur_err)
                self.seg_mu.append(gaps[seg_start:i].mean())
                self.seg_sigma.append(gaps[seg_start:i].std())
                self.seg_num += 1
                seg_start = i
                cur_err = 0
            else:
                res = abs(int(x / mu - y))
                all_err += res
                cur_err += res
                self.err_all.append(res)

        # last seg
        self.mae = all_err / data_len
        self.seg_num += 1
        self.seg_len.append(data_len - seg_start)
        self.seg_err.append(cur_err)
        self.seg_mu.append(gaps[seg_start:].mean())
        self.seg_sigma.append(gaps[seg_start:].std())
        print(self.seg_num, self.mae)
        return

    def get_err_std_p99(self):
        self.all_err = np.array(self.err_all)

        mean = np.mean(self.all_err)
        self.all_err = np.sort(self.all_err)
        std = np.std(self.all_err)
        p99 = self.all_err[int(len(self.all_err)/100 * 99)]

        log2_err = np.log2(self.all_err + 1) + 1
        std_log = np.std(log2_err)
        mean_log = np.mean(log2_err)
        p99_log = log2_err[int(len(self.all_err)/100 * 99)]

        return std / mean, p99 / mean, std_log / mean_log, p99_log / mean_log

    def MET_init(self, data, gaps):
        data_len = len(data)
        seg_start = 0
        mu = gaps[0:100].mean()
        cur_err = 0
        k = 5
        self.init_epsilon = [self.epsilon for i in range(k)]
        for epsilon in self.init_epsilon:
            for i in range(seg_start, data_len):
                x = data[i] - data[seg_start]
                y = i - seg_start
                if (x / mu > (y + epsilon) or x / mu < (y - epsilon)):
                    self.seg_mu.append(gaps[seg_start:i].mean())
                    self.seg_sigma.append(gaps[seg_start:i].std())
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

    def learn_index_lookahead(self, data, lookn=0.4):
        self.seg_num = 0
        self.mae = 0
        self.seg_mu = []
        self.seg_sigma = []
        self.seg_len = []
        epsilon = self.epsilon
        data_len = len(data)
        gaps = np.diff(data)
        gaps = np.append(gaps, gaps.mean())
        self.MET_init(data, gaps)
        look_mu = gaps[0:int(lookn * self.mean_len) + 1].mean()
        mu = look_mu
        cur_err = 0
        seg_start = 0
        all_err = 0

        for i in tqdm(range(data_len)):
            x = data[i] - data[seg_start]
            y = i - seg_start
            if (x / mu > (y + epsilon) or x / mu < (y - epsilon)):
                self.seg_len.append(y)
                self.seg_err.append(cur_err)
                self.seg_mu.append(gaps[seg_start:i].mean())
                self.seg_sigma.append(gaps[seg_start:i].std())
                self.seg_num += 1
                seg_start = i
                cur_err = 0
                self.mean_len = (self.mean_len * (self.seg_num - 1) + self.seg_len[-1]) / self.seg_num
                look_list = gaps[seg_start:seg_start + int(lookn * self.mean_len) + 1]
                look_mu = look_list.mean()
                mu = look_mu
            else:
                res = abs(int(x / mu - y))
                cur_err += res
                self.err_all.append(res)
                all_err += res

        # last seg
        self.mae = all_err / data_len
        self.seg_num += 1
        self.seg_len.append(data_len - seg_start)
        self.seg_err.append(cur_err)
        self.seg_mu.append(gaps[seg_start:].mean())
        self.seg_sigma.append(gaps[seg_start:].std())
        print(self.seg_num, self.mae)
        return
