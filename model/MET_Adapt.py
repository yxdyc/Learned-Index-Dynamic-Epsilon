import numpy as np
from tqdm import tqdm


class MET_Adapt:

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

    def __init__(self, low=10, high=500, rescale = 1):
        self.epsilon_low = low
        self.epsilon_high = high
        self.rescale = rescale
        self.seg_num = 0
        self.mae = 0
        self.seg_mu = []
        self.seg_sigma = []
        self.seg_len = []
        self.seg_mae = []
        self.seg_epsilon = []

    def choose_epsilon(self, mu, sigma, D):
        pass

    def learn_index(self, data):
        self.seg_num = 0
        self.mae = 0
        self.seg_mu = []
        self.seg_sigma = []
        self.seg_len = []
        self.seg_mae = []
        self.seg_epsilon = []
        data_len = len(data)
        left_len = data_len
        gaps = []
        for i in range(1, data_len):
            gaps.append(data[i] - data[i-1])
        gaps = np.array(gaps)

        mu = gaps.mean()
        sigma = gaps.std()
        epsilon = self.choose_epsilon(mu, sigma, data_len)
        cur_err = 0
        seg_start = 0
        all_err = 0

        for i in tqdm(range(data_len)):
            x = data[i] - data[seg_start]
            y = i - seg_start
            if(x / mu > (y + epsilon) or x / mu < (y - epsilon)):
                self.seg_len.append(y)
                self.seg_mae.append(cur_err / y)
                self.seg_mu.append(gaps[seg_start:i].mean())
                self.seg_sigma.append(gaps[seg_start:i].std())
                self.seg_num += 1
                self.seg_epsilon.append(epsilon)
                left_len = data_len - i
                seg_start = i
                mu = gaps[i:].mean()
                sigma = gaps[i:].std()
                epsilon = self.choose_epsilon(mu, sigma, left_len)
                cur_err = 0
            else:
                res = abs(int(x / mu - y))
                cur_err += res
                self.err_all = np.append(self.err_all, res)
                all_err += abs(int(x / mu - y))

        # last seg
        self.mae = all_err / data_len
        self.seg_num += 1
        self.seg_epsilon.append(epsilon)
        self.seg_len.append(data_len - seg_start)
        self.seg_mae.append(cur_err / (data_len - seg_start))
        self.seg_mu.append(gaps[seg_start:].mean())
        self.seg_sigma.append(gaps[seg_start:].std())
        print(self.seg_num,self.mae)

        return

    def learn_index_lookahead(self, data, lookn=200):
        self.seg_num = 0
        self.mae = 0
        self.seg_mu = []
        self.seg_sigma = []
        self.seg_len = []
        self.seg_mae = []
        self.seg_epsilon = []
        data_len = len(data)
        left_len = data_len
        gaps = []
        for i in range(1, data_len):
            gaps.append(data[i] - data[i-1])
        gaps = np.array(gaps)

        mu = gaps[0:lookn].mean()
        sigma = gaps[0:lookn].std()
        epsilon = self.choose_epsilon(mu, sigma, data_len)
        cur_err = 0
        seg_start = 0
        all_err = 0

        for i in tqdm(range(data_len)):
            x = data[i] - data[seg_start]
            y = i - seg_start
            if(x / mu > (y + epsilon) or x / mu < (y - epsilon)):
                self.seg_len.append(y)
                self.seg_mae.append(cur_err / y)
                self.seg_mu.append(gaps[seg_start:i].mean())
                self.seg_sigma.append(gaps[seg_start:i].std())
                self.seg_num += 1
                left_len = data_len - i
                seg_start = i
                mu = gaps[i:i+lookn].mean()
                sigma = gaps[i:i+lookn].std()
                epsilon = self.choose_epsilon(mu, sigma, left_len)
                cur_err = 0
            else:
                cur_err += abs(int(x / mu - y))
                all_err += abs(int(x / mu - y))

        # last seg
        self.mae = all_err / data_len
        self.seg_num += 1
        self.seg_epsilon.append(epsilon)
        self.seg_len.append(data_len - seg_start)
        self.seg_mae.append(cur_err / (data_len - seg_start))
        self.seg_mu.append(gaps[seg_start:].mean())
        self.seg_sigma.append(gaps[seg_start:].std())
        print(self.seg_num,self.mae)
        return


class MET_Adapt_Smax(MET_Adapt):

    def __init__(self, Smax, low=10, high=500):
        self.epsilon_low = low
        self.epsilon_high = high
        self.seg_num = 0
        self.mae = 0
        self.seg_mu = []
        self.seg_sigma = []
        self.seg_len = []
        self.seg_mae = []
        self.seg_epsilon = []
        self.Smax = Smax

    def choose_epsilon(self, mu, sigma, D):
        left_seg_num = (self.Smax - self.seg_num) if self.Smax >= self.seg_num + 1 else 1
        if sigma == 0:
            return self.epsilon_low
        epsilon = int((D * sigma ** 2 / mu ** 2 / left_seg_num) ** 0.5)
        if epsilon < self.epsilon_low:
            epsilon = self.epsilon_low
        elif epsilon > self.epsilon_high:
            epsilon = self.epsilon_high


        return epsilon


class MET_Adapt_Tmax(MET_Adapt):

    def __init__(self, Tmax, low=10, high=500):
        self.epsilon_low = low
        self.epsilon_high = high
        self.seg_num = 0
        self.mae = 0
        self.seg_mu = []
        self.seg_sigma = []
        self.seg_len = []
        self.seg_mae = []
        self.seg_epsilon = []
        self.Tmax = Tmax

    def choose_epsilon(self, mu, sigma, D):
        learned_seg_len = np.array(self.seg_len)
        learned_seg_mae = np.array(self.seg_mae)
        D_ = learned_seg_len.sum()
        allowed_mae = (self.Tmax * (D+D_) - (learned_seg_len * learned_seg_mae).sum()) / D
        epsilon = int((allowed_mae - 1 - 0.53 * sigma / mu) / 0.53)
        if epsilon < self.epsilon_low:
            epsilon = self.epsilon_low
        elif epsilon > self.epsilon_high:
            epsilon = self.epsilon_high

        return epsilon