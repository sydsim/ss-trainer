from collections import defaultdict
import pickle
import h5py
import numpy as np
from torch.utils.data import Dataset


class KlinesDataset(Dataset):
    def __init__(
        self,
        data_path,
        input_length,
        selected_columns=None,
        y_prediction_length=20,
    ):
        self.input_length = input_length
        self.selected_columns = selected_columns
        self.y_prediction_length = y_prediction_length

        with h5py.File(data_path, "r") as f:
            self.base_price = f["base_price"][:].astype(np.float64)
            self.best_ask = f["best_ask"][:].astype(np.float64)
            self.best_bid = f["best_bid"][:].astype(np.float64)

            self.profit_long = f["profit_long"][:].astype(np.float32)
            self.profit_short = f["profit_short"][:].astype(np.float32)

            self.threshold_long = f["threshold_long"][:].astype(np.float32)
            self.threshold_short = f["threshold_short"][:].astype(np.float32)
            self.exec_price_long = f["exec_price_long"][:].astype(np.float32)
            self.exec_price_short = f["exec_price_short"][:].astype(np.float32)

            n = len(self.profit_long)
            self.ts_total = f["ts_total"][:n]
            feature = f["feature"][:n].astype(np.float64)
            self.size = n - input_length + 1
            self.n = n

        self.init(feature)

    def init(self, feature):
        dates = self.ts_total.astype('datetime64[D]')
        unique_dates, inverse = np.unique(dates, return_inverse=True)
        counts = np.bincount(inverse)
        pct = int(np.percentile(counts, 50))

        # 그룹별 인덱스를 contiguous하게 만들기 위해 정렬
        sorted_idx = np.argsort(inverse)
        boundaries = np.concatenate(([0], np.cumsum(counts)))

        # 샘플링 결과 저장
        trigger = np.zeros(len(self.ts_total), dtype=bool)

        # 각 그룹별로 한 번만 인덱스 슬라이스
        for i in range(len(counts)):
            start, end = boundaries[i], boundaries[i+1]
            idxs = sorted_idx[start:end]
            if len(idxs) <= pct:
                trigger[idxs] = True
            else:
                selected = np.random.choice(idxs, size=pct, replace=False)
                trigger[selected] = True

        last_nan = np.argwhere(np.any(np.isnan(
            np.concatenate([
                feature,
            ], axis=-1)
        ), axis=1))

        if len(last_nan) > 0:
            last_nan = last_nan.max()
        else:
            last_nan = 0
        trigger = trigger & (np.arange(self.n) >= last_nan + self.input_length)

        is_nan = np.any(np.isnan(np.stack([
            self.base_price,
            self.best_ask,
            self.best_bid,
            self.profit_long,
            self.profit_short,
            self.threshold_long,
            self.threshold_short,
            self.exec_price_long,
            self.exec_price_short,
        ], axis=-1)), axis=-1)

        trigger = trigger & ~is_nan

        self.feature = feature
        self.data = None
        self.indices = np.argwhere(trigger)[..., 0]
        if self.selected_columns is not None:
            self.feature = self.feature[:, self.selected_columns]
        self.num_features = self.feature.shape[-1]
        self.size = len(self.indices)

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        index = self.indices[index]
        ts = self.ts_total[index]
        x = self.data[index - self.input_length + 1:index + 1]
        b_p = self.base_price[index]
        # e_l = self.exec_price_long[index]
        # e_s = self.exec_price_short[index]
        p_l = self.profit_long[index]
        p_s = self.profit_short[index]
        t_l = self.threshold_long[index]
        t_s = self.threshold_short[index]
        return ts, x, b_p, p_l, p_s, t_l, t_s

    def normalize(self, indices, normalizer):
        self.data, self.norm = normalizer(self.feature, indices)
        self.data = self.data.astype(dtype=np.float32)

    def oversampling(self, indices):
        index_d = defaultdict(list)
        for index in indices:
            index_d[self.get_label(index)].append(index)
        print("Oversampling - current size", {k: len(indices) for k, indices in index_d.items()})

        n = max(len(v) for v in index_d.values())
        new_indices = [indices]
        for v in index_d.values():
            if n > len(v):
                new_indices.append(np.random.choice(v, n - len(v), replace=True))
        new_indices = np.concatenate(new_indices, axis=-1)

        count_d = defaultdict(int)
        for index in new_indices:
            count_d[self.get_label(index)] += 1
        print("Oversampling - current size", count_d)
        return new_indices

    def get_label(self, index):
        index = self.indices[index]
        p = 0
        if self.profit_long[index] > self.threshold_long[index]:
            p += 1
        if self.profit_short[index] > self.threshold_short[index]:
            p += 2
        if p == 3:
            p = 0
        return p

    def filter_by_period(self, period_start, period_end, sample_rate=None, sample_size=None, is_train=False):
        period_start = period_start.timestamp() * 1e6
        period_end = period_end.timestamp() * 1e6
        indices = np.argwhere(
            (self.ts_total[self.indices + is_train * self.y_prediction_length] > period_start) &
            (self.ts_total[self.indices + is_train * self.y_prediction_length] < period_end)
        )[:, 0]
        if sample_rate is not None:
            target_size = int(len(indices) * sample_rate)
        elif sample_size is not None:
            target_size = sample_size
        return np.sort(np.random.choice(indices, target_size, replace=False))

    def save_norm(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.norm, f)


class DummyKlinesDataset(KlinesDataset):
    def __init__(
        self,
        input_length,
        start_date,
        end_date,
        selected_columns=None,
        y_prediction_length=20,
        n=10000,
    ):
        self.input_length = input_length
        self.selected_columns = selected_columns
        self.y_prediction_length = y_prediction_length

        self.base_price = np.abs(np.empty([n], np.float64))
        self.best_ask = np.abs(np.empty([n], np.float64))
        self.best_bid = np.abs(np.empty([n], np.float64))

        self.profit_long = np.abs(np.empty([n], np.float64))
        self.profit_short = np.abs(np.empty([n], np.float64))

        self.threshold_long = np.abs(np.empty([n], np.float64))
        self.threshold_short = np.abs(np.empty([n], np.float64))
        self.exec_price_long = np.abs(np.empty([n], np.float64))
        self.exec_price_short = np.abs(np.empty([n], np.float64))

        start_ts = start_date.timestamp() * 1e6
        end_ts = end_date.timestamp() * 1e6
        self.ts_total = np.sort(np.random.randint(start_ts, end_ts, n, dtype=np.int64))
        feature = np.empty([n, 50], np.float64)
        self.size = n - input_length + 1
        self.n = n

        self.init(feature)
