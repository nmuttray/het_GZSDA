import pandas as pd
import numpy as np
import anndata as ad
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
import warnings

# implements helper functions for the initial data QC
def prepare_adata(PATH):
    raw_data = pd.read_csv(PATH, index_col=0)
    labels = raw_data["label"]
    raw_data.drop(columns=["label"], inplace=True)
    adata = ad.AnnData(raw_data, dtype=np.int32)
    adata.obs["label"] = labels
    return adata

# exports final Seurat preprocessing
def export_final(PATH, PATH_label, adata):
    final_data = pd.DataFrame(adata.obsm["label"], index=adata.obs_names)
    final_data.to_csv(PATH_label)
    final_data = pd.DataFrame(adata.obsm["X_pca"], index=adata.obs_names)
    final_data.to_csv(PATH)
    return





def balance_sampling(X, y, n=100):
    """
    Re-balances data by over-sampling with SMOTE and under-sampling randomly
    :param X: feature matrix
    :param y: labels
    :param n: desired samples per class
    :return: resampled feature matrix, resampled labels
    """
    warnings.filterwarnings('ignore')
    counts = Counter(y)
    under = np.array([], dtype="int32")
    over = np.array([], dtype="int32")
    for i in counts.keys():
        if counts[i] <= n:
            over = np.concatenate((over, np.array([i])))
        else:
            under = np.concatenate((under, np.array([i])))
    if len(over) == 0:
        dict_under = dict(zip(under, [n for i in range(len(under))]))
        under_sam =  RandomUnderSampler(sampling_strategy=dict_under)
        X_under, y_under = under_sam.fit_resample(X, y)
        return X_under, y_under
    elif len(under) == 0:
        dict_over = dict(zip(over, [n for i in range(len(over))]))
        over_sam = SMOTE(sampling_strategy=dict_over)
        X_over, y_over = over_sam.fit_resample(X, y)
        return X_over, y_over
    else:
        if len(over) == 1:
            # Tricks SMOTE into oversampling one class
            pseudo_X = np.full((n, X.shape[1]), 10000)
            pseudo_y = np.full(n, 10000)
            dict_over = dict()
            dict_over[over[0]] = n
            dict_over[10000] = n
            is_over = np.in1d(y, over)
            over_sam = SMOTE(sampling_strategy=dict_over)
            is_over = np.in1d(y, over)
            X_over_, y_over_ = over_sam.fit_resample(np.concatenate((X[is_over], pseudo_X)),
                                                     np.concatenate((y[is_over], pseudo_y)))
            X_over = X_over_[y_over_==over[0]]
            y_over = y_over_[y_over_==over[0]]

        else:
            dict_over = dict(zip(over, [n for i in range(len(over))]))
            over_sam = SMOTE(sampling_strategy=dict_over)
            is_over = np.in1d(y, over)
            X_over, y_over = over_sam.fit_resample(X[is_over], y[is_over])

        if len(under) == 1:
            # Tricks RandomUnderSampler into working with one class
            pseudo_X = np.full((n, X.shape[1]), 10000)
            pseudo_y = np.full(n, 10000)
            dict_under = dict()
            dict_under[under[0]] = n
            dict_under[10000] = n
            is_under = np.in1d(y, under)
            under_sam = RandomUnderSampler(sampling_strategy=dict_under)
            is_under = np.in1d(y, under)
            X_under_, y_under_ = under_sam.fit_resample(np.concatenate((X[is_under], pseudo_X)),
                                                        np.concatenate((y[is_under], pseudo_y)))
            X_under = X_under_[y_under_==under[0]]
            y_under = y_under_[y_under_==under[0]]
        else:
            dict_under = dict(zip(under, [n for i in range(len(under))]))
            under_sam = RandomUnderSampler(sampling_strategy=dict_under)
            is_under = np.in1d(y, under)
            X_under, y_under = under_sam.fit_resample(X[is_under], y[is_under])

        X_combined_sampling = np.concatenate((X_over, X_under))
        y_combined_sampling = np.concatenate((y_over, y_under))
        return X_combined_sampling, y_combined_sampling


def split_masked_cells(X_t, y_t, masked_cells, balance=False, n=500):
    """
    Maskes cells for generalized zero-shot learning
    :param X_t: feature matrix of target data
    :param y_t: labels of target data
    :param masked_cells: list of cells to be masked from data
    :param balance: whether to balance seen train data
    :param n: desired number of samples per class
    :return: features of seen classes, features of unseen classes, labels seen classes, labels unseen classes
    """
    keep = np.in1d(y_t, masked_cells, invert=True)
    X_t_seen = X_t[keep]
    X_t_unseen = X_t[~keep]
    y_seen = y_t[keep]
    y_unseen = y_t[~keep]
    if balance:
        X_t_seen, y_seen = balance_sampling(X_t_seen, y_seen, n)
    return X_t_seen, X_t_unseen, y_seen, y_unseen

