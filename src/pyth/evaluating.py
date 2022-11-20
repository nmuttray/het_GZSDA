import pandas as pd
import numpy as np
from itertools import combinations
from sklearn.metrics import balanced_accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from collections import Counter
from scipy.stats import wilcoxon


def h_score(y_true, y_pred, masked_cells):
    """
    H score
    :param y_true: true values
    :param y_pred: predictions
    :param masked_cells: list of masked cells
    :return: h-score, acc. of known classes, acc of unknown classes
    """
    warnings.filterwarnings('ignore')
    known = np.in1d(y_true, masked_cells, invert=True)
    acc_known = balanced_accuracy_score(y_true[known], y_pred[known])
    acc_unknown = balanced_accuracy_score(y_true[~known], y_pred[~known])
    h = (2 * acc_known * acc_unknown) / (acc_known + acc_unknown)
    return h, acc_known, acc_unknown


# Selects run based on combination
def select_run(results, missing):
    """
    Helper function selecting run based on missing classes
    :param results: dataframe with true test labels and predictions as produced by Wrapper.run_mode()
    :param missing: list of missing classes
    :return: subset of predictions for the chosen combination
    """
    n_missing = len(missing)
    for i in range(n_missing):
        results = results[results[("Missing " + str(i + 1))].isin(missing)]
    return results

def analyze_confusion(PATH_wd, PATH_organ):
    """
    Calculates most common miss-classification for all preprocessing methods
    :param PATH_wd: main directory
    :param PATH_organ: path to data set relative to directory
    :return: dataframe with miss-classification for each cell type and preprocessing method
    """
    PATH_dict = PATH_wd + PATH_organ + "dict.csv"
    PATH_in = [PATH_wd + PATH_organ + "scanpy_pca_pred.csv",
               PATH_wd + PATH_organ + "dca_pca_pred.csv",
               PATH_wd + PATH_organ + "scetm_pred.csv"]

    PATH_out = PATH_wd + PATH_organ + "common_mis.csv"
    modes = ["CDSPP+Seurat", "CDSPP+DCA", "CDSPP+scETM"]
    cell_types = pd.read_csv(PATH_dict, index_col=0)
    df_dict = {"cell type": cell_types["label"]}
    for i in range(3):
        result = pd.read_csv(PATH_in[i], index_col=0)
        conf_list = []
        for j in range(11):
            conf = get_most_confused(result, j)
            conf_list.append(cell_types["label"].loc[conf])
        df_dict[modes[i]] = conf_list

    confusion = pd.DataFrame.from_dict(df_dict)
    confusion.to_csv(PATH_out)
    return confusion

def get_most_confused(results, cell_type):
    """
    Retrieves most frequent wrong prediction for cell type
    :param results:  dataframe with true test labels and predictions as produced by Wrapper.run_mode()
    :param cell_type: cell type of inquery
    :return: most frequent wrong prediction
    """
    keep = np.full(len(results), False)
    for i in range(2):
        keep = keep | (results[("Missing " + str(i + 1))] == cell_type)
    results = results[keep]
    results = results[results["y_true"] == cell_type]
    results = results[results["y_pred_semi"] != cell_type]
    if len(results) == 0:
        conf = cell_type
    else:
        counts = Counter(results["y_pred_semi"])
        conf = list(counts.keys())[0]
    return conf

# Creates result file
def get_all(results, to_csv, PATH_out):
    """
    Calculates all perfomance scores for each masked combination
    :param results: dataframe with true test labels and predictions as produced by Wrapper.run_mode()
    :param to_csv: whether to write csv
    :param PATH_out: path to write to
    :return:
    """
    n_missing = results.shape[1] - 4
    unique_classes = set(results["y_true"])
    combs = list(combinations(unique_classes, n_missing))
    cols = ["Missing", "alpha", "H", "Acc_known", "Acc_unknown", "H_semi", "Acc_known_semi", "Acc_unknown_semi"]
    missing = []
    alpha = []
    h_list = []
    known_list = []
    unknown_list = []
    h_list_semi = []
    known_list_semi = []
    unknown_list_semi = []
    for i in combs:
        selected = select_run(results, list(i))
        h, acc_known, acc_unknown = h_score(selected["y_true"].to_numpy(dtype="int32"), selected["y_pred"].to_numpy(dtype="int32"), list(i))
        missing.append(str(i))
        alpha.append(selected["alpha"].iloc[0])
        h_list.append(h)
        known_list.append(acc_known)
        unknown_list.append(acc_unknown)

        h, acc_known, acc_unknown = h_score(selected["y_true"].to_numpy(dtype="int32"),
                                            selected["y_pred_semi"].to_numpy(dtype="int32"), list(i))
        h_list_semi.append(h)
        known_list_semi.append(acc_known)
        unknown_list_semi.append(acc_unknown)

    scores = pd.DataFrame(zip(missing, alpha, h_list, known_list, unknown_list,
                              h_list_semi, known_list_semi, unknown_list_semi), columns=cols)
    if to_csv:
        scores.to_csv(PATH_out)
    return scores

def get_all_for_desc(results,combs, to_csv, PATH_out):
    """
    Modified version when testing descending number of known classes
    """
    n_missing = results.shape[1] - 4
    cols = ["Missing", "alpha", "H", "Acc_known", "Acc_unknown", "H_semi", "Acc_known_semi", "Acc_unknown_semi"]
    missing = []
    alpha = []
    h_list = []
    known_list = []
    unknown_list = []
    h_list_semi = []
    known_list_semi = []
    unknown_list_semi = []
    for i in combs:
        selected = select_run(results, list(i))
        h, acc_known, acc_unknown = h_score(selected["y_true"].to_numpy(dtype="int32"), selected["y_pred"].to_numpy(dtype="int32"), list(i))
        missing.append(str(i))
        alpha.append(selected["alpha"].iloc[0])
        h_list.append(h)
        known_list.append(acc_known)
        unknown_list.append(acc_unknown)

        h, acc_known, acc_unknown = h_score(selected["y_true"].to_numpy(dtype="int32"),
                                            selected["y_pred_semi"].to_numpy(dtype="int32"), list(i))
        h_list_semi.append(h)
        known_list_semi.append(acc_known)
        unknown_list_semi.append(acc_unknown)

    scores = pd.DataFrame(zip(missing, alpha, h_list, known_list, unknown_list,
                              h_list_semi, known_list_semi, unknown_list_semi), columns=cols)
    if to_csv:
        scores.to_csv(PATH_out)
    return scores

# plot function for confusion matrix
def make_confusion_matrix(cf,
                          group_names=None,
                          categories='auto',
                          count=True,
                          percent=True,
                          cbar=True,
                          xyticks=True,
                          xyplotlabels=True,
                          sum_stats=True,
                          figsize=None,
                          cmap='Blues',
                          title=None):


    blanks = ['' for i in range(cf.size)]

    if group_names and len(group_names) == cf.size:
        group_labels = ["{}\n".format(value) for value in group_names]
    else:
        group_labels = blanks

    if count:
        group_counts = ["{0:0.0f}\n".format(value) for value in cf.flatten()]
    else:
        group_counts = blanks

    if percent:
        cf = cf.astype('float') / cf.sum(axis=1)[:, np.newaxis]
        group_percentages = ["{0:.2%}".format(value) for value in cf.flatten()]
    else:
        group_percentages = blanks

    box_labels = [f"{v1}{v2}{v3}".strip() for v1, v2, v3 in zip(group_labels, group_counts, group_percentages)]
    box_labels = np.asarray(box_labels).reshape(cf.shape[0], cf.shape[1])

    # CODE TO GENERATE SUMMARY STATISTICS & TEXT FOR SUMMARY STATS
    if sum_stats:
        # Accuracy is sum of diagonal divided by total observations
        accuracy = np.trace(cf) / float(np.sum(cf))

        # if it is a binary confusion matrix, show some more stats
        if len(cf) == 2:
            # Metrics for Binary Confusion Matrices
            precision = cf[1, 1] / sum(cf[:, 1])
            recall = cf[1, 1] / sum(cf[1, :])
            f1_score = 2 * precision * recall / (precision + recall)
            stats_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}".format(
                accuracy, precision, recall, f1_score)
        else:
            stats_text = "\n\nAccuracy={:0.3f}".format(accuracy)
    else:
        stats_text = ""

    # SET FIGURE PARAMETERS ACCORDING TO OTHER ARGUMENTS
    if figsize == None:
        # Get default figure size if not set
        figsize = plt.rcParams.get('figure.figsize')

    if xyticks == False:
        # Do not show categories if xyticks is False
        categories = False

    # MAKE THE HEATMAP VISUALIZATION
    plt.figure(figsize=figsize)
    sns.heatmap(cf, annot=box_labels, fmt="", cmap=cmap, cbar=cbar, xticklabels=categories, yticklabels=categories,
                linewidth=0.5, linecolor="black")

    if xyplotlabels:
        plt.ylabel('True')
        plt.xlabel('Predicted' + stats_text)
    else:
        plt.xlabel(stats_text)

    if title:
        plt.title(title)


# Creates table of zero-shot accuracies for each model and cell
def get_all_unknow_acc(PATH_wd, PATH_organ):
    """
    Calculates zero-shot accuracy averaged over all combinations
    :param PATH_wd: main directory
    :param PATH_organ: path to data set relative to directory
    :return: dataframe with zero-shot accuracy for each cell type and preprocessing method
    """
    PATH_dict = PATH_wd + PATH_organ + "dict.csv"
    PATH_source = PATH_wd + PATH_organ + "mouse_label.csv"
    PATH_in = [PATH_wd + PATH_organ + "scanpy_pca_pred.csv",
               PATH_wd + PATH_organ + "dca_pca_pred.csv",
               PATH_wd + PATH_organ + "scetm_pred.csv",
               PATH_wd + PATH_organ + "scadapt_pred.csv",
               PATH_wd + PATH_organ + "scnym_pred.csv"]
    PATH_out = PATH_wd + PATH_organ + "unknown_acc_per_cell.csv"
    modes = ["CDSPP+Seurat", "CDSPP+DCA", "CDSPP+scETM", "scAdapt", "scNYM"]
    cell_types = pd.read_csv(PATH_dict, index_col=0)["label"]
    df_dict = {"cell type": cell_types}
    for i in range(3):
        result = pd.read_csv(PATH_in[i], index_col=0)
        acc_list = []
        for j in range(11):
            acc_unknown = get_unknown_acc(result, j)
            acc_list.append(acc_unknown)
        df_dict[modes[i]] = acc_list
    for i in range(3, 5):
        result = pd.read_csv(PATH_in[i], index_col=0)
        acc_list = []
        for j in range(11):
            acc_unknown = get_unknown_acc_us(result, j)
            acc_list.append(acc_unknown)
        df_dict[modes[i]] = acc_list
    count_target = Counter(result["y_true"])
    df_dict["n target"] = [count_target[i] for i in range(11)]
    result = pd.read_csv(PATH_source, index_col=0)
    count_source = Counter(result["label"])
    df_dict["n source"] = [count_source[i] for i in range(11)]
    unknown_acc = pd.DataFrame.from_dict(df_dict)
    unknown_acc.to_csv(PATH_out)
    return unknown_acc



def get_unknown_acc(results, cell_type):
    """
    Helper function extracting zero-shot accuracy
    :param results: dataframe with true test labels and predictions as produced by Wrapper.run_mode()
    :param cell_type: cell type of inquery
    :return: unknown accuracy
    """
    keep = np.full(len(results), False)
    for i in range(2):
        keep = keep | (results[("Missing " + str(i + 1))] == cell_type)
    results = results[keep]
    results = results[results["y_true"] == cell_type]
    unknown_acc = sum(results["y_pred_semi"] == results["y_true"]) / len(results)
    return unknown_acc


def get_unknown_acc_us(results, cell_type):
    """
    Helper function extracting zero-shot accuracy for scNym, scAdapt
    :param results: dataframe with true test labels and predictions
    :param cell_type: cell type of inquery
    :return: unknown accuracy
    """
    results = results[results["y_true"] == cell_type]
    unknown_acc = sum(results["y_pred"] == results["y_true"]) / len(results)
    return unknown_acc

def analyze_difference(PATH_wd, PATH_organ):
    """
    Tests whether pseudo-labeling statistically significant
    :param PATH_wd: main directory
    :param PATH_organ: path to data set relative to directory
    :return: dataframe with test statistic
    """
    PATH_in = [PATH_wd + PATH_organ + "scanpy_pca_h.csv",
               PATH_wd + PATH_organ + "dca_pca_h.csv",
               PATH_wd + PATH_organ + "scetm_h.csv"]

    PATH_out = PATH_wd + PATH_organ + "pvalue_ssl.csv"
    modes = ["CDSPP+Seurat", "CDSPP+DCA", "CDSPP+scETM"]
    df_dict = dict()
    for i in range(3):
        result = pd.read_csv(PATH_in[i], index_col=0)
        W, p = wilcoxon(result["Acc_unknown"], result["Acc_unknown_semi"], alternative="less")
        p = p
        df_dict[modes[i]] = [W, p, p*3]

    test = pd.DataFrame.from_dict(df_dict)
    test.to_csv(PATH_out)
    return test