import pandas as pd
from sklearn.model_selection import train_test_split
import anndata as ad
import scanpy as sc
import pandas as pd
from src.pyth.mousipy import *

m2h_tab = pd.read_csv('biomart/mouse_to_human_biomart_export.csv').set_index('Gene name')

# final preparation of data
def prepare_data(PATH_in, PATH_out):
    """
    Creates test split and classes as integers
    :param PATH_in: path to data files (everything up to species and mode)
    :param PATH_out: path for output
    :return: None
    """
    modes = ["scanpy_pca.csv", "dca_pca.csv", "scetm.csv"]
    human_label = pd.read_csv(PATH_in + "human_labels.csv", index_col=0)
    mouse_label = pd.read_csv(PATH_in + "mouse_labels.csv", index_col=0)
    shared = set(human_label["label"]).intersection(set(mouse_label["label"]))

    human = pd.read_csv(PATH_in + "human_" + modes[0], index_col=0)
    mouse = pd.read_csv(PATH_in + "mouse_" + modes[0], index_col=0)
    human_red = human[human_label["label"].isin(shared)]
    mouse_red = mouse[mouse_label["label"].isin(shared)]
    human_label_red = human_label[human_label["label"].isin(shared)]
    mouse_label_red = mouse_label[mouse_label["label"].isin(shared)]
    mouse_index = mouse_label_red.index
    dictionary = dict(zip(list(shared), range(len(shared))))
    dict_df = pd.DataFrame(list(dictionary.items()), columns=['label', 'value']).to_csv(PATH_out + "dict.csv")
    human_label_red["label"] = human_label_red["label"].map(dictionary)
    mouse_label_red["label"] = mouse_label_red["label"].map(dictionary)
    mouse_label_red.to_csv(PATH_out + "mouse_red_label.csv")

    X_train, X_test, y_train, y_test = train_test_split(human_red, human_label_red, test_size=0.2, random_state=3,
                                                        stratify=human_label_red)
    train_index = y_train.index
    test_index = y_test.index
    y_train.to_csv(PATH_out + "human_red_train_label.csv")
    y_test.to_csv(PATH_out + "human_red_test_label.csv")

    mouse_red.loc[mouse_index].to_csv(PATH_out + "mouse_red_" + modes[0])
    X_train.to_csv(PATH_out + "human_red_train_" + modes[0])
    X_test.to_csv(PATH_out + "human_red_test_" + modes[0])

    for i in range(2):
        human = pd.read_csv(PATH_in + "human_" + modes[i + 1], index_col=0)
        mouse = pd.read_csv(PATH_in + "mouse_" + modes[i + 1], index_col=0)
        mouse.loc[mouse_index].to_csv(PATH_out + "mouse_red_" + modes[i + 1])
        human.loc[train_index].to_csv(PATH_out + "human_red_train_" + modes[i + 1])
        human.loc[test_index].to_csv(PATH_out + "human_red_test_" + modes[i + 1])
    return

# final preparation of data for human to mouse
def prepare_data_rev(PATH_in, PATH_out):
    modes = ["scanpy_pca.csv", "dca_pca.csv", "scetm.csv"]
    human_label = pd.read_csv(PATH_in + "mouse_labels.csv", index_col=0)
    mouse_label = pd.read_csv(PATH_in + "human_labels.csv", index_col=0)
    shared = set(human_label["label"]).intersection(set(mouse_label["label"]))

    human = pd.read_csv(PATH_in + "mouse_" + modes[0], index_col=0)
    mouse = pd.read_csv(PATH_in + "human_" + modes[0], index_col=0)
    human_red = human[human_label["label"].isin(shared)]
    mouse_red = mouse[mouse_label["label"].isin(shared)]
    human_label_red = human_label[human_label["label"].isin(shared)]
    mouse_label_red = mouse_label[mouse_label["label"].isin(shared)]
    mouse_index = mouse_label_red.index
    dictionary = dict(zip(list(shared), range(len(shared))))
    dict_df = pd.DataFrame(list(dictionary.items()), columns=['label', 'value']).to_csv(PATH_out + "dict2.csv")
    human_label_red["label"] = human_label_red["label"].map(dictionary)
    mouse_label_red["label"] = mouse_label_red["label"].map(dictionary)
    mouse_label_red.to_csv(PATH_out + "human_red_label.csv")

    X_train, X_test, y_train, y_test = train_test_split(human_red, human_label_red, test_size=0.2, random_state=3,
                                                        stratify=human_label_red)
    train_index = y_train.index
    test_index = y_test.index
    y_train.to_csv(PATH_out + "mouse_red_train_label.csv")
    y_test.to_csv(PATH_out + "mouse_red_test_label.csv")

    mouse_red.loc[mouse_index].to_csv(PATH_out + "human_red_" + modes[0])
    X_train.to_csv(PATH_out + "mouse_red_train_" + modes[0])
    X_test.to_csv(PATH_out + "mouse_red_test_" + modes[0])

    for i in range(2):
        human = pd.read_csv(PATH_in + "mouse_" + modes[i + 1], index_col=0)
        mouse = pd.read_csv(PATH_in + "human_" + modes[i + 1], index_col=0)
        mouse.loc[mouse_index].to_csv(PATH_out + "human_red_" + modes[i + 1])
        human.loc[train_index].to_csv(PATH_out + "mouse_red_train_" + modes[i + 1])
        human.loc[test_index].to_csv(PATH_out + "mouse_red_test_" + modes[i + 1])
    return


# prepares data for homogeneous DA models
def prepare_adata_for_comparison(PATH_in_m, PATH_in_ml, PATH_in_h, PATH_in_h_train, PATH_in_h_test, PATH_out):
    m2h_tab = pd.read_csv('biomart/mouse_to_human_biomart_export.csv').set_index('Gene name')
    adata_m = ad.read_h5ad(PATH_in_m)
    labels_m = pd.read_csv(PATH_in_ml, index_col=0)
    adata_m = adata_m[labels_m.index, :].copy()
    adata_m.obs["domain"] = 0
    adata_m.obs["test"] = False
    adata_m.obs["cell_type"] = labels_m["label"]
    adata_m.obs["label"] = labels_m["label"]

    adata_h = ad.read_h5ad(PATH_in_h)
    labels_h_train = pd.read_csv(PATH_in_h_train, index_col=0)
    labels_h_test = pd.read_csv(PATH_in_h_test, index_col=0)
    labels_comb = pd.concat((labels_h_train, labels_h_test))
    adata_h = adata_h[labels_comb.index, :].copy()
    adata_h.obs["domain"] = 1
    adata_h.obs["test"] = False
    adata_h.obs["test"][labels_h_test.index] = True
    adata_h.obs["cell_type"] = labels_comb["label"]
    adata_h.obs["label"] = labels_comb["label"]

    direct, multiple, no_hit, no_index = check_orthologs(adata_m.var_names, tab=m2h_tab)
    new_data = translate_direct(adata_m, direct, no_index)
    new_data = collapse_duplicate_genes(new_data, True)
    adata_final = ad.concat([new_data, adata_h], join='inner')
    sc.pp.normalize_total(adata_final, target_sum=1e6)
    sc.pp.log1p(adata_final)
    adata_final.write(PATH_out, compression="gzip")

    return adata_final