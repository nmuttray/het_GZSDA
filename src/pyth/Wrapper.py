from src.pyth.CDSPP import *
from src.pyth.preprocessing import *
from src.pyth.evaluating import *
from statistics import mean
from sklearn.model_selection import KFold
import multiprocessing as mp


# This class handles the testing of one dataset, testing all combinations of masked cells and preprocessing methods

class Wrapper:

    def __init__(self, PATH_in_source, PATH_in_target, PATH_out, modes, n_missing, dim=11,
                 alpha=[0.01, 0.1, 1, 10, 100, 1000], n_resample_source=300, n_resample_target=300):
        """
        Constructor
        :param PATH_in_source: final path = PATH_in_source + mode + .csv
        :param PATH_in_target: final path = PATH_in_target + train_/test_ + mode/label + .csv
        :param PATH_out: final path = PATH_out + mode + _pred/_h + .csv
        :param modes: list of preprocessing modes
        :param n_missing: number of masked labels
        :param dim: dimension of common latent space
        :param alpha: regularisation term
        :param n_resample_source: number obervations to sample
        :param n_resample_target: number obervations to sample
        """

        self.path_source = PATH_in_source
        self.path_target = PATH_in_target
        self.path_out = PATH_out
        self.modes = modes
        self.n_missing = n_missing
        self.dim = dim
        self.alpha = alpha
        self.n_source = n_resample_source
        self.n_target = n_resample_target


    def run_complete(self):
        # runs every preprocessing mode
        for i in self.modes:
            print("Starting with mode "+i)
            self.run_mode(i)
        return


    def test_alpha(self, comb, alpha, mode):
        """
        Runs the model for the same combination with different alphas
        :param comb: list of masked cells
        :param alpha: list of hyperparameters to test
        :param mode: preprocessing mode to choose
        :return: dataframe with performance metrics
        """
        PATH_in_source = self.path_source + mode + ".csv"
        PATH_in_source_label = self.path_source + "label.csv"

        PATH_in_target_train = self.path_target + "train_" + mode + ".csv"
        PATH_in_target_train_label = self.path_target + "train_label.csv"
        PATH_in_target_test = self.path_target + "test_" + mode + ".csv"
        PATH_in_target_test_label = self.path_target + "test_label.csv"

        X_source = pd.read_csv(PATH_in_source, index_col=0).to_numpy()
        y_source = pd.read_csv(PATH_in_source_label, index_col=0)["label"].to_numpy("int32")
        X_source, y_source = balance_sampling(X_source, y_source, self.n_source)
        print(Counter(y_source))

        X_train = pd.read_csv(PATH_in_target_train, index_col=0).to_numpy()
        y_train = pd.read_csv(PATH_in_target_train_label, index_col=0)["label"].to_numpy("int32")
        X_test = pd.read_csv(PATH_in_target_test, index_col=0).to_numpy()
        y_test = pd.read_csv(PATH_in_target_test_label, index_col=0)["label"].to_numpy("int32")
        X_train, y_train = balance_sampling(X_train, y_train, self.n_target)
        X_seen, X_unseen, y_seen, y_unseen = split_masked_cells(X_train, y_train, masked_cells=comb)
        print(Counter(y_seen))
        df_dict = {"alpha" : alpha}
        h_list = []
        acc_k_list = []
        acc_uk_list = []
        h_semi_list = []
        acc_k_semi_list = []
        acc_uk_semi_list = []
        for i in alpha:

            # fit model both normal and semisupervised
            model = CDSPP(X_source.T, y_source, i, self.dim, list(comb))
            model.fit(X_seen.T, y_seen)
            pred = model.predict(X_test.T)

            h, acc_known, acc_unknown = h_score(y_test, pred, comb)
            h_list.append(h)
            acc_k_list.append(acc_known)
            acc_uk_list.append(acc_unknown)

            model.fit_semi_supervised(X_seen.T, X_test.T, y_seen)
            pred = model.predict(X_test.T)

            h, acc_known, acc_unknown = h_score(y_test, pred, comb)
            h_semi_list.append(h)
            acc_k_semi_list.append(acc_known)
            acc_uk_semi_list.append(acc_unknown)

        df_dict["H"] = h_list
        df_dict["Acc_known"] = acc_k_list
        df_dict["Acc_unknown"] = acc_uk_list
        df_dict["H_semi"] = h_semi_list
        df_dict["Acc_known_semi"] = acc_k_semi_list
        df_dict["Acc_unknown_semi"] = acc_uk_semi_list
        results = pd.DataFrame.from_dict(df_dict)
        results.to_csv(self.path_out + mode + "_alphas.csv")
        return


    def run_desc(self, missing_start, missing_end, mode):
        """
        Runs the model with an increasing share of masked cells
        :param missing_start: lowest number of missing classes
        :param missing_end: highest number of missing classes
        :param mode: preprocessing mode
        :return: None
        """
        PATH_in_source = self.path_source + mode + ".csv"
        PATH_in_source_label = self.path_source + "label.csv"

        PATH_in_target_train = self.path_target + "train_" + mode + ".csv"
        PATH_in_target_train_label = self.path_target + "train_label.csv"
        PATH_in_target_test = self.path_target + "test_" + mode + ".csv"
        PATH_in_target_test_label = self.path_target + "test_label.csv"

        X_source = pd.read_csv(PATH_in_source, index_col=0).to_numpy()
        y_source = pd.read_csv(PATH_in_source_label, index_col=0)["label"].to_numpy("int32")
        X_source, y_source = balance_sampling(X_source, y_source, self.n_source)

        X_train = pd.read_csv(PATH_in_target_train, index_col=0).to_numpy()
        y_train = pd.read_csv(PATH_in_target_train_label, index_col=0)["label"].to_numpy("int32")
        X_test = pd.read_csv(PATH_in_target_test, index_col=0).to_numpy()
        y_test = pd.read_csv(PATH_in_target_test_label, index_col=0)["label"].to_numpy("int32")
        X_train, y_train = balance_sampling(X_train, y_train, self.n_target)
        for j in range(missing_start, missing_end+1):
            # extracts possible combinations
            self.n_missing = j
            print(j)
            unique_classes = set(y_source)
            combs = list(combinations(unique_classes, self.n_missing))
            for i in range(self.n_missing):
                cols.append("Missing " + str(i + 1))
            cols = cols + ["alpha", "y_true", "y_pred", "y_pred_semi"]
            results = pd.DataFrame(columns=cols)

            # every combinations is run on a different core if possible
            arguments = [(X_source, X_train, X_test, y_source, y_train, y_test, list(i), cols) for i in combs]
            pool = mp.Pool()
            all_results = pool.starmap(self.run_combination, arguments)
            pool.close()

            for result in all_results:
                results = pd.concat((results, result), ignore_index=True)
            results.to_csv(self.path_out + mode + "_pred.csv")
            scores = get_all_for_desc(results, combs, True, self.path_out + str(j) + "_" + mode + "_h.csv")
        return


    def run_mode(self, mode):
        """
        Runs all combinations with one preprocessing mode
        :param mode: preprocessing mode
        :return: None
        """
        PATH_in_source = self.path_source + mode + ".csv"
        PATH_in_source_label = self.path_source + "label.csv"

        PATH_in_target_train = self.path_target + "train_" + mode + ".csv"
        PATH_in_target_train_label = self.path_target + "train_label.csv"
        PATH_in_target_test =  self.path_target + "test_" + mode + ".csv"
        PATH_in_target_test_label = self.path_target + "test_label.csv"

        X_source = pd.read_csv(PATH_in_source, index_col=0).to_numpy()
        y_source = pd.read_csv(PATH_in_source_label, index_col=0)["label"].to_numpy("int32")
        X_source, y_source = balance_sampling(X_source, y_source, self.n_source)

        X_train = pd.read_csv(PATH_in_target_train, index_col=0).to_numpy()
        y_train = pd.read_csv(PATH_in_target_train_label, index_col=0)["label"].to_numpy("int32")
        X_test = pd.read_csv(PATH_in_target_test, index_col=0).to_numpy()
        y_test = pd.read_csv(PATH_in_target_test_label, index_col=0)["label"].to_numpy("int32")
        # for SMOTE resampling can already been done at this stage due to irrelevance of different classes
        X_train, y_train = balance_sampling(X_train, y_train, self.n_target)

        # extracts possible combinations
        unique_classes = set(y_source)
        combs = list(combinations(unique_classes, self.n_missing))

        # prepares dataframe
        cols = []
        for i in range(self.n_missing):
            cols.append("Missing " + str(i + 1))
        cols = cols + ["alpha", "y_true", "y_pred", "y_pred_semi"]
        results = pd.DataFrame(columns=cols)

        # every combinations is run on a different core if possible
        arguments = [(X_source, X_train, X_test, y_source, y_train, y_test, list(i), cols) for i in combs]
        pool = mp.Pool()
        all_results = pool.starmap(self.run_combination, arguments)
        pool.close()

        for result in all_results:
            results = pd.concat((results, result), ignore_index=True)
        results.to_csv(self.path_out + mode + "_pred.csv")
        scores = get_all(results, True, self.path_out + mode + "_h.csv")
        return


    def run_combination(self, X_source, X_train, X_test, y_source, y_train, y_test, comb, cols):
        """
        Runs one combination of masked cells
        :param X_source: source domain features
        :param X_train: target domain features
        :param X_test: test features
        :param y_source: source domain labels
        :param y_train: target domain labels
        :param y_test: test labels
        :param comb: list of masked cells
        :param cols: colnames of final dataframe
        :return: dataframe with results
        """
        # Mask combinations to be tested
        X_seen, X_unseen, y_seen, y_unseen = split_masked_cells(X_train, y_train, masked_cells=comb)


        # Cross-validate regularization
        alpha = self.get_alpha(X_source, X_seen, y_source, y_seen)
        print("Masked cells: "+str(comb))
        print("Best alpha: "+str(alpha))

        # fit model both normal and semisupervised
        print(list(comb))
        model = CDSPP(X_source.T, y_source, alpha, self.dim, list(comb))
        model.fit(X_seen.T, y_seen)
        pred = model.predict(X_test.T)
        print("Simple prediction...")
        model.fit_semi_supervised(X_seen.T, X_test.T, y_seen)
        pred_semi = model.predict(X_test.T)
        print("Semi-supervised prediction...")
        n_test = len(y_test)
        results_dict = dict()
        for i in range(len(cols)-4):
            results_dict[cols[i]] = n_test * [comb[i]]
        results_dict["alpha"] = n_test * [alpha]
        results_dict["y_true"] = y_test
        results_dict["y_pred"] = pred
        results_dict["y_pred_semi"] = pred_semi
        return pd.DataFrame.from_dict(results_dict)


    def get_alpha(self, X_source, X, y_source, y):
        """
        Performs sample- and class-wise cross validation
        :param X_source: source features
        :param X: target features (train)
        :param y_source: source labels
        :param y: target labels (train)
        :return:
        """
        overall_score = []
        for i in range(len(self.alpha)):
            score = []
            for train_index, test_index in KFold(shuffle=True, n_splits=5).split(X):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                for j in set(y):
                    X_seen, X_unseen, y_seen, y_unseen = split_masked_cells(X_train, y_train, masked_cells=[j])
                    model = CDSPP(X_source.T, y_source, self.alpha[i], self.dim)
                    model.fit(X_seen.T, y_seen)
                    pred = model.predict(X_test.T)
                    h, acc_known, acc_unknown = h_score(y_test, pred, [j])
                    score.append(acc_unknown)
            overall_score.append(mean(score))
        return self.alpha[np.argmax(np.array(overall_score))]

