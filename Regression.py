import numpy as np
from collections.abc import Iterable
import matplotlib.pyplot as plt
from scipy.special import expit

class LogisticClassifier:
    def __init__(self, training_data, train_count=-1):
        """Pass in non-normalized data without the bias column for all functions"""
        if train_count == -1: train_count = training_data.shape[0]
        training_data = training_data[np.random.choice(training_data.shape[0], train_count, replace=False), :]
        self.x_dim = training_data.shape[1] - 1
        self.featureNorms = None
        self.raw_training_data = training_data.copy()
        self.normalized_training_data = self.normalize(training_data)
        self.w = np.zeros(self.x_dim + 1)

    def normalize(self, data, addBias=True):
        """normalizes columns and optionally adds bias at beginning of design matrix"""
        new = self.normalizeColumns(data)
        new = self.normalizeRows(new)
        if addBias:
            if np.average(new[:, 0]) != 1:  # make sure bias has not already been added!
                new = np.hstack((np.ones((len(new), 1)), new))
        return new

    def normalizeRows(self, data):
        new = data.copy()
        if data.shape[1] == self.x_dim + 1:
            norm = np.linalg.norm(new[:, :-1], axis=1)
            norm[norm == 0] = 1
            new[:, :-1] = (new[:, :-1].T / norm).T
            return new
        else:  # data does not contain label column
            norm = np.linalg.norm(new, axis=1)
            norm[norm == 0] = 1
            new = (new.T / norm).T
            return new

    def normalizeColumns(self, data):
        new = data.copy()
        if self.featureNorms is None:
            self.featureNorms = np.linalg.norm(new[:, :-1], axis=0).reshape((self.x_dim, 1))
            self.featureNorms[self.featureNorms == 0] = 1
        if new.shape[1] == self.x_dim + 1:
            new[:, :-1] = (new[:, :-1].T / self.featureNorms).T
            return new
        else:  # data does not contain label column
            new = (new.T / self.featureNorms).T
            return new

    def train(self, optimizer="batch", eps=0.001, lmb=0.1, num_iters=10000, verbose=True, plot_acc=True,
              plot_title="Cost vs Number of Iterations"):
        """Trains the model on the specified optimizer and returns the final cost after num_iters iterations."""
        assert optimizer in ["batch", "stochastic",
                             "decay"], f"Only batch and stochastic gradient descent are allowed! Got: {optimizer}"
        self.w = np.zeros(self.x_dim + 1)
        x = self.normalized_training_data[:, :-1]
        y = self.normalized_training_data[:, -1]

        # print out progress 10 times:
        print_interval = int(num_iters / 10)
        if print_interval == 0: print_interval = 1

        # calculate scoring at 50 evenly spaced intervals
        scoring_interval = int(num_iters / 50)
        if scoring_interval == 0: scoring_interval = 1
        scoringIters = []
        scoringCosts = []
        scoringAccs = []

        gradientFunc = self.batchGradient
        if optimizer in ["stochastic", "decay"]:
            gradientFunc = self.stochasticGradient
        for i in range(num_iters):
            epsAti = eps
            if optimizer == "decay":
                epsAti = (eps * num_iters) / (i + 1)
            gradient = gradientFunc(self.w, epsAti, lmb, x, y)
            self.w += gradient
            if verbose and i % print_interval == 0 and i != 0:
                s = f"Cost at iteration {i}: {self.cost(self.w, lmb, x, y)}"
                s += f" (training accuracy: {round(self.accuracy(self.raw_training_data), 4)})"
                print(s)
            if i % scoring_interval == 0:
                scoringIters.append(i)
                scoringCosts.append(self.cost(self.w, lmb, x, y))
                scoringAccs.append(self.accuracy(self.raw_training_data))
        if verbose: self.plotAccuracy(scoringIters, scoringCosts, scoringAccs, plot_acc, plot_title)
        return self.cost(self.w, lmb, x, y)

    def plotAccuracy(self, iters, costs, accs, plot_acc, title, x_label="Iteration Number", y_label="Training Accuracy",
                     x_scale="linear"):
        # referenced https://matplotlib.org/gallery/api/two_scales.html
        fig, ax1 = plt.subplots()
        ax1.set_xlabel(x_label)
        ax1.set_ylabel('Cost', color="tab:blue")
        plt.plot(iters, costs, color="tab:blue", label="Cost")
        ax1.tick_params(axis='y', labelcolor="tab:blue")
        if plot_acc:
            ax2 = ax1.twinx()
            ax2.set_ylabel(y_label, color="tab:green")
            ax2.plot(iters, accs, color="tab:green", label=y_label)
            ax2.tick_params(axis='y', labelcolor="tab:green")
            ax2.set_xscale(x_scale)
        fig.tight_layout()
        plt.title(title)
        ax1.set_xscale(x_scale)
        plt.show()

    def testBestCutoff(self, data, cutoffs_to_test=np.linspace(0, 1, 1000)):
        best_c = 0
        best_acc = 0
        for c in cutoffs_to_test:
            c_acc = self.accuracy(data, c)
            if c_acc > best_acc:
                best_c = c
                best_acc = c_acc
        print(f"The cutoff that maximizes accuracy is: {best_c} (accuracy: {best_acc})")
        return best_c

    def accuracy(self, data, cutoff=0.5):
        "Pass in UN normalized data as X!"
        data = self.normalize(data)
        x, y = data[:, :-1], data[:, -1]
        pred = self.sigmoid(x, self.w)
        labels = pred > cutoff
        return np.average(labels == y)
        err = np.linalg.norm(y - self.sigmoid(x, self.w))
        return err

    def predict(self, data, cutoff=0.5):
        data = self.normalize(data)
        pred = self.sigmoid(data, self.w)
        labels = pred > cutoff
        return labels.astype(int)

    def sigmoid(self, x, w):
        return np.clip(expit(x @ w), 1e-15, 1 - 1e-15)

    def cost(self, w, lmb, x, y):
        res = (lmb * np.linalg.norm(w) ** 2) / 2
        res -= y @ np.log(self.sigmoid(x, w))
        pred = self.sigmoid(x, w)
        logPred = np.log(1 - pred)
        res -= (1 - y) @ logPred
        return res

    def batchGradient(self, w, eps, lmb, x, y):
        return -(eps * ((lmb * w) - x.T @ (y - self.sigmoid(x, w))))

    def stochasticGradient(self, w, eps, lmb, x, y):
        randomI = np.random.randint(0, len(x) - 1)
        randomX = x[[randomI], :]
        randomY = y[[randomI]]
        return self.batchGradient(w, eps, lmb, randomX, randomY)

    def calculateFolds(self, data, num_folds):
        idxs = np.arange(len(data))
        np.random.shuffle(idxs)
        IDXGroups = np.array_split(idxs, num_folds)
        return np.array(IDXGroups)

    def getFold(self, data, fold):
        mask = np.ones(len(data), np.bool)
        mask[fold] = 0
        val = data[fold]
        train = data[mask]
        return train, val

    def crossValidate(self, eps=[10 ** i for i in range(-10, 1)], lmb=0.01, optimizer="batch", num_folds=5,
                      num_iters=10000):
        """Cross validate a parameter for the model.
        You must specify a set eps or lmb, and set the other one to None.
        The parameter set to None will be tested.
        """
        assert isinstance(eps, Iterable) or isinstance(lmb,
                                                       Iterable), "You must specify a list of values for either epsilon or lambda!"
        assert not (isinstance(eps, Iterable) and isinstance(lmb,
                                                             Iterable)), "You cannot set both epsilon and lambda for cross validation!"
        param = "epsilon"
        if not isinstance(eps, Iterable):
            eps = [eps] * len(lmb)
            param = "lambda"
        else:
            lmb = [lmb] * len(eps)
            if optimizer == "decay":
                param = "theta"
        data = self.normalized_training_data
        folds = self.calculateFolds(data, num_folds)
        all_costs = []
        all_accs = []
        best_cost = float("inf")
        best_acc = float("inf")
        best_eps = float("inf")
        best_lmb = float("inf")

        def params(eps, lmb):
            if param in ["epsilon", "theta"]:
                if param == "theta":
                    return f"[{param}={num_iters}*{eps}]"
                return f"[{param}={eps}]"
            return f"[lambda={lmb}]"

        for eps_l, lmb_l in zip(eps, lmb):
            costs = []
            accs = []
            for fold in folds:
                train, val = self.getFold(data, fold)
                classifier = LogisticClassifier(train)
                train_cost = classifier.train(optimizer, eps_l, lmb_l, num_iters, verbose=False)
                val_n = classifier.normalize(val)
                cost = classifier.cost(classifier.w, 0, val_n[:, :-1], val_n[:, -1])
                val_acc = classifier.accuracy(val)
                costs.append(cost)
                accs.append(val_acc)
            avgCost = np.average(costs)
            avgAcc = np.average(accs)
            all_costs.append(avgCost)
            all_accs.append(avgAcc)
            print(f"{params(eps_l, lmb_l)} Cross-validated cost: {avgCost} (accuracy: {round(avgAcc, 4)})")
            if avgCost < best_cost:
                best_cost = avgCost
                best_acc = avgAcc
                best_eps = eps_l
                best_lmb = lmb_l
        print(
            f"\nCross-validation complete! The best hyperparameter is: {params(best_eps, best_lmb)} (cost: {round(best_cost, 4)}, accuracy: {round(best_acc, 4)})")
        paramToPlot = eps
        if param == "lambda":
            paramToPlot = lmb
        self.plotAccuracy(paramToPlot, all_costs, all_accs, False,
                          f"Cross-Validation of {param}",
                          param,
                          "Validation Accuracy",
                          "log")
        if param in ["epsilon", "theta"]:
            return best_eps
        return best_lmb



class RidgeRegression:
    def __init__(self, inputFunction, outputFunction, add_bias=True):
        """This is a ridge regression classifier.
        :param inputFunction: A function whose only input is an nxd dimensional dataset and outputs a dynamically standardized dataset
        :param outputFunction: A function that reverses the inputFunction
        # :param loss_metric: How to compute the loss metric. {mae:mean absolute error}
        # :param optimizer: The optimizer to use on gradient descent.
        :param add_bias: whether or not to add a bias term. Default True.

        Calling reverse_normalization_function(normalization_function(data)) must equal data.

        **Important** All data in this model should be passed in NON-normalized. Normalization will happen automatically using the provided functions."""
        # assert loss_metric.lower() in ["mae"], f"Invalid loss_metric. Got {loss_metric}"
        # assert optimizer in ["batch"], f"Invalid optimizer. Got {optimizer}"
        self.normalize = inputFunction
        self.reverse_normalize = outputFunction
        # self.loss_metric = loss_metric.lower()
        # self.optimizer = optimizer.lower()
        self.x_dim = -1
        self.add_bias = add_bias

    def __checkData__(self, data):
        assert data.shape[1] in [self.x_dim, self.x_dim + 1], f"Incorrect data dimension. Expected {self.x_dim} or {self.x_dim - 1} dimensions but got {data.shape[1]}"

    def __addBias__(self, data):
        """Adds a bias term to the data in the first dimension"""
        self.__checkData__(data)
        return np.hstack((np.ones((len(data), 1)), data))

    def __normalize__(self, data):
        self.__checkData__(data)
        data = data.copy()
        data = self.normalize(data)
        if self.add_bias:
            data = self.__addBias__(data)
        return data

    def __reverseNormalize__(self, data):
        self.__checkData__(data)
        data = data.copy()
        if self.add_bias:
            data = data[:,1:]
        data = self.reverse_normalize(data)
        return data

    # def train(self, training_data, epsilon=0.001, lmb=0.1, num_iters=10000, convergence_delta = 1e-10, print_interval=100):
    def train(self, training_data, lmb = 1e-5):
        """Pass in training_data as an nxd array where the last column is the column we are trying to predict.
        Gradient descent will converge until change in loss is < convergence_delta, OR num_iters. Whichever comes first.
        :param training_data: the data to train the model on.
        # :param epsilon: the learning rate.
        :param lmb: The regularization parameter. A higher lmb will encourage regularization.
        # :param num_iters: The maximum number of iterations to perform.
        # :param convergence_delta: When the change in the loss sqrt(MSE) is < convergence_delta, gradient descent will stop,
        # :param print_interval: Print the loss every print_interval iterations. For quiet training, set print_interval=-1"""
        self.x_dim = training_data.shape[1] - 1
        data = self.__normalize__(training_data)
        x = data[:,:-1]
        y = data[:,-1]

        xTx = (x.T @ x) + (lmb * np.eye(x.shape[1]))
        self.w = np.linalg.solve(xTx, x.T @ y)

    def predict(self, data):
        """Take in one single datapoint and predict the output from it."""
        if len(data.shape) == 1:
            data = np.array([data])
        data = self.__normalize__(data)
        return self.reverse_normalize(data @ self.w)

    # def error(self, data, metric="mae"):
    #     """Calculate the error of the model using the specified metric.
    #     :param metric: must be either 'mae' (mean abs error) or 'mse' (mean sqrd error) """
    #     metric = metric.lower()
    #     assert metric in ["mse", "mae"], f"Imvalid error metric. Got: {metric}"










