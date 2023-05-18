from collections import Counter

import numpy as np
import pandas as pd

# set random seed
np.random.seed(0)

"""
Tips for debugging:
- Use `print` to check the shape of your data. Shape mismatch is a common error.
- Use `ipdb` to debug your code
    - `ipdb.set_trace()` to set breakpoints and check the values of your variables in interactive mode
    - `python -m ipdb -c continue hw3.py` to run the entire script in debug mode. Once the script is paused, you can use `n` to step through the code line by line.
"""


# 1. Load datasets
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    DO NOT MODIFY THIS FUNCTION.
    """
    # Load iris dataset
    iris = pd.read_csv(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
        header=None,
    )
    iris.columns = [
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width",
        "class",
    ]

    # Load Boston housing dataset
    boston = pd.read_csv(
        "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
    )

    return iris, boston


# 2. Preprocessing functions
def train_test_split(
    df: pd.DataFrame, target: str, test_size: float = 0.3
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Shuffle and split dataset into train and test sets
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    split_idx = int(len(df) * (1 - test_size))
    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]

    # Split target and features
    X_train = train.drop(target, axis=1).values
    y_train = train[target].values
    X_test = test.drop(target, axis=1).values
    y_test = test[target].values

    return X_train, X_test, y_train, y_test


def normalize(X: np.ndarray) -> np.ndarray:
    # Normalize features to [0, 1]
    # You can try other normalization methods, e.g., z-score, etc.
    # TODO: 1%
    std_arr = np.std(X, axis=0)
    mean_arr = np.mean(X, axis=0)
    # Z-score
    ret = X.copy()
    size = len(std_arr)
    ret = (ret - mean_arr) / std_arr
    '''
    for flow in ret:
        for i in range(size):
            flow[i] = (flow[i] - mean_arr[i]) / std_arr[i]
    '''
    max_arr = np.max(ret, axis=0)
    min_arr = np.min(ret, axis=0)
    ret = (ret - min_arr) / (max_arr - min_arr)
    '''
    for flow in ret:
        for i in range(size):
            flow[i] = (flow[i] - min_arr[i]) / (max_arr[i] - min_arr[i])
    '''
    return ret
    raise NotImplementedError


code_table = dict()
num_class = 0


def encode_labels(y: np.ndarray) -> np.ndarray:
    """
    Encode labels to integers.
    """
    # TODO: 1%
    ret = list()
    global code_table
    global num_class
    for cls in y:
        if cls in code_table:
            ret.append(code_table[cls])
        else:
            ret.append(num_class)
            code_table[cls] = num_class
            code_table[num_class] = cls
            # doubly
            num_class += 1
    return ret
    raise NotImplementedError


# 3. Models
class LinearModel:
    def __init__(
        self, learning_rate=0.01, iterations=1000, model_type="linear"
    ) -> None:
        self.learning_rate = learning_rate
        self.iterations = iterations
        # You can try different learning rate and iterations
        self.model_type = model_type
        self.weights = None
        self.bias = 0
        self.n_features = None
        self.n_classes = None

        assert model_type in [
            "linear",
            "logistic",
        ], "model_type must be either 'linear' or 'logistic'"

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        X = np.insert(X, 0, 1, axis=1)
        # bias included
        self.n_classes = len(np.unique(y))
        self.n_features = X.shape[1]
        # print("n_features:", self.n_features)
        if self.model_type == "logistic":
            self.weights = np.zeros((self.n_features, self.n_classes))
            self.bias = np.zeros(self.n_classes)
            for i in range(self.iterations):
                grad = self._compute_gradients(X, y)
                self.weights -= self.learning_rate * grad
            return
        else:
            self.weights = np.zeros(self.n_features)
            for i in range(self.iterations):
                grad = self._compute_gradients(X, y)
                self.weights -= self.learning_rate * grad
            return
            pass
        # TODO: 2%
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.insert(X, 0, 1, axis=1)
        if self.model_type == "linear":
            # TODO: 2%
            ret = np.dot(X, self.weights)
            return ret
            raise NotImplementedError
        elif self.model_type == "logistic":
            # TODO: 2%
            fuck = np.dot(X, self.weights) + self.bias
            ret = np.zeros(X.shape[0])
            for i in range(X.shape[0]):
                ret[i] = np.argmax(fuck[i])
            return ret
            raise NotImplementedError

    def one_hot_encode(self, y: np.ndarray) -> np.ndarray:
        ret = np.zeros((len(y), self.n_classes))
        for i in range(len(y)):
            ret[i][y[i]] = 1
        return ret

    def _compute_gradients(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        if self.model_type == "linear":
            # TODO: 3%
            now_vals = np.dot(X, self.weights)
            ret = np.zeros(X.shape[1])
            for i in range(X.shape[1]):
                ret[i] = np.sum((now_vals - y) * X[:, i]) * (1 / len(y))
            return ret
            raise NotImplementedError
        elif self.model_type == "logistic":
            # TODO: 3%
            prob = self._softmax(np.dot(X, self.weights) + self.bias)
            y_hot = self.one_hot_encode(y)
            weight_grad = (1 / self.n_features) * np.dot(X.T, prob - y_hot)
            bias_grad = (1 / len(y)) * np.sum(prob - y_hot)
            self.bias -= self.learning_rate * bias_grad
            return weight_grad
            raise NotImplementedError

    def _softmax(self, z: np.ndarray) -> np.ndarray:
        exp = np.exp(z)
        return exp / np.sum(exp, axis=1, keepdims=True)


class DecisionTree:
    def __init__(self, max_depth: int = 5, model_type: str = "classifier"):
        self.max_depth = max_depth
        self.model_type = model_type
        self.n_features = None
        self.n_classes = None
        self.which_features = None
        assert model_type in [
            "classifier",
            "regressor",
        ], "model_type must be either 'classifier' or 'regressor'"

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.n_features = X.shape[1]
        self.n_classes = len(np.unique(y))
        #self.max_depth = min(self.max_depth, self.n_features)
        self.tree = self._build_tree(X, y, 0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([self._traverse_tree(x, self.tree) for x in X])

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int) -> dict:
        if depth >= self.max_depth or self._is_pure(y):
            return self._create_leaf(y)

        feature, threshold = self._find_best_split(X, y)
        # TODO: 4%
        if feature == None or threshold == None:
            return self._create_leaf(y)
        node = dict()
        left_x = list()
        left_y = list()
        right_x = list()
        right_y = list()
        for i in range(len(y)):
            if X[i][feature] <= threshold:
                left_x.append(X[i])
                left_y.append(y[i])
            else:
                right_x.append(X[i])
                right_y.append(y[i])
        left_x, left_y = np.array(left_x), np.array(left_y)
        right_x, right_y = np.array(right_x), np.array(right_y)
        # return the node
        node["feature"] = feature
        node["threshold"] = threshold
        node["left"] = self._build_tree(left_x, left_y, depth + 1)
        node["right"] = self._build_tree(right_x, right_y, depth + 1)
        return node
        raise NotImplementedError

        return {
            "feature": feature,
            "threshold": threshold,
            "left": left_child,
            "right": right_child,
        }

    def _is_pure(self, y: np.ndarray) -> bool:
        return len(set(y)) == 1

    def _create_leaf(self, y: np.ndarray):
        if self.model_type == "classifier":
            # TODO: 1%
            cnt = np.zeros(self.n_classes)
            for fuck in y:
                cnt[fuck] += 1
            return np.argmax(cnt)
            raise NotImplementedError
        else:
            # TODO: 1%
            return np.mean(y)
            raise NotImplementedError

    def _find_best_split(self, X: np.ndarray, y: np.ndarray) -> tuple[int, float]:
        best_gini = float("inf")
        best_mse = float("inf")
        best_feature = None
        best_threshold = None

        for feature in range(X.shape[1]):
            sorted_indices = np.argsort(X[:, feature])
            for i in range(1, len(X)):
                if X[sorted_indices[i - 1], feature] != X[sorted_indices[i], feature]:
                    threshold = (
                        X[sorted_indices[i - 1], feature]
                        + X[sorted_indices[i], feature]
                    ) / 2
                    mask = X[:, feature] <= threshold
                    left_y, right_y = np.array(y)[mask], np.array(y)[~mask]

                    if self.model_type == "classifier":
                        gini = self._gini_index(left_y, right_y)
                        if gini < best_gini:
                            best_gini = gini
                            best_feature = feature
                            best_threshold = threshold
                    else:
                        mse = self._mse(left_y, right_y)
                        if mse < best_mse:
                            best_mse = mse
                            best_feature = feature
                            best_threshold = threshold

        return best_feature, best_threshold

    def _gini_index(self, left_y: np.ndarray, right_y: np.ndarray) -> float:
        # TODO: 4%
        total_num = len(left_y) + len(right_y)
        cnt = np.zeros(self.n_classes)
        left_gini = 1
        for fuck in left_y:
            cnt[fuck] += 1
        for fuck in cnt:
            left_gini -= (fuck / len(left_y)) ** 2
        cnt = np.zeros(self.n_classes)
        right_gini = 1
        for fuck in right_y:
            cnt[fuck] += 1
        for fuck in cnt:
            right_gini -= (fuck / len(right_y)) ** 2
        return (len(left_y) / total_num) * left_gini + (
            len(right_y) / total_num
        ) * right_gini
        raise NotImplementedError

    def _mse(self, left_y: np.ndarray, right_y: np.ndarray) -> float:
        # TODO: 4%
        mean = np.mean(left_y)
        left_mean_arr = [mean for i in range(len(left_y))]
        mean = np.mean(right_y)
        right_mean_arr = [mean for i in range(len(right_y))]

        return mean_squared_error(left_y, left_mean_arr) * len(left_y)/ (len(left_y) + len(right_y)) + mean_squared_error(right_y, right_mean_arr) * len(right_y) / (len(left_y) + len(right_y))

        raise NotImplementedError

    def _traverse_tree(self, x: np.ndarray, node: dict):
        if isinstance(node, dict):
            feature, threshold = node["feature"], node["threshold"]
            if x[feature] <= threshold:
                return self._traverse_tree(x, node["left"])
            else:
                return self._traverse_tree(x, node["right"])
        else:
            return node


class RandomForest:
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 5,
        model_type: str = "classifier",
    ):
        # TODO: 1%
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.model_type = model_type
        self.n_features = None
        self.trees = [
            DecisionTree(model_type=self.model_type) for i in range(n_estimators)
        ]
        self.n_classes = None
        return
        raise NotImplementedError

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        for tree in self.trees:
            # TODO: 2%
            # bootstrap_indices = np.random.choice(
            if self.n_features == None:
                self.n_features = X.shape[1]
                self.n_classes = np.unique(y).shape[0]
            chosen_num = np.random.randint(0, self.n_features)
            chosen_num += 1
            indices = [i for i in range(self.n_features)]
            chosen = np.random.choice(indices, chosen_num, replace=False)
            self.max_depth = min(self.max_depth, len(chosen))
            X_chosen = X[:, chosen]
            X_chosen = np.array(X_chosen)
            tree.which_features = chosen
            tree.fit(X_chosen, y)
            # raise NotImplementedError
        return

    def predict(self, X: np.ndarray) -> np.ndarray:
        # TODO: 2%
        if self.model_type == "classifier":
            votion = np.zeros((X.shape[0], self.n_classes))
            for tree in self.trees:
                result = tree.predict(X[:, tree.which_features])
                for i in range(len(result)):
                    votion[i][result[i]] += 1
            ret = np.zeros(X.shape[0])
            for i in range(X.shape[0]):
                ret[i] = np.argmax(votion[i])
            #print(ret)
            return ret
        else:
            fuck = np.zeros(X.shape[0])
            for tree in self.trees:
                result = tree.predict(X[:, tree.which_features])
                fuck += result
            ret = fuck / len(self.trees)
            return ret
            pass
        # raise NotImplementedError


# 4. Evaluation metrics
def accuracy(y_true, y_pred):
    # TODO: 1%
    amount = len(y_true)
    true_cnt = 0
    for i in range(amount):
        if y_true[i] == y_pred[i]:
            true_cnt += 1
    return true_cnt / amount
    raise NotImplementedError


def mean_squared_error(y_true, y_pred):
    # TODO: 1%
    fuck = y_true - y_pred
    ret = 0
    for i in range(len(fuck)):
        ret += fuck[i] ** 2
    return ret / len(fuck)
    raise NotImplementedError


# 5. Main function
def main():
    iris, boston = load_data()

    # Iris dataset - Classification
    X_train, X_test, y_train, y_test = train_test_split(iris, "class")
    X_train, X_test = normalize(X_train), normalize(X_test)
    y_train, y_test = encode_labels(y_train), encode_labels(y_test)

    logistic_regression = LinearModel(model_type="logistic")
    logistic_regression.fit(X_train, y_train)
    y_pred = logistic_regression.predict(X_test)
    print("Logistic Regression Accuracy:", accuracy(y_test, y_pred))

    decision_tree_classifier = DecisionTree(model_type="classifier")
    decision_tree_classifier.fit(X_train, y_train)
    y_pred = decision_tree_classifier.predict(X_test)
    print("Decision Tree Classifier Accuracy:", accuracy(y_test, y_pred))

    random_forest_classifier = RandomForest(model_type="classifier")
    random_forest_classifier.fit(X_train, y_train)
    y_pred = random_forest_classifier.predict(X_test)
    print("Random Forest Classifier Accuracy:", accuracy(y_test, y_pred))

    # Boston dataset - Regression
    X_train, X_test, y_train, y_test = train_test_split(boston, "medv")
    X_train, X_test = normalize(X_train), normalize(X_test)

    linear_regression = LinearModel(model_type="linear")
    linear_regression.fit(X_train, y_train)
    y_pred = linear_regression.predict(X_test)
    print("Linear Regression MSE:", mean_squared_error(y_test, y_pred))

    decision_tree_regressor = DecisionTree(model_type="regressor")
    decision_tree_regressor.fit(X_train, y_train)
    y_pred = decision_tree_regressor.predict(X_test)
    print("Decision Tree Regressor MSE:", mean_squared_error(y_test, y_pred))

    random_forest_regressor = RandomForest(model_type="regressor")
    random_forest_regressor.fit(X_train, y_train)
    y_pred = random_forest_regressor.predict(X_test)
    print("Random Forest Regressor MSE:", mean_squared_error(y_test, y_pred))


if __name__ == "__main__":
    main()
