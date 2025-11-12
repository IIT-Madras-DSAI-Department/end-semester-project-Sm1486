import numpy as np

# PCA + KNN
class PCAModel:
    def __init__(self, n_components):
        self.n_components = n_components
        self.mean = None
        self.components = None
        self.explained_variance = None
    
    def fit(self, X):
        X = np.array(X, dtype = float)
        self.mean = np.mean(X, axis = 0)
        X_centered = X - self.mean
        cov_matrix = np.cov(X_centered, rowvar = False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        sorted_idx = np.argsort(eigenvalues)[::-1]
        self.explained_variance = eigenvalues[sorted_idx][:self.n_components]
        self.components = eigenvectors[:, sorted_idx][:, :self.n_components]
    
    def predict(self, X):
        if self.mean is None or self.components is None:
            raise ValueError("The PCA model has not been fitted yet")
        X_centered = X - self.mean
        return np.dot(X_centered, self.components)

class KNN:
    def __init__(self, n_neighbors = 5):
        self.n_neighbors = n_neighbors
    def euclidean_distance(self, x1, x2):
        return np.linalg.norm(x1 - x2)
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
    def predict(self, X):
        from scipy.spatial.distance import cdist
        distances = cdist(X, self.X_train)
        idx = np.argsort(distances, axis=1)[:, :self.n_neighbors]
        top_labels = self.y_train[idx]
        from scipy.stats import mode
        y_pred = mode(top_labels, axis=1)[0].ravel()
        return y_pred
    def _predict(self, x):
        distances = np.linalg.norm(self.X_train - x, axis=1)
        n_neighbors_idxs = np.argsort(distances)[:self.n_neighbors]
        labels = self.y_train[n_neighbors_idxs]
        most_occuring_value = np.bincount(labels).argmax()
        return most_occuring_value

# Multinomial Regression
class SoftmaxRegression:
    def __init__(self, learning_rate=0.1, epochs=800):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.W = None
        self.b = None 

    def _softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def _one_hot(self, y, num_classes):
        return np.eye(num_classes)[y]

    def _cross_entropy_loss(self, y_true, y_pred):
        return -np.mean(np.sum(y_true * np.log(y_pred + 1e-15), axis=1))

    def fit(self, X, y):
        num_samples, num_features = X.shape
        num_classes = np.max(y) + 1

        self.W = np.random.randn(num_features, num_classes) * 0.01
        self.b = np.zeros((1, num_classes))
        Y_onehot = self._one_hot(y, num_classes)

        for epoch in range(self.epochs):
            logits = np.dot(X, self.W) + self.b
            probs = self._softmax(logits)
            loss = self._cross_entropy_loss(Y_onehot, probs)
            grad_logits = (1./ num_samples) * (Y_onehot - probs) 
            grad_W = -np.dot(X.T, grad_logits)
            grad_b = -np.sum(grad_logits, axis=0, keepdims=True)
            self.W -= self.learning_rate * grad_W
            self.b -= self.learning_rate * grad_b

    def predict_proba(self, X):
        logits = np.dot(X, self.W) + self.b
        return self._softmax(logits)

    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

# XGBoost
class DecisionTreeRegressor:
    def __init__(self, max_depth=10, min_samples_split=2, feature_indices=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.feature_indices = feature_indices
        self.tree = None

    def fit(self, X, y):
        self.tree = self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth):
        n_samples, n_features = X.shape
        if n_samples < self.min_samples_split or (self.max_depth and depth >= self.max_depth):
            return np.mean(y)

        features = self.feature_indices if self.feature_indices is not None else np.arange(n_features)

        best_mse = float('inf')
        best_split = None

        for feature in features:
            Xf = X[:, feature]
            sort_idx = np.argsort(Xf)
            X_sorted = Xf[sort_idx]
            y_sorted = y[sort_idx]

            if np.all(X_sorted == X_sorted[0]):
                continue

            cumsum = np.cumsum(y_sorted)
            cumsum2 = np.cumsum(y_sorted ** 2)
            total_sum = cumsum[-1]
            total_sum2 = cumsum2[-1]
            n = len(y_sorted)

            for i in range(1, n):
                if X_sorted[i] == X_sorted[i - 1]:
                    continue
                left_n = i
                right_n = n - i
                left_sum = cumsum[i - 1]
                left_sum2 = cumsum2[i - 1]
                right_sum = total_sum - left_sum
                right_sum2 = total_sum2 - left_sum2
                left_var = left_sum2 / left_n - (left_sum / left_n) ** 2
                right_var = right_sum2 / right_n - (right_sum / right_n) ** 2
                mse = (left_n / n) * left_var + (right_n / n) * right_var

                if mse < best_mse:
                    best_mse = mse
                    best_split = {
                        'feature': feature,
                        'threshold': (X_sorted[i] + X_sorted[i - 1]) / 2.0
                    }

        if best_split is None:
            return np.mean(y)

        feature = best_split['feature']
        threshold = best_split['threshold']
        left_idx = X[:, feature] <= threshold
        right_idx = ~left_idx

        left_tree = self._build_tree(X[left_idx], y[left_idx], depth + 1)
        right_tree = self._build_tree(X[right_idx], y[right_idx], depth + 1)

        return {
            'feature': feature,
            'threshold': threshold,
            'left': left_tree,
            'right': right_tree
        }

    def predict(self, X):
        return np.array([self._predict_one(x, self.tree) for x in X])

    def _predict_one(self, x, tree):
        if not isinstance(tree, dict):
            return tree
        if x[tree['feature']] <= tree['threshold']:
            return self._predict_one(x, tree['left'])
        else:
            return self._predict_one(x, tree['right'])

class XGBoostMultiClass:
    def __init__(self, n_estimators=10, max_depth=10, min_samples_split=2, learning_rate=0.1, max_features=None, n_classes=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.learning_rate = learning_rate
        self.max_features = max_features
        self.n_classes = n_classes
        self.estimators = [] 

    def softmax(self, x):
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / np.sum(e_x, axis=1, keepdims=True)

    def grad(self, preds, labels):
        probs = self.softmax(preds)
        grad = probs.copy()
        grad[np.arange(len(labels)), labels] -= 1
        return grad

    def fit_tree(self, X, grad_col):
        n_features = X.shape[1]
        if self.max_features is not None:
            feature_indices = np.random.choice(n_features, self.max_features, replace=False)
        else:
            feature_indices = np.arange(n_features)
        tree = DecisionTreeRegressor(max_depth=self.max_depth, min_samples_split=self.min_samples_split,
                                        feature_indices=feature_indices)
        tree.fit(X, -grad_col)
        return tree

    def fit(self, X, y):
        if self.n_classes is None:
            self.n_classes = len(np.unique(y))

        n_samples = X.shape[0]
        y_pred = np.zeros((n_samples, self.n_classes))

        for _ in range(self.n_estimators):
            grad = self.grad(y_pred, y)
            trees = []
            for k in range(self.n_classes):
                tree = self.fit_tree(X, grad[:, k])
                update = self.learning_rate * tree.predict(X)
                y_pred[:, k] += update
                trees.append(tree)
            self.estimators.append(trees)

    def predict(self, X):
        n_samples = X.shape[0]
        y_pred = np.zeros((n_samples, self.n_classes))
        for trees in self.estimators:
            for k, tree in enumerate(trees):
                y_pred[:, k] += self.learning_rate * tree.predict(X)
        probs = self.softmax(y_pred)
        return np.argmax(probs, axis=1)

# Ensemble
class EnsembleSoftmaxXGBoost:
    def __init__(self, reg_model, xgb_model, weights=(0.5, 0.5)):
        self.reg_model = reg_model
        self.xgb_model = xgb_model
        self.weights = weights

    def predict_proba(self, X):
        reg_probs = self.reg_model.predict_proba(X)
        n_classes = self.xgb_model.n_classes
        n_samples = X.shape[0]
        y_pred_xgb = np.zeros((n_samples, n_classes))
        for trees in self.xgb_model.estimators:
            for k, tree in enumerate(trees):
                y_pred_xgb[:, k] += self.xgb_model.learning_rate * tree.predict(X)
        xgb_probs = self.xgb_model.softmax(y_pred_xgb)
        w1, w2 = self.weights
        combined_probs = w1 * reg_probs + w2 * xgb_probs
        return combined_probs / (w1 + w2)

    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)