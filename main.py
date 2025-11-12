import numpy as np
import pandas as pd
from algorithms import PCAModel, KNN

def f1_score_macro(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    classes = np.unique(np.concatenate((y_true, y_pred)))
    f1_scores = []
    for cls in classes:
        tp = np.sum((y_true == cls) & (y_pred == cls))
        fp = np.sum((y_true != cls) & (y_pred == cls))
        fn = np.sum((y_true == cls) & (y_pred != cls))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        f1_scores.append(f1)
    macro_f1 = np.mean(f1_scores)
    return macro_f1

def accuracy_fn(y_true, y_pred):
    correct = np.equal(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc

def read_data(trainfile='MNIST_train.csv', validationfile='MNIST_validation.csv'):
    dftrain = pd.read_csv(trainfile)
    dfval = pd.read_csv(validationfile)
    featurecols = list(dftrain.columns)
    featurecols.remove('label')
    featurecols.remove('even')
    targetcol = 'label'
    Xtrain = np.array(dftrain[featurecols])
    ytrain = np.array(dftrain[targetcol])
    Xval = np.array(dfval[featurecols])
    yval = np.array(dfval[targetcol])
    return (Xtrain, ytrain, Xval, yval)

Xtrain, ytrain, Xval, yval = read_data()

# PCA + KNN
pca_model = PCAModel(n_components = 30)
pca_model.fit(Xtrain)
Xtrain_pca = pca_model.predict(Xtrain)
Xval_pca = pca_model.predict(Xval)

knn = KNN(n_neighbors=5)
knn.fit(Xtrain_pca, ytrain)

ytrain_pred = knn.predict(Xtrain_pca)
yval_pred = knn.predict(Xval_pca)

train_acc = accuracy_fn(ytrain, ytrain_pred)
val_acc = accuracy_fn(yval, yval_pred)

print("Method: PCA + KNN")
print("Train F1:", f1_score_macro(ytrain, ytrain_pred))
print("Validation F1:", f1_score_macro(yval, yval_pred))
print(f"Training Accuracy: {train_acc:.4f}")
print(f"Validation Accuracy: {val_acc:.4f}")