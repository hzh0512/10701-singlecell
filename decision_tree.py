import numpy as np
import pandas as pd
import pickle
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


pca_n_components = 10


def do_pca():
    store = pd.HDFStore('train_data.h5')
    train_features = store['rpkm']  # 21389

    print("start PCA")
    pca = Pipeline([
        ("pca", PCA(n_components=pca_n_components, random_state=10701)),
        ("scaler", StandardScaler())])
    pca.fit(train_features)
    print("finish PCA")

    cache_file_name = "models/pca_%d_std.model" % pca_n_components
    pickle.dump(pca, open(cache_file_name, "wb"))


def load_data(need_pca=True): 
    store = pd.HDFStore('train_data.h5')
    train_features = store['rpkm']  # 21389
    train_labels = store['labels']
    store.close()

    store = pd.HDFStore('test_data.h5')
    test_features = store['rpkm']   # 2855
    test_labels = store['labels']
    store.close()

    pca = None
    if need_pca:
        cache_file_name = "models/pca_%d_std.model" % pca_n_components
        if not os.path.exists(cache_file_name):
            do_pca()
        pca = pickle.load(open(cache_file_name, "rb"))

        train_features = pca.transform(train_features)
        test_features = pca.transform(test_features)
        print("finish PCA transformation")

    return train_features, test_features, train_labels, test_labels


def do_decision_tree(X_train, X_test, y_train, y_test):
    model = DecisionTreeClassifier(min_samples_split=90, min_samples_leaf=40)
    print("start training decision tree")
    model.fit(X_train, y_train)
    print("finish training decision tree")
    print("start testing decision tree")
    y_predict = model.predict(X_test)
    print("finish testing decision tree")
    print("start calculating accuracy")
    acc = accuracy_score(y_test, y_predict)
    print("accuracy using decision tree is %g" % acc)


def do_random_forest(X_train, X_test, y_train, y_test):
    # model = RandomForestClassifier(n_estimators=100, max_features=100, n_jobs=-1, min_samples_split=40, min_samples_leaf=20)
    model = RandomForestClassifier(n_estimators=100, max_features=20, n_jobs=-1, min_samples_split=90, min_samples_leaf=40)
    print("start training random forest")
    model.fit(X_train, y_train)
    print("finish training random forest")
    print("start testing decision tree")
    y_predict = model.predict(X_test)
    print("finish testing random forest")
    print("start calculating accuracy")
    acc = accuracy_score(y_test, y_predict)
    print("accuracy using random forest is %g" % acc)


def do_ada_boost(X_train, X_test, y_train, y_test):
    rf_clf = RandomForestClassifier(n_estimators=20, max_features=30, n_jobs=-1, min_samples_split=80, min_samples_leaf=40, verbose=1)
    model = AdaBoostClassifier(rf_clf, n_estimators=10, learning_rate=1.5, algorithm="SAMME")
    print("start training Adaboost")
    model.fit(X_train, y_train)
    print("finish training Adaboost")
    print("start testing decision tree")
    y_predict = model.predict(X_test)
    print("finish testing Adaboost")
    print("start calculating accuracy")
    acc = accuracy_score(y_test, y_predict)
    print("accuracy using Adaboost is %g" % acc)


def do_gradient_boosting(X_train, X_test, y_train, y_test):
    model = GradientBoostingClassifier(n_estimators=100, min_samples_split=80, min_samples_leaf=40, verbose=1)
    print("start training GBDT")
    model.fit(X_train, y_train)
    print("finish training GBDT")
    print("start testing decision tree")
    y_predict = model.predict(X_test)
    print("finish testing GBDT")
    print("start calculating accuracy")
    acc = accuracy_score(y_test, y_predict)
    print("accuracy using GBDT is %g" % acc)


def main():
    do_pca()
    # X_train, X_test, y_train, y_test = load_data()
    # print("number of features = %d" % X_train.shape[1])
    # print("number of training samples = %d" % len(y_train))
    # print("number of testing samples = %d" % len(y_test))
    # do_decision_tree(X_train, X_test, y_train, y_test)     # 0.336252, PCA: 0.300175
    # do_random_forest(X_train, X_test, y_train, y_test)     # 0.408441, PCA: 0.460245
    # do_ada_boost(X_train, X_test, y_train, y_test)         # 5: 0.374431, 50: 0.27
    # do_gradient_boosting(X_train, X_test, y_train, y_test) # 50: 0.327145, 100: 0.370578, 150: 0.334151


if __name__ == '__main__':
    main()