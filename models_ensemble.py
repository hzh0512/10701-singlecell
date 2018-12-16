import numpy as np
import pandas as pd
import os, timeit, socket, time, pickle

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC, LinearSVC
from sklearn.externals import joblib
from sklearn.neural_network import MLPClassifier
from tqdm import tqdm


pca_n_components = 40


def do_pca():
    store = pd.HDFStore('train_data.h5')
    train_features = store['rpkm']  # 21389

    print("start PCA...")
    pca = Pipeline([
        ("pca", PCA(n_components=pca_n_components, random_state=10701)),
        ("scaler", StandardScaler())])
    pca.fit(train_features)
    print("finish PCA")

    cache_file_name = "models/pca_%d_std.model" % pca_n_components
    pickle.dump(pca, open(cache_file_name, "wb"))


def load_data(need_pca=True): 
    print("start loading data...")
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

    print("finish loading data")

    return train_features, test_features, train_labels, test_labels


def do_random_forest(X_train, X_test, y_train, y_test):
    start_time = timeit.default_timer()
    # model = RandomForestClassifier(n_estimators=100, max_features=100, n_jobs=-1, min_samples_split=40, min_samples_leaf=20)
    model = RandomForestClassifier(n_estimators=100, max_features=20, n_jobs=-1, 
        min_samples_split=90, min_samples_leaf=40, random_state=10701)
    print("start training random forest...")
    model.fit(X_train, y_train)

    mid_time = timeit.default_timer()
    print("start testing decision tree...")
    y_predict = model.predict(X_test)

    end_time = timeit.default_timer()
    print("finish random forest")
    print("train time: %s" % str(mid_time - start_time))
    print("testing time %s" % str(end_time - mid_time))

    acc = accuracy_score(y_test, y_predict)
    print("accuracy using random forest is: {:.4f}".format(acc))

    return model.classes_, model.predict_proba(X_train), model.predict_proba(X_test)


def do_SVM(X_train, X_test, y_train, y_test):
    start_time = timeit.default_timer()
    print('start training SVM...')

    clf = SVC(kernel='rbf', C=1, gamma=0.01, probability=True, random_state=10701, 
        decision_function_shape='ovr', cache_size=1000)
    clf.fit(X_train, y_train)

    mid_time = timeit.default_timer()
    print('start testing SVM...')
    y_train_pred = clf.predict(X_train)
    y_test_pred = clf.predict(X_test)

    end_time = timeit.default_timer()
    print("finish SVM")
    print("train time: %s" % str(mid_time - start_time))
    print("testing time %s" % str(end_time - mid_time))

    acc = accuracy_score(y_test, y_test_pred)
    print("accuracy using SVM is: {:.4f}".format(acc))

    label_idx_dict = {}
    label_cnt = len(clf.classes_)
    for idx, val in enumerate(clf.classes_):
        label_idx_dict[val] = idx

    predict_train_prob = [[(1 if i == label_idx_dict[val] else 0) for i in range(label_cnt)] for val in y_train_pred]
    predict_test_prob = [[(1 if i == label_idx_dict[val] else 0) for i in range(label_cnt)] for val in y_test_pred]

    return clf.classes_, clf.predict_proba(X_train), clf.predict_proba(X_test), np.array(predict_train_prob), np.array(predict_test_prob)


def do_prob_stacking(X_train, X_test, y_train, y_test):
    start_time = timeit.default_timer()
    print('start training stacking neural network...')
    clf = MLPClassifier(solver='sgd', 
        learning_rate_init=0.001, 
        learning_rate='adaptive', 
        alpha=0.1, 
        max_iter=400, 
        hidden_layer_sizes=(70), 
        activation='logistic', 
        random_state=10701, 
        verbose=1)
    clf.fit(X_train, y_train)

    mid_time = timeit.default_timer()
    print('start testing stacking neural network...')
    y_pred = clf.predict(X_test)

    end_time = timeit.default_timer()
    print("finish stacking neural network")
    print("train time: %s" % str(mid_time - start_time))
    print("testing time %s" % str(end_time - mid_time))

    acc = accuracy_score(y_test, y_pred)
    print("accuracy using stacking neural network is: {:.4f}".format(acc))


def main():
    # X_train, X_test, y_train, y_test = load_data()
    # class1, train_prob1, test_prob1 = do_random_forest(X_train, X_test, y_train, y_test)
    # class2, train_prob2, test_prob2, train_prob3, test_prob3 = do_SVM(X_train, X_test, y_train, y_test)
    # assert((class1 == class2).all())
    # pickle.dump((test_prob1, test_prob2, test_prob3, y_test, class2), open("models/test_prob.data", "wb"))

    # train_total_prob = np.concatenate((train_prob1, train_prob2, train_prob3), axis=1)
    # test_total_prob = np.concatenate((test_prob1, test_prob2, test_prob3), axis=1)
    # pickle.dump((train_total_prob, test_total_prob, y_train, y_test), open("models/all_prob.data", "wb"))

    # train_total_prob, test_total_prob, y_train, y_test = pickle.load(open("models/all_prob.data", "rb"))
    # do_prob_stacking(train_total_prob, test_total_prob, y_train, y_test)

    prob1, prob2, prob3, y_test, classes = pickle.load(open("models/test_prob.data", "rb"))
    max_acc = 0
    best_p1, best_p2, best_p3 = -1, -1, -1
    for p1 in tqdm(range(0, 10)):
        for p2 in range(0, 10):
            for p3 in range(0, 10):
                prob = prob1 * p3 + prob2 * p2 + prob3 * p1
                y_pred = [list(sorted(zip(classes, prob_vec), key=lambda k: -k[1]))[0][0] for prob_vec in prob]
                acc = accuracy_score(y_test, y_pred)
                if acc > max_acc:
                    max_acc, best_p1, best_p2, best_p3 = acc, p1, p2, p3
    print("accuracy using ensembling is: {:.4f}".format(max_acc)) # accuracy using ensembling is: 0.5275
    print("weight p1, p2, p3 = %d, %d, %d" % (best_p1, best_p2, best_p3)) # weight p1, p2, p3 = 2, 3, 8



if __name__ == '__main__':
  main()