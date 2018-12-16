import pandas as pd
import os, timeit, socket, time, pickle
from sklearn.svm import SVC, LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV 
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier


data_path = './'
train_file = os.path.join(data_path, 'train_data.h5')
test_file = os.path.join(data_path, 'test_data.h5')

pca_n_components = 40
encoding_method = 'pca'

def load_data(): 
    print('start loading data...')
    train_data = pd.HDFStore(train_file)
    X_train = train_data['rpkm'].values
    y_train = train_data['labels'].values
    train_data.close()

    test_data = pd.HDFStore(test_file)
    X_test = test_data['rpkm'].values
    y_test = test_data['labels'].values
    test_data.close()

    print("start %s transformation..." % encoding_method)
    cache_file_name = "models/%s_%d_std.model" % (encoding_method, pca_n_components)
    encoder = pickle.load(open(cache_file_name, "rb"))
    X_train = encoder.transform(X_train)
    X_test = encoder.transform(X_test)

    return X_train, X_test, y_train, y_test


def do_SVM(X_train, X_test, y_train, y_test):
    start_time = timeit.default_timer()
    print('training...')

    clf = SVC(kernel='rbf', C=1, gamma=0.01, random_state=10701, decision_function_shape='ovr', cache_size=1000)
    clf.fit(X_train, y_train)

    mid_time = timeit.default_timer()
    print('testing...')
    y_pred = clf.predict(X_test)

    end_time = timeit.default_timer()
    print("finish SVM")
    print("train time: %s" % str(mid_time - start_time))
    print("testing time %s" % str(end_time - mid_time))

    acc = accuracy_score(y_test, y_pred)
    print("accuracy: {:.4f}".format(acc))

    # joblib.dump(clf, "models/svm_{:.4f}.pkl".format(acc))


def do_ada_boost(X_train, X_test, y_train, y_test):
    start_time = timeit.default_timer()
    print('training...')
    clf = SVC(kernel='rbf', C=1, gamma=0.01, probability=True, random_state=10701, decision_function_shape='ovr', cache_size=1000)
    model = AdaBoostClassifier(clf, n_estimators=50, algorithm='SAMME.R')
    model.fit(X_train, y_train)

    mid_time = timeit.default_timer()
    print('testing...')
    y_pred = model.predict(X_test)
    
    end_time = timeit.default_timer()
    print("finish Adaboost")
    print("train time: %s" % str(mid_time - start_time))
    print("testing time %s" % str(end_time - mid_time))

    acc = accuracy_score(y_test, y_pred)
    print("accuracy using Adaboost is %g" % acc)

    pickle.dump(mode, "models/Adaboost_50.model")


def do_bagging_boost(X_train, X_test, y_train, y_test):
    start_time = timeit.default_timer()
    print('training...')
    clf = SVC(kernel='rbf', C=1, gamma=0.01, random_state=10701, decision_function_shape='ovr', cache_size=1000)
    bdt = BaggingClassifier(clf)
    bdt.fit(X_train, y_train)

    mid_time = timeit.default_timer()
    print('testing...')
    y_pred = bdt.predict(X_test)

    end_time = timeit.default_timer()
    print("finish bagging boost")
    print("train time: %s" % str(mid_time - start_time))
    print("testing time %s" % str(end_time - mid_time))

    acc = accuracy_score(y_test, y_pred)
    print("accuracy using Adaboost is %g" % acc)


def main():
    X_train, X_test, y_train, y_test = load_data()
    # do_SVM(X_train, X_test, y_train, y_test)
    do_ada_boost(X_train, X_test, y_train, y_test)
    # do_bagging_boost(X_train, X_test, y_train, y_test)


if __name__ == '__main__':
    main()
