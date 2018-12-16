import numpy as np
import pandas as pd
import pickle
import os
import timeit

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from keras.layers import Input, Dense
from keras.models import Model


encoding_dim = 100


def load_data(): 
    store = pd.HDFStore('train_data.h5')
    train_features = store['rpkm']  # 21389
    store.close()

    store = pd.HDFStore('test_data.h5')
    test_features = store['rpkm']   # 2855
    store.close()

    scaler = StandardScaler().fit(train_features)
    train_features = scaler.transform(train_features)
    test_features = scaler.transform(test_features)

    return train_features, test_features


def do_auto_encoder(X_train, X_test):
    ncol = X_train.shape[1]
    input_dim = Input(shape = (ncol, ))

    encoded1 = Dense(1000, activation = 'relu')(input_dim)
    encoded2 = Dense(encoding_dim, activation = 'relu')(encoded1)
    decoded1 = Dense(1000, activation = 'sigmoid')(encoded2)
    decoded2 = Dense(ncol, activation = 'sigmoid')(decoded1)
    
    print("start training...")
    start_time = timeit.default_timer()
    autoencoder = Model(input = input_dim, output = decoded2)
    autoencoder.compile(optimizer = 'sgd', loss='mean_squared_error', metrics=['mse'])
    autoencoder.fit(X_train, X_train, epochs = 20, batch_size = 100, shuffle = True, validation_data = (X_test, X_test))
    end_time = timeit.default_timer()
    print("training time: %s", str(end_time - start_time))
    
    encoder = Model(input = input_dim, output = encoded2)
    cache_file_name = "models/encoder_%d.model" % encoding_dim
    pickle.dump(encoder, open(cache_file_name, "wb"))


def main():
    X_train, X_test = load_data()
    print("number of features = %d" % X_train.shape[1])
    
    do_auto_encoder(X_train, X_test)


if __name__ == '__main__':
    main()