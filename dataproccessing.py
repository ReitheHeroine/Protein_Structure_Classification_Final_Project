from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import tensorflow as tf
from tensorboard.compat.tensorflow_stub import string
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Conv2D, MaxPooling1D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
import numpy as np

def classification(imported_data):
    x0 = imported_data['alphaAcidsAverage']
    x1 = imported_data['betaAcidsAverage']
    x2 = imported_data['cohumuloneAverage']
    hop_labels = imported_data['hopType']
    data = np.concatenate((x0,x1,x2), axis=0)
    data_labels = []

    for label in hop_labels:
        if label == 'bitter':
            data_labels.append(0)
        elif label == 'aroma':
            data_labels.append(1)
        elif label == 'dual':
            data_labels.append(2)

    cnn(data,data_labels)

def cnn(data, labels, rs=42, epochs=10, batch_size=16):
    N = max([len(data_point) for data_point in data])

    X_train, X_test, y_train, y_test = train_test_split(data,labels,test_size=0.25,random_state=rs)

    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    model = Sequential([
        Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(N, 1)),
        MaxPooling1D(pool_size=2),
        Conv1D(filters=32, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(32, activation='relu'),
        Dense(3, activation='softmax')
    ])

    opt = Adam(learning_rate=1e-3)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    model.summary()
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
    print("Finish training")

    y_pred_prob = model.predict(X_test)
    print("y_test:", y_test)
    print("y_pred_prob:", y_pred_prob)

    y_pred, truth = [], []
    for prob in y_pred_prob:
        y_pred.append(np.argmax(prob))
    for prob in y_test:
        truth.append(np.argmax(prob))

    my_evaluate(truth, y_pred)


def my_evaluate(truth, y_pred):
    print("Accuracy:", accuracy_score(truth, y_pred) * 100, "%")