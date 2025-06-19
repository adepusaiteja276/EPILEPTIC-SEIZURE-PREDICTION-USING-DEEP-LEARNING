import tkinter as tk
from tkinter import filedialog, Text
import pandas as pd
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential, Model
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical
from sklearn import svm

# Initialize main window
main = tk.Tk()
main.title("Epileptic Seizures Prediction Using Deep Learning")
main.geometry("1200x1200")

# Global variables
X_train, X_test, y_train, y_test, cnn_model, svm_model, scaler = [None] * 7

def uploadDataset():
    global dataset, pathlabel
    filename = filedialog.askopenfilename(initialdir="Dataset")
    dataset = pd.read_csv(filename)
    pathlabel.config(text=f"{filename} Dataset Loaded")
    displayDataInfo(dataset)

def displayDataInfo(dataset):
    text.delete('1.0', tk.END)
    text.insert(tk.END, str(dataset.head()) + "\n\n")
    target = dataset['labels']
    unique, count = np.unique(target, return_counts=True)
    plt.bar(np.arange(len(count)), count, tick_label=['Normal', 'Seizure'])
    plt.show()

def preprocessDataset():
    global X, Y, scaler, X_train, X_test, y_train, y_test
    dataset.fillna(0, inplace=True)
    X, Y = dataset.iloc[:, :-1].values, dataset.iloc[:, -1].values
    X = StandardScaler().fit_transform(X)
    X = X.reshape(X.shape[0], 20, 20, 3)  # Reshape for CNN
    X_train, X_test, y_train, y_test = train_test_split(X, to_categorical(Y), test_size=0.2)

def createCNNModel():
    model = Sequential([
        Convolution2D(32, (3, 3), input_shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3]), activation='relu'),
        MaxPooling2D((2, 2)),
        Convolution2D(32, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(256, activation='relu'),
        Dense(y_train.shape[1], activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def runCNNSVM():
    global cnn_model, svm_model
    cnn_model = createCNNModel()
    if not os.path.exists("model_weights.hdf5"):
        checkpoint = ModelCheckpoint("model_weights.hdf5", save_best_only=True, verbose=1)
        cnn_model.fit(X_train, y_train, batch_size=16, epochs=30, validation_data=(X_test, y_test), callbacks=[checkpoint])
    cnn_model.load_weights("model_weights.hdf5")
    cnn_features = Model(cnn_model.inputs, cnn_model.layers[-2].output).predict(X_train)
    svm_model = svm.SVC().fit(cnn_features, y_train)
    predict = svm_model.predict(cnn_features)
    calculateMetrics(predict)

def calculateMetrics(predict):
    acc = accuracy_score(y_test, predict) * 100
    p = precision_score(y_test, predict, average='macro') * 100
    r = recall_score(y_test, predict, average='macro') * 100
    f = f1_score(y_test, predict, average='macro') * 100
    text.insert(tk.END, f"Accuracy: {acc}\nPrecision: {p}\nRecall: {r}\nF1 Score: {f}\n")
    plotConfusionMatrix(predict)

def plotConfusionMatrix(predict):
    conf_matrix = confusion_matrix(y_test.argmax(axis=1), predict)
    sns.heatmap(conf_matrix, annot=True, cmap="viridis", fmt="g")
    plt.show()

def predictEpilepsy():
    filename = filedialog.askopenfilename(initialdir="Dataset")
    test_data = pd.read_csv(filename).values
    test_data = StandardScaler().fit_transform(test_data)
    test_data = test_data.reshape(test_data.shape[0], 20, 20, 3)
    features = cnn_model.predict(test_data)
    predictions = svm_model.predict(features)
    displayPredictions(predictions, test_data)

def displayPredictions(predictions, test_data):
    labels = ['Normal', 'Seizure']
    for i, pred in enumerate(predictions):
        text.insert(tk.END, f"Test Data: {test_data[i]} => Predicted: {labels[pred]}\n")

# Layout
title = tk.Label(main, text='Epileptic Seizures Prediction Using Deep Learning', bg='gold', fg='black', font=('times', 14, 'bold'))
title.place(x=5, y=5, width=1200, height=50)

uploadButton = tk.Button(main, text="Upload Dataset", command=uploadDataset, font=('times', 12, 'bold'))
uploadButton.place(x=50, y=100)

preprocessButton = tk.Button(main, text="Preprocess Dataset", command=preprocessDataset, font=('times', 12, 'bold'))
preprocessButton.place(x=50, y=150)

cnnsvmButton = tk.Button(main, text="Run CNN + SVM", command=runCNNSVM, font=('times', 12, 'bold'))
cnnsvmButton.place(x=50, y=200)

predictButton = tk.Button(main, text="Predict Epilepsy", command=predictEpilepsy, font=('times', 12, 'bold'))
predictButton.place(x=50, y=250)

exitButton = tk.Button(main, text="Exit", command=main.quit, font=('times', 12, 'bold'))
exitButton.place(x=50, y=300)

pathlabel = tk.Label(main, font=('times', 12, 'bold'))
pathlabel.place(x=500, y=100)

text = Text(main, height=25, width=90, font=('times', 12, 'bold'))
text.place(x=450, y=150)

main.config(bg='lightblue')
main.mainloop()