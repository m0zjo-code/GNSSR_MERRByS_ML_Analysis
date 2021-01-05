#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 19 15:10:22 2020

@author: jonathan
"""
import os, random, sys, glob

# Generate data labels
MLDataLabelNames = ["Ocean", "SeaIce"]

dataFolder = os.path.join(os.getcwd() , '../Data/')
labelLookup = {}
partition = {}
labels = []


trainSize = 0.6
testSize = 0.2

for i in range(0, len(MLDataLabelNames)):
    for filename in os.listdir(os.path.join(dataFolder, MLDataLabelNames[i])):
        if filename.startswith("D-ID-"):
            ID = os.path.splitext(filename)[0]
            labelLookup[ID] = i
            labels.append(ID)

random.shuffle(labels)
cut = int(trainSize*len(labels))
partition["train"] = labels[:int(trainSize*len(labels))]
partition["validation"] = labels[int(trainSize*len(labels)):int(testSize*len(labels))]
partition["test"]  = labels[int(testSize*len(labels)):]

import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from kerasDataGenerator import DataGenerator
from keras.utils import to_categorical

params = {'dim': 2560,
          'batch_size': 20,
          'n_classes': 2,
          'n_channels': 1,
          'shuffle': True}

# Generators
training_generator = DataGenerator(partition['train'], labelLookup, **params)
validation_generator = DataGenerator(partition['validation'], labelLookup, **params)

# Design model
model = Sequential()
model.add(Dense(100, input_dim=params['dim'], activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(2, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='Nadam', metrics=['accuracy'])

model.summary()

# Train model on dataset
history = model.fit(training_generator, validation_data=validation_generator, epochs=30)

import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'], label='train')
plt.legend()
plt.show()

def get_test_data(testData):
    n = len(testData)
    outputArray = np.zeros((n, 2560))
    outputArraytest = np.zeros((n))
    for i in range(0, n):
        lb = labels[i]
        outputArray[i,:] = np.reshape(np.load('../Data/' + MLDataLabelNames[labelLookup[lb]] + '/' + lb + '.npy'), (2560))
        outputArraytest[i] = labelLookup[lb]
    return outputArraytest, outputArray


y_test_e, X_test = get_test_data(partition["test"])


y_test = to_categorical(y_test_e, num_classes=2)

scores = model.evaluate(X_test, y_test, verbose=1)  # Evaluate the trained model on the test set!
y_predict = model.predict(X_test)


from sklearn.metrics import confusion_matrix

conf_matx = confusion_matrix(y_test.argmax(axis=1), y_predict.argmax(axis=1))
print(conf_matx)
print(model.metrics_names)
print(scores)
