# multi-class classification with Keras
import pandas
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
# load dataset
dataframe = pandas.read_csv("/home/jonathan/gnssr/GNSSR_MERRByS_ML_Analysis/IceTrainingDataV1.csv", header=None)
dataset = dataframe.values
X = dataset[:,0:8].astype(float)
Y = dataset[:,8]

X_max = np.max(X, axis = 0)
X_min = np.min(X, axis = 0)

X = (X - X_min) / (X_max - X_min)
print(X)

# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)

# define baseline model
def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(50, input_dim=8, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='Nadam', metrics=['accuracy'])
    return model

print("Starting Training")
estimator = KerasClassifier(build_fn=baseline_model, epochs=100, batch_size=100, verbose=1)
kfold = KFold(n_splits=100, shuffle=True)
results = cross_val_score(estimator, X, dummy_y, cv=kfold)
print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))