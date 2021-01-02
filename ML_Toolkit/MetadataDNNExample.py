# multi-class classification with Keras
import pandas
import numpy as np

import pickle

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from keras.callbacks import TensorBoard, CSVLogger, EarlyStopping

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# load dataset
dataframe = pandas.read_csv("/home/jonathan/gnssr/GNSSR_MERRByS_ML_Analysis/IceTrainingDataV1.csv", header=None)
dataset = dataframe.values
X = dataset[:,0:8].astype(float)
Y = dataset[:,8]

X_max = np.max(X, axis = 0)
X_min = np.min(X, axis = 0)

X = (X - X_min) / (X_max - X_min)

# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)

X_train, X_test, y_train, y_test = train_test_split(X, dummy_y, test_size=0.33, random_state=42)

# define baseline model
def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(50, input_dim=8, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(50, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(8, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='Nadam', metrics=['accuracy'])
    return model


batch_size = 16
num_epochs = 2000 # we iterate 2000 times over the entire training set
earlystop_p = 10

filename_prefix = "IceDNN"

earlystop = EarlyStopping(
        monitor='val_accuracy', 
        min_delta=0.001, 
        patience=earlystop_p, 
        verbose=1, 
        mode='auto')
    
csv_logger = CSVLogger('training_%s.log'%filename_prefix)
callbacks_list = [earlystop, csv_logger]


model = baseline_model()

model.fit(  X_train, 
            y_train,                # Train the model using the training set...
            batch_size=batch_size, 
            epochs=num_epochs,
            verbose=1, 
            shuffle = True,
            callbacks=callbacks_list,
            validation_split=0.2) # ...holding out 15% of the data for validation

scores = model.evaluate(X_test, y_test, verbose=1)  # Evaluate the trained model on the test set!

y_predict = model.predict(X_test)
conf_matx = confusion_matrix(y_test.argmax(axis=1), y_predict.argmax(axis=1))
print(conf_matx)
print(model.metrics_names)
print(scores)

model.save("%s.h5"%filename_prefix)
print("Saved model to disk!")
print("ScaleMin:")
print(X_min)
print("ScaleMax:")
print(X_max)

pkldict = {}
pkldict["X_min"] = X_min
pkldict["X_max"] = X_max

with open("%s.nncfg"%filename_prefix, "wb") as pklfile:
    pickle.dump(pkldict,pklfile)


# print("Starting Training")
# estimator = KerasClassifier(build_fn=baseline_model, epochs=100, batch_size=100, verbose=1)
# kfold = KFold(n_splits=100, shuffle=True)
# results = cross_val_score(estimator, X, dummy_y, cv=kfold)
# print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))