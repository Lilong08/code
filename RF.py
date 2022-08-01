# Random Forest Classifier

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# # Importing the datasets

#TODO: read the datasets
X = np.array([[]])  # numpy.ndarray
Y = np.array([[]])  # numpy.ndarray

# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)




# StandardScaler
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
x_train = sc_X.fit_transform(x_train)
x_test = sc_X.transform(x_test)


from scaler import my_scaler
# Log scaling
log_scaling = my_scaler()
x_train = log_scaling.fit(x_train)
x_test = log_scaling.fit(x_test)


# Fitting the classifier into the Training set

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 200, criterion = 'entropy', random_state = 0)
classifier.fit(x_train,y_train)

# Predicting the test set results
y_pred = classifier.predict(x_test)


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Visualising the Training set results
