import pandas as pd
import matplotlib.pyplot as plt  
from sklearn import svm
from sklearn import metrics
import joblib
from sklearn.decomposition import PCA
import numpy as np
from sklearn.utils import shuffle
#Get data from csv 
dataframe = pd.read_csv('csv/dataset6labels.csv')
dataframe = dataframe.sample(frac=1).reset_index(drop=True)

print(dataframe)
#Separate Labels and Features
X = dataframe.drop(['label'], axis=1)
Y = dataframe['label']

# X_train, Y_train =  X[0:198], Y[0:198]
# X_test,Y_test = X[198:],Y[198:] 
X_train, Y_train =  X, Y
X_test,Y_test = X,Y


grid_data = X_train.values[20].reshape(28,28)
plt.imshow(grid_data,interpolation=None,cmap="gray")
plt.title(Y_train.values[20])
plt.show()



model = svm.SVC(kernel="linear")

print("Fitting this might take some time .....")

model.fit(X_train,Y_train)

joblib.dump(model, "model/svm_6label_linear") 
#model = joblib.load("svm_class_1")
print("predicting .....")
predictions = model.predict(X_test)

print("Getting Accuracy .....")
print("Score", metrics.accuracy_score(Y_test, predictions))

