import pandas as pd
import matplotlib.pyplot as plt  
from sklearn import svm
from sklearn import metrics
import joblib
import numpy as np
from sklearn.utils import shuffle


##Step 1: Get Data from CSV
dataframe = pd.read_csv('csv/dataset6labels.csv')
dataframe = dataframe.sample(frac=1).reset_index(drop=True) 
print(dataframe)


##Step 2: Seperate Labels and Features
X = dataframe.drop(['label'],axis=1)
y = dataframe['label']
X_train,y_train = X[0:130],y[0:130]
X_test,y_test = X[130:],y[130:]


##Step 3: Make sure you have the correct Feature / label combination in training



##Step 4: Build a Model and Save it
model = svm.SVC(kernel="linear")
model.fit(X_train,y_train)
joblib.dump(model,"model/svm_6label_linear")
 

##Step5 : Print Accuracy 
predictions = model.predict(X_test)
print("Model Accuracy is",metrics.accuracy_score(y_test,predictions))
