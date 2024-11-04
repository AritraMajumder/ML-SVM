import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings("ignore")

df = pd.read_csv("diabetes.csv")

x = df.drop(["Outcome"],axis=1)
y = df["Outcome"]

scaler = StandardScaler()
scaler.fit(x)

x = scaler.transform(x)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state=101)

#model
classifier = svm.SVC(kernel="linear")
classifier.fit(x_train,y_train)


#single prediction
#ip = [6,148,72,35,0,33.6,0.627,50]  #1
ip = [5,116,74,0,0,25.6,0.20,130]  #0

print('Prediction: ',classifier.predict([ip]))

test_prediction = classifier.predict(x_test)
test_accuracy = accuracy_score(test_prediction,y_test)
print('Accuracy: ',test_accuracy*100)



#metrics
cf = confusion_matrix(y_test,test_prediction)
cm_display = ConfusionMatrixDisplay(confusion_matrix = cf, display_labels = [False, True])
cm_display.plot()
plt.show()



#HYPERPARAMETER TUNING(optional)

#1.check kernerls
kernel = ['linear','rbf','poly','sigmoid']
for i in kernel:
    model = svm.SVC(kernel=i,C=1) #C is penalty for error, encourages model to not make errors
    model.fit(x_train,y_train)
    #print("Model: ",i,"|| Accuracy: ",model.score(x_test,y_test))

#2.check degrees
for i in range(1,11):
    model = svm.SVC(kernel='linear',degree=i,C=50)
    model.fit(x_train,y_train)
    #print("Degree: ",i,"|| Accuracy: ",model.score(x_test,y_test))


#GRIDSEARCH CV 
from sklearn.model_selection import GridSearchCV

params = {'C':[0.1,1,100],'kernel':['linear','rbf','poly','sigmoid'],'degree':[1,2,3,4,5,6] }
#dict of params and their values we want to check

grid = GridSearchCV(svm.SVC(),params)
grid.fit(x_train,y_train)

print(grid.best_params_)
print(grid.score(x_test,y_test))