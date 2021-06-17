#Load and pre-proccess data
import pandas as pd
import pickle
inputFolder = 'C:/Users/telelabpc2/Downloads/IrisFlowerSpecies/'
df = pd.read_csv(inputFolder + 'irisDocker.csv')
print(df.head())
#Split data into features and target
X = df.loc[:, df.columns != 'class']
print(X.head())
y = df['class']
print(y.head())
#Split data into train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.25)
print(X_train.shape)
print(X_train.head())
print(X_test.shape)
print(X_test.head())
print(y_train.shape)
print(y_train.head())
print(y_test.shape)
print(y_test.head())
#Model creation
from sklearn.ensemble import RandomForestClassifier
#create object of RandomForestClassifier 
model = RandomForestClassifier()
#train model
model.fit(X_train, y_train)
#print score
model.score(X_train,y_train)
#predict X_test data
predictions = model.predict(X_test)
predictions[:10]
#Scoring
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
print(accuracy_score(y_test, predictions))
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))
#Saving model
#save model in output directory
pickle.dump(model, open('model.pkl','wb'))
#Predict with new data
import numpy as np
test_data = [5.1, 3.2, 1.5, 0.4]
#convert test_data into numpy array
test_data = np.array(test_data)
#reshape
test_data = test_data.reshape(1,-1)
print(test_data)
#Load trained model
#declare path where you saved your model
outFileFolder = ''
filePath = outFileFolder + 'model.pkl'
#open file
file = open(filePath, "rb")
#load the trained model
trained_model = pickle.load(file)
#Predict with trained model
prediction = trained_model.predict(test_data)
print(prediction)