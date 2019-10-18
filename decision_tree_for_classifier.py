import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report , confusion_matrix


data = pd.read_csv("authentication.csv")
print(data.head(10))

x = data.iloc[:,0:4].values
y = data.iloc[:,4].values

print(x)
print(y)

train_x , test_x , train_y , test_y = train_test_split(x,y,test_size=0.2 )

classifier = DecisionTreeClassifier()
classifier.fit(train_x , train_y)

pred = classifier.predict(test_x)

print(confusion_matrix(test_y , pred))
print(classification_report(test_y , pred))





