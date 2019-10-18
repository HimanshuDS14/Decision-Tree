import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics , preprocessing , model_selection


data = pd.read_csv("Position_Salaries.csv")
print(data.head(10))

x = data.iloc[:,1:2].values
y = data.iloc[:,2].values


reg = DecisionTreeRegressor()
reg.fit(x,y)


#prediction
z = np.array([7])
z = z.reshape(1,-1)

pred = reg.predict(z)
print(pred)









