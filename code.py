

# import the libraries
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

# read the dataset
dataset=pd.read_csv(r'C:\Users\ADMIN\Downloads\23rd- Poly\23rd- Poly\1.POLYNOMIAL REGRESSION\emp_sal.csv')

# x and y variables
x=dataset.iloc[: , 1:2].values
y = dataset.iloc[:,2].values

# decisison tree 
from sklearn.tree import DecisionTreeRegressor
dt_reg = DecisionTreeRegressor( criterion= "absolute_error", splitter="random", max_depth=3, min_samples_split=4,random_state=0,ccp_alpha=2)
dt_reg.fit(x,y)

# prediction
y_pred_dt =dt_reg.predict([[6.5]])
y_pred_dt
