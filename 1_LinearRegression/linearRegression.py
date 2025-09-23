import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

df = pd.read_csv("homeprices.csv")
print(df)
plt.xlabel('area')
plt.ylabel('price(US$)')
plt.scatter(df.area, df.price, color='red', marker='+')
plt.show() #necessary to show the plot graph

reg = linear_model.LinearRegression()
reg.fit(df[['area']], df.price) #we use double [] to make it dataFrame not Series

print(reg.predict([[3300]])) #the value to pass in reg.predict must be a dataFrame
print(f"reg.coef_ = {reg.coef_} reg.intercept = {reg.intercept_}")

d = pd.read_csv("areas.csv")
d['prices'] = reg.predict(d)
d.to_csv("predictions.csv") # to export it to a csv file

#a plot to compare between the prediction od the regression and the real prices
plt.xlabel('area')
plt.ylabel('price')
plt.scatter(df.area, df.price, color='red', marker='+')
plt.plot(df.area, reg.predict(df[['area']]), color='blue') #plot create a line
plt.show()