import pandas as pd
import numpy as np
from sklearn import linear_model
import math

df = pd.read_csv("homeprices.csv")

#une astuce pour traiter les valeurs nuls est de le accorder le median de la column
bedrooms_median = math.floor(df.bedrooms.median()) #on a une valeure nulle dans la column bedrooms donc on le attribuer le median
df.bedrooms = df.bedrooms.fillna(bedrooms_median)
print(df)

reg = linear_model.LinearRegression()
reg.fit(df[['area', 'bedrooms', 'age']], df.price)
print(reg.coef_)

print(reg.predict([[3000, 3, 15]]))