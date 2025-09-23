import pandas as pd
import numpy as np

#----------Categorical Variables (the One Hot Encoding Method)-----------#

df = pd.read_csv("homeprices.csv")
dummies = pd.get_dummies(df.town) #this what make the 0/1 indicators

merged = pd.concat([df, dummies], axis="columns") #to concat the original table with the one hot encoding result
final = merged.drop(['town', 'west windsor'], axis="columns") #we drop the categorial column and one od the dummy variables -> WHY?: see in the resume txt file

print(final) #it's not a prblm if you didn't drop the categorial column because the methode will do that automatically(ex: linear regression)

from sklearn.linear_model import LinearRegression

model = LinearRegression()

X = final.drop('price', axis="columns")
y = final.price

model.fit(X, y)
print(f"Prediction for area= 2800 and from robinsville: { model.predict([[2800,0,1]])}")
print(f"Prediction for area= 3300 and from west windsor: { model.predict([[2800,0,1]])}")

#--------------Another methode (fit_transform)----------#
from sklearn.preprocessing import LabelEncoder
le= LabelEncoder()

dfle = df
dfle.town= le.fit_transform(dfle.town) #transforms the town column from text into numbers: 0, 1, 2
print(dfle)

X = dfle[['town', 'area']]
y = dfle.price

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer([
    ('town_enc', OneHotEncoder(drop='first'), ['town'])  # drop='first' to avoid dummy trap
], remainder='passthrough')  # leave 'area' untouched

#Remarque: This applies one-hot encoding to the first column (town, that is now encoded into number). 
#But wait, did that mean this methode needs numerical values? -> well from the update.20 we can work on strings directly
#Knt ntba3 tutorial win kan fi version 9dima bs7 drwk njmou direct nappliquo 3la strings

X_transformed = ct.fit_transform(X)

model.fit(X_transformed, y)

# robinsville → depends on how encoding is done. You can check column order by:
print(ct.get_feature_names_out())

print(f"Prediction for area= 2900 and from robinsville { model.predict([[0,1, 2900]])}")

