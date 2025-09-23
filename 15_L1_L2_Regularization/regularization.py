import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
# Suppress Warnings for clean notebook
import warnings
warnings.filterwarnings('ignore')

dataset = pd.read_csv('Melbourne_housing_FULL.csv')
print(dataset.head())

print(dataset.nunique())
print(f"number of rows and columns: {dataset.shape}")

cols_to_use = ['Suburb', 'Rooms', 'Type', 'Method', 'SellerG', 'Regionname', 'Propertycount', 
               'Distance', 'CouncilArea', 'Bedroom2', 'Bathroom', 'Car', 'Landsize', 'BuildingArea', 'Price']
dataset = dataset[cols_to_use]
print(dataset.head())

print(f"number of null values in each column: {dataset.isna().sum()}")

# Some feature's missing values can be treated as zero (another class for NA values or absence of that feature)
# like 0 for Propertycount, Bedroom2 will refer to other class of NA values
# like 0 for Car feature will mean that there's no car parking feature with house
cols_to_fill_zero = ['Propertycount', 'Distance', 'Bedroom2', 'Bathroom', 'Car']
dataset[cols_to_fill_zero] = dataset[cols_to_fill_zero].fillna(0)

# other continuous features can be imputed with mean for faster results since our focus is on Reducing overfitting
dataset['Landsize'] = dataset['Landsize'].fillna(dataset.Landsize.mean())
dataset['BuildingArea'] =  dataset['BuildingArea'].fillna(dataset.BuildingArea.mean())

print(f"number of null values in each column after: {dataset.isna().sum()}")

#We Drop NA values of Price, since it's our predictive variable we won't impute it
dataset.dropna(inplace=True)

#The dummies:
dataset = pd.get_dummies(dataset, drop_first=True)

#The dataset and target (Price):
X = dataset.drop('Price', axis='columns')
y = dataset.Price

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


from sklearn.linear_model import LinearRegression
reg = LinearRegression().fit(X_train, y_train)
print(f"the score for Linear Regression: {reg.score(X_test, y_test)}")

#The Regularization for Regression, there is lasso and ridge for the classifiaction also:
from sklearn.linear_model import Lasso, Ridge
lasso_reg = Lasso(alpha=50, max_iter=100, tol=0.1)
# alpha=50 → This is λ (regularization strength).
# Higher α → stronger penalty → smaller weights.
# Lower α → weaker penalty → weights can grow more freely.
# max_iter=100 → The maximum number of optimization iterations allowed.
# tol=0.1 → The tolerance for stopping the optimization (stop early if improvement is smaller than this).

lasso_reg.fit(X_train, y_train)
print(f"The score for the training set with Lasso(L1): {lasso_reg.score(X_train, y_train)}")
print(f"The score for the test set with Lasso(L1): {lasso_reg.score(X_test, y_test)}")

ridge_reg = Ridge(alpha=50, max_iter=100, tol=0.1)
ridge_reg.fit(X_train, y_train)
print(f"The score for the training set with Ridge(L2): {ridge_reg.score(X_train, y_train)}")
print(f"The score for the test set with Ridge(L2): {ridge_reg.score(X_test, y_test)}")
