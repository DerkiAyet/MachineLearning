import pandas as pd

df = pd.read_csv('titanic.csv')
df.drop(['PassengerId','Name','SibSp','Parch','Ticket','Cabin','Embarked'],axis='columns',inplace=True)
print(df.head())

inputs = df.drop('Survived',axis='columns')
target = df.Survived

dummies = pd.get_dummies(inputs.Sex, drop_first=True) #we use the dummies for sex cause there are 2 types: male, female and of course in the table they are not numerical values, and to avoid the dummy trap we always drop one 
print(dummies.head(3))

inputs = pd.concat([inputs, dummies], axis='columns')
print(inputs.head())

inputs.drop('Sex', axis='columns', inplace=True)

#in case we have null values we fill it with the mean of the column, for example the age could be null
inputs.Age = inputs.Age.fillna(inputs.Age.mean())

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(inputs, target, train_size=0.8)

#the naive bayes model
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()

model.fit(X_train, y_train)
print(f"the score {model.score(X_test, y_test)}")

print(f"the true first 10 values the X_test (y_test): {y_test[0:10]}")
print(f"the predicted values of the 10 first values of X_test(0: didn't survive, 1: survived): {model.predict(X_test[0:10])}")



