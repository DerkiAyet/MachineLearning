import pandas as pd

df = pd.read_csv("diabetes.csv")
print(df.head())
print(df.isnull().sum()) #isna() == isnull()
print(df.describe())
print(f"values of Outcome with their total rows: {df.Outcome.value_counts()}")

X = df.drop('Outcome', axis='columns')
y = df.Outcome

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X = X_scaled

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, stratify=y, random_state=10) #the stratify parameter is used to make sure the class distribution is the same in both the training and test sets.

print(X_train.shape)
print(y_train.value_counts())

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

scores = cross_val_score(DecisionTreeClassifier(), X, y, cv=5)
print(scores)
print(f"the mean of cross validation (without bagging):  {scores.mean()}")

from sklearn.ensemble import BaggingClassifier

bag_model = BaggingClassifier(
    estimator=DecisionTreeClassifier(),
    n_estimators= 100,
    max_samples=0.8, #the training set in each tree is 80%
    oob_score=True, #setting this true means to tell the bagging model to calculate the mean of the score on each 20% reamining for the test
    random_state=0
)
# oob_score=True :Out-of-Bag (OOB) scoring.
# Since each tree is trained on only 80% of the data, the remaining 20% (on average) are not seen by that tree.
# These “out-of-bag” samples can be used like a built-in validation set to estimate accuracy without doing an explicit train_test_split.
# Setting this to True tells the model to calculate and store that score in bag_model.oob_score_.

bag_model.fit(X_train, y_train)
print(f"the oob-score of the bagging: {bag_model.oob_score_}")
print(f"the score after using the bagging: {bag_model.score(X_test, y_test)}")

bag_model = BaggingClassifier(
    estimator= DecisionTreeClassifier(),
    n_estimators= 100,
    max_samples=0.8, #the training set in each tree is 80%
    oob_score=True, #setting this true means to tell the bagging model to calculate the mean of the score on each 20% reamining for the test
    random_state=0
)

scores = cross_val_score(bag_model, X, y, cv=5)
print(f"the bagging score with cross validation (mean): {scores.mean()}")

#Compare with the Random Forest
from sklearn.ensemble import RandomForestClassifier
scores = cross_val_score(RandomForestClassifier(n_estimators=50), X, y, cv=5)
print(f"the score with Random Forest: {scores.mean()}")