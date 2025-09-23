import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_digits

digits = load_digits()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, train_size=0.7)

lr = LogisticRegression()
lr.fit (X_train, y_train)
print(f"score for logistic regression {lr.score(X_test, y_test)}")

svm = SVC()
svm.fit (X_train, y_train)
print(f"score for svm {svm.score(X_test, y_test)}")

rf = RandomForestClassifier()
rf.fit (X_train, y_train)
print(f"score for random forest {rf.score(X_test, y_test)}")

# but testing each time is not a solution especialy when train_test_split change the dataframes each timewe run the code
from sklearn.model_selection import KFold
kf = KFold(n_splits=3)

for train_index, test_index in kf.split([1,2,3,4,5,6,7,8,9]): #this data doesn't have anythig to do with the real data set
    print(train_index, test_index) #this shows how the kfold works, where: Each time:One chunk becomes the test set, The other two chunks become the training set and it goes into a loop
  
# a function to test each methode  
def get_score(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    return model.score(X_test, y_test)

print(f"score for random forest: {get_score(RandomForestClassifier(), X_train, X_test, y_train, y_test)}")


from sklearn.model_selection import StratifiedKFold
folds = StratifiedKFold(n_splits=3)

# Initialize empty lists to store accuracy scores for each model across the folds
scores_logistic = []
scores_svm = []
scores_rf = []

# Loop through each train/test split provided by StratifiedKFold
for train_index, test_index in folds.split(digits.data, digits.target):
    
    # Use the train and test indices to split the dataset into training and testing sets
    X_train, X_test = digits.data[train_index], digits.data[test_index]
    y_train, y_test = digits.target[train_index], digits.target[test_index]
    
    # Train a Logistic Regression model and append its accuracy to the list
    scores_logistic.append(
        get_score(LogisticRegression(solver='liblinear', multi_class='ovr'),
                  X_train, X_test, y_train, y_test)
    )

    # Train a Support Vector Machine model and append its accuracy to the list
    scores_svm.append(
        get_score(SVC(gamma='auto'),
                  X_train, X_test, y_train, y_test)
    )

    # Train a Random Forest model and append its accuracy to the list
    scores_rf.append(
        get_score(RandomForestClassifier(n_estimators=40),
                  X_train, X_test, y_train, y_test)
    )



print(f"scores for svm: {scores_svm}")
print(f"scores for logistic: {scores_logistic}")
print(f"scores for random forest: {scores_rf}")

from sklearn.model_selection import cross_val_score
cross_val_score(LogisticRegression(), digits.data, digits.target, cv=folds) 
#this do the same thing like the for loop block, if we don't specify the cv= it uses StratifiedKFold(n_splits=5)
#Uses KFold(n_splits=5) if y is not provided or it’s a regression task (the value of y is continues "number value").