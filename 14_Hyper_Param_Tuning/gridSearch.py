from sklearn import svm, datasets
iris = datasets.load_iris()
import pandas as pd
import numpy as np

df = pd.DataFrame(iris.data,columns=iris.feature_names)
df['flower'] = iris.target
df['flower'] = df['flower'].apply(lambda x: iris.target_names[x])
print(df[47:150])

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3)

model = svm.SVC(kernel='rbf',C=30,gamma='auto')
model.fit(X_train,y_train)
print(f"the score for svm with kernel= 'rbf and C=30: {model.score(X_test, y_test)}")

from sklearn.model_selection import cross_val_score
print(f"the score with 5 folds for svm with kernel= 'linear' and C=10{cross_val_score(svm.SVC(kernel='linear',C=10,gamma='auto'),iris.data, iris.target, cv=5)}")
print(f"the score with 5 folds for svm with kernel= 'rbf' and C=10{cross_val_score(svm.SVC(kernel='rbf',C=10,gamma='auto'),iris.data, iris.target, cv=5)}")

#to test what the paremeteres the most suitabel with svm for this data set
kernels = ['rbf', 'linear']
C = [1,10,20]
avg_scores = {}
for kval in kernels:
    for cval in C:
        cv_scores = cross_val_score(svm.SVC(kernel=kval,C=cval,gamma='auto'),iris.data, iris.target, cv=5)
        avg_scores[kval + '_' + str(cval)] = np.average(cv_scores) #so the result we're going to have using the np.average we will get the mean of the scores of the 5 folds

print(f"the scores (without the Grid Search): {avg_scores}")

#The Grid Search for finding the best parameteres
from sklearn.model_selection import GridSearchCV
clf = GridSearchCV(svm.SVC(gamma='auto'), {
    'C': [1,10,20],
    'kernel': ['rbf','linear']
}, cv=5, return_train_score=False)

clf.fit(X_train, y_train)

print(f"Grid Search results: {clf.cv_results_}")

df = pd.DataFrame(clf.cv_results_)
print(df.head())
print(df[['param_C','param_kernel','mean_test_score']])
print(f"the best parameteres: {clf.best_params_}")
print(f"the best score: {clf.best_score_}")

#As for the Randomize we give it the parameteres and it's going to pick random combinations and pick the best off it, this one doesn't test them all
from sklearn.model_selection import RandomizedSearchCV
rs = RandomizedSearchCV(svm.SVC(gamma='auto'), {
        'C': [1,10,20],
        'kernel': ['rbf','linear']
    }, 
    cv=5, 
    return_train_score=False, 
    n_iter=2 #this is stands for how many different random combinations of hyperparameters you want to try -> so we gonna test only two combinations
)
rs.fit(iris.data, iris.target)
results = pd.DataFrame(rs.cv_results_)[['param_C','param_kernel','mean_test_score']]
print(f"Random Search Results with n_iter = 2:\n {results}")

#Finding the best model
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

model_params = {
    'svm': {
        'model': svm.SVC(gamma='auto'),
        'params' : {
            'C': [1,10,20],
            'kernel': ['rbf','linear']
        }  
    },
    'random_forest': {
        'model': RandomForestClassifier(),
        'params' : {
            'n_estimators': [1,5,10]
        }
    },
    'logistic_regression' : {
        'model': LogisticRegression(solver='liblinear'),
        'params': {
            'C': [1,5,10]
        }
    }
}

scores = []

for model_name, mp in model_params.items():
    clf =  GridSearchCV(mp['model'], mp['params'], cv=5, return_train_score=False)
    clf.fit(iris.data, iris.target)
    scores.append({
        'model': model_name,
        'best_score': clf.best_score_,
        'best_params': clf.best_params_
    })
    
df = pd.DataFrame(scores,columns=['model','best_score','best_params'])
print(df.head())