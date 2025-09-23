import pandas as pd
from sklearn.datasets import load_iris
iris = load_iris()

print(f"the data columns{dir(iris)}")
print(iris.feature_names)

df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target
print(df.head(5))

print(iris.target_names) #well notice 0: setosa, 1:versicolor, 2: virginca
df['flower_name'] = df.target.apply(lambda x: iris.target_names[x])
print(df.head(10))

from matplotlib import pyplot as plt
df0 = df[df.target== 0]
df1 = df[df.target== 1]
df2 = df[df.target== 2] #sperate each type of flower into sperate data frames

plt.title('Sepal length vs Sepal Width (Setosa vs Versicolor)')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.scatter(df0['sepal length (cm)'], df0['sepal width (cm)'],color="green",marker='+')
plt.scatter(df1['sepal length (cm)'], df1['sepal width (cm)'],color="blue",marker='.')
plt.show()

plt.title('Petal length vs Pepal Width (Setosa vs Versicolor)')
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.scatter(df0['petal length (cm)'], df0['petal width (cm)'],color="green",marker='+')
plt.scatter(df1['petal length (cm)'], df1['petal width (cm)'],color="blue",marker='.')
plt.show()

from sklearn.model_selection import train_test_split
X= df.drop(['target', 'flower_name'], axis='columns')
y= df.target

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)

from sklearn.svm import SVC

model = SVC(C=10) #the more c is high the more the score get lower, and we can choose liner and poly for the kernel, the linear one give a higher score
model.fit(X_train, y_train)
print(f"score: {model.score(X_test, y_test)}")