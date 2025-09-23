from sklearn.datasets import load_digits
import pandas as pd

digits = load_digits()

df = pd.DataFrame(digits.data)
print(df.head(4))
df['target'] = digits.target
print(df.head(4))

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df.drop(['target'], axis='columns'), df.target, train_size=0.8)
print(len(X_train))

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=20) #number of trees
model.fit(X_train, y_train)

print(f"score: {model.score(X_test, y_test)}")
y_predicted = model.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_predicted)

import matplotlib.pyplot as plt
import seaborn as sn
plt.figure(figsize=(10,7))
sn.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()