import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

digits = load_digits()  #well this a library that continue a blurred images of numbers
print(dir(digits)) #there is a lot of infos such data(that include the pixels of the image), images, frame..
print(digits.data[0])

#----show an image of one data for example the first image---#
plt.gray()
plt.matshow(digits.images[0])

print(digits.target[0:5])

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(digits.data,digits.target,train_size=0.8)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)

print(f"the score: {model.score(X_test, y_test)}")

print(f"the prediction for object 67: {model.predict([digits.data[67]])}")
print(f"and the real answer for object 67: {digits.target[67]}")

#-------------Confusion Matrix---------#
from sklearn.metrics import confusion_matrix
y_predicted = model.predict(X_test)
cm = confusion_matrix(y_test, y_predicted)
print(f"Confusion Matrix: \n{cm}")

import seaborn as sn
plt.figure(figsize = (10,7))
sn.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()