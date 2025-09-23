import pandas as pd
df = pd.read_csv("carprices.csv")
df.head()

import matplotlib.pyplot as plt

#Car Mileage Vs Sell Price ($)
plt.scatter(df['Mileage'],df['Sell Price($)'])
plt.show()

#Car Age Vs Sell Price ($)
plt.scatter(df['Age(yrs)'],df['Sell Price($)'])
plt.show()

X = df[['Mileage', 'Age(yrs)']]
y = df['Sell Price($)']

from sklearn.model_selection import train_test_split #this the model used for the splitting

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10) #using random_state (is not necessary to be 10 as value)means that I'm telling Python to: "Please split the data the same way every time I run the code".
print(len(X_train))
print(len(X_test))

from sklearn.linear_model import LinearRegression
reg = LinearRegression()

reg.fit(X_train, y_train)

print(f"the prediction for the test data set: \n{reg.predict(X_test)}")
print(f"the real prices are: \n{y_test}")

print(f"Accracy: {reg.score(X_test, y_test) * 100}%")