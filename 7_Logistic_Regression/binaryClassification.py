import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

df = pd.read_csv("insurance_data.csv")
df.head()

plt.scatter(df.age,df.bought_insurance,marker='+',color='red')

#-----------split the data set---------------#
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df[['age']],df.bought_insurance,train_size=0.8)
print(X_test)

#-----------Logistic Regression-------------#
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()

model.fit(X_train, y_train)
print(f"the prediction: \n{model.predict(X_test)}")
print(f"the score: {model.score(X_test, y_test)}")
print(f"the probalities of prediction: {model.predict_proba(X_test)}") #this will print a matrix of 2 columns the first column show the proba of being 0 of each age and the second column shox the proba of taking value 1 

# --- Create the sigmoid curve --- #
# Generate 100 age values evenly spaced between min and max
age_range = np.linspace(df.age.min(), df.age.max(), 100).reshape(-1, 1)
# Get predicted probability of class 1 (bought insurance)
probabilities = model.predict_proba(age_range)[:, 1]  # take column 1 (P(class=1))

# Plot the sigmoid curve
plt.plot(age_range, probabilities, color='blue', label='Sigmoid curve (model prediction)')

plt.xlabel('Age')
plt.ylabel('Probability of buying insurance')
plt.legend()
plt.title('Logistic Regression: Age vs Insurance Purchase Probability')
plt.grid(True)
plt.show()