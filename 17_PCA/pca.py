from sklearn.datasets import load_digits
import pandas as pd

dataset = load_digits()
print(f"the keys: \n{dataset.keys()}")
print(f"rows x columns: {dataset.data.shape}")

df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
print(df.head())
print(df.describe())

X = df
y = dataset.target

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_scaled = scaler.fit_transform(X) #we are scaling here so the values will be betwen 0 and 1

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x_scaled, y,  test_size=0.2, random_state=30)

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)
(f"The score: {model.score(X_test, y_test)}")

#Apply PCA with Use components such that 95% of variance is retained
from sklearn.decomposition import PCA

pca = PCA(0.95)
X_pca = pca.fit_transform(X)
print(f"The number of columns that retained 95% of variance: {X_pca.shape[1]}")
print(f"Variance Expliqué: {pca.explained_variance_ratio_}")

#Now we apply on the reduced dataset
X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(X_pca, y,  test_size=0.2, random_state=30)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_pca, y_train)
print(f"The score after reducing the components that retained 95% of the variance: {model.score(X_test_pca, y_test)}")

pca = PCA(n_components=2) #reduce to 2 principla components this methode is not the best one always use the variance
X_pca = pca.fit_transform(X)
X_pca.shape

#Now we apply on the reduced dataset
X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(X_pca, y,  test_size=0.2, random_state=30)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_pca, y_train)
print(f"The score after reducing to only 2 components {model.score(X_test_pca, y_test)}")
