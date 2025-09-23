import pandas as pd
from matplotlib import pyplot as plt

df = pd.read_csv("income.csv")

plt.scatter(df.Age,df['Income($)'])
plt.xlabel('Age')
plt.ylabel('Income($)')
plt.show()

from sklearn.cluster import KMeans
km = KMeans(n_clusters=3)
y_predicted = km.fit_predict(df[['Age', 'Income($)']])
print(y_predicted)

df['cluster'] = y_predicted
print(df.head())

df1 = df[df.cluster== 0]
df2 = df[df.cluster== 1]
df3 = df[df.cluster== 2]
plt.title('malintroduced (without scaling)')
plt.scatter(df1.Age,df1['Income($)'],color='green')
plt.scatter(df2.Age,df2['Income($)'],color='red')
plt.scatter(df3.Age,df3['Income($)'],color='black')
plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],color='purple',marker='*',label='centroid') #mark each centroid by * [:, 0] means all rows, first column → x-coordinate of each centroid, [:,1] means all rows, second column → y-coordinate of each centroid.
plt.xlabel('Age')
plt.ylabel('Income ($)')
plt.legend()
plt.show()

#here we will see that clusters aren't grouped correctly in the scatter plot:
#that happend because:
 
#Age values: 20, 25, 40, 60 → range maybe 20–60.
#Income($) values: 20,000 to 200,000 → range much bigger.
# If you don’t scale:
# The big numbers in Income dominate the distance calculation.
# Age barely affects the result because 40 years is “small” compared to 200,000 dollars.
# So clusters will form mostly based on Income, not Age.

#Solution: we use MinMaxScaler() that transforms each feature so it’s between 0 and 1.

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(df[['Income($)']])
df['Income($)'] = scaler.transform(df[['Income($)']])

scaler.fit(df[['Age']])
df.Age = scaler.transform(df[['Age']])

#after that we again apply the kmeans
km = KMeans(n_clusters=3)
y_predicted = km.fit_predict(df[['Age', 'Income($)']])
print(y_predicted)
df['cluster'] = y_predicted

print(km.cluster_centers_) #the coordinates of the centroids

df1 = df[df.cluster==0]
df2 = df[df.cluster==1]
df3 = df[df.cluster==2]
plt.title('After Scaling')
plt.scatter(df1.Age,df1['Income($)'],color='green')
plt.scatter(df2.Age,df2['Income($)'],color='red')
plt.scatter(df3.Age,df3['Income($)'],color='black')
plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],color='purple',marker='*',label='centroid') #mark each centroid by * [:, 0] means all rows, first column → x-coordinate of each centroid, [:,1] means all rows, second column → y-coordinate of each centroid.
plt.xlabel('Age')
plt.ylabel('Income ($)')
plt.legend()
plt.show()

#The Elbow Methode:
k_rng = range(1, 10)
sse = []
for k in k_rng :
  km = KMeans(n_clusters=k)
  km.fit(df[['Age', 'Income($)']])
  sse.append(km.inertia_)

print(sse)

plt.title('The Elbow Plot')
plt.xlabel('K')
plt.ylabel('sse')
plt.plot(k_rng,sse)
plt.show()