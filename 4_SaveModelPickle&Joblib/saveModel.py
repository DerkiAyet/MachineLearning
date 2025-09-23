import pandas as pd
import numpy as np
from sklearn import linear_model

df = pd.read_csv("homeprices.csv")
df.head()

reg = linear_model.LinearRegression()
reg.fit(df[['area']], df.price) #we use double [] to make it dataFrame not Series

print(reg.predict([[3300]])) #the value to pass in reg.predict must be a dataFrame

import pickle #a library used to serialize a series of date into a file

with open ('model_pickle', 'wb') as f: #Ouvre un fichier appelé 'model_pickle' en mode 'wb' (write binary = écriture binaire) et on le nommer f dans le code
    pickle.dump(reg, f)

with open ('model_pickle', 'rb') as f: #'wb' (read binary = lecteur binaire)
    np = pickle.load(f)
    
print(np.predict([[5000]])) #the thing is that not only pickle load the data itself but the model "reg" as it is
