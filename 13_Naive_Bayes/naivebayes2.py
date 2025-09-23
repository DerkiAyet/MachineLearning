import pandas as pd

df = pd.read_csv('spam.csv')
print(df.head())

df.groupby('Category').describe() #a methode that regroupe the rows by Category that we specified

#first we label the spam and ham individuals

df['spam'] =  df['Category'].apply(lambda x: 1 if x=="spam" else 0)
print(df.head())

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df.Message, df.spam, train_size=0.8)

#to turn the messages into a matrix:
from sklearn.feature_extraction.text import CountVectorizer
v = CountVectorizer()
X_train_count = v.fit_transform(X_train.values)
X_train_count.toarray()[:2]

#in case of detecting the spam or work with words in naive bayes we use Multinomial Naive Bayes
from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
model.fit(X_train_count,y_train)

emails = [
    'Hey mohan, can we get together to watch footbal game tomorrow?',
    'Upto 20% discount on parking, exclusive offer just for you. Dont miss this reward!'
]
emails_count = v.transform(emails)
print(f"the two emails are: (0: not spam, 1: spam) {model.predict(emails_count)}")

X_test_count = v.transform(X_test)
print(f"the score: {model.score(X_test_count, y_test)}")

from sklearn.pipeline import Pipeline
clf = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('nb', MultinomialNB())
])
clf.fit(X_train, y_train) #in this we observe that we don't need to vectorize the X_train in this case the Pipeline always do this instead of us 

print(f"the two emails are: (0: not spam, 1: spam) {clf.predict(emails)}")#the pipeline make it easy for us because it's always vectorise on it's own
print(f"the score: {clf.score(X_test, y_test)}")

