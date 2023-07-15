# Importing necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm

# Loading the dataset
data = pd.read_csv('Tweets.csv')

# Specify column names for text and labels
text_column = 'text'
label_column = 'label'

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data[text_column], data[label_column], test_size=0.2, random_state=42)

# Converting text to numerical features
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# Training the model
classifier = svm.SVC(kernel='linear')
classifier.fit(X_train, y_train)

# Making predictions
predictions = classifier.predict(X_test)

# Evaluating the model
accuracy = classifier.score(X_test, y_test)
print('Accuracy:', accuracy)
