from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import pandas as pd
import joblib

# Load the data
data = pd.read_csv("train.csv", encoding='latin1')

# Display the first few rows of the DataFrame
print(data.head())

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(data['Denumire'], data['Categorie'], test_size=0.2, random_state=42)

# Convert text data into numerical features using CountVectorizer
vectorizer = CountVectorizer()
X_train_counts = vectorizer.fit_transform(X_train)
X_test_counts = vectorizer.transform(X_test)

# Train a Naive Bayes classifier
clf = MultinomialNB()
clf.fit(X_train_counts, y_train)

# Make predictions
predictions = clf.predict(X_test_counts)

# Evaluate accuracy
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)

# Save the trained model
joblib.dump(clf, 'naive_bayes_model.pkl')

# Save the CountVectorizer
joblib.dump(vectorizer, 'count_vectorizer.pkl')

# Save the classes
classes = y_train.unique().tolist()  # Assuming y_train contains class labels
joblib.dump(classes, 'classes.pkl')