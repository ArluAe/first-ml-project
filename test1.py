import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Load the data
data = pd.read_csv('sentiment_data.csv')

# Split the data into features (X) and labels (y)
X = data['text']
y = data['label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Convert text data into TF-IDF features
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train a Naive Bayes classifier
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test_tfidf)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print("Classification Report:")
print(report)

while True:
    # Function to predict sentiment of a new sentence
    def predict_sentiment(sentence):
        sentence_tfidf = vectorizer.transform([sentence])  # Convert sentence to TF-IDF feature
        prediction = model.predict(sentence_tfidf)         # Predict sentiment
        return prediction[0]                               # Return the predicted sentiment

    # Example usage
    input_sentence = input()
    predicted_sentiment = predict_sentiment(input_sentence)
    print(f"The predicted sentiment for the sentence '{input_sentence}' is: {predicted_sentiment}")