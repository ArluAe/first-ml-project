# Sentiment Analysis Model

This repository provides a text classification model for sentiment analysis. It uses the Naive Bayes algorithm on TF-IDF transformed features to classify text data into sentiment labels (e.g., positive, negative). 

## Project Structure

- `sentiment_analysis.py`: Main script to train the model, evaluate it on test data, and allow real-time sentiment predictions.

## How It Works

### 1. Data Loading and Preprocessing
   - **Data Loading**: Loads text data from a CSV file (`sentiment_data.csv`), which should contain two columns: `text` (containing sentences or phrases) and `label` (sentiment labels).
   - **Splitting**: Divides the dataset into training and testing sets using an 80-20 split.

### 2. Feature Extraction
   - **TF-IDF Vectorization**: Converts text data into numerical features using the TF-IDF (Term Frequency-Inverse Document Frequency) approach, which captures the importance of words in each document relative to the entire dataset.

### 3. Model Training
   - **Naive Bayes Classifier**: A Multinomial Naive Bayes model is trained on the TF-IDF features of the training data. This classifier is suitable for text data, especially for problems with categorical text features.

### 4. Evaluation
   - The trained model is evaluated on the test set using accuracy score and a classification report, which includes precision, recall, and F1-score for each class.

### 5. Real-Time Sentiment Prediction
   - The script allows for real-time sentiment prediction. After training, users can input new sentences, and the model will predict the sentiment based on the TF-IDF features of the sentence.

## Usage

Run `sentiment_analysis.py` to train the model, evaluate it, and enter sentences for real-time sentiment predictions. Example usage:

```bash
python sentiment_analysis.py
