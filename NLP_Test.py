import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import make_pipeline
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Step 1: Load and explore the dataset
dataset_path = r'C:\Users\0042H8744\IMDB Dataset.csv'
df = pd.read_csv(dataset_path)

# Step 2: Data Preprocessing
df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})
train_texts, test_texts, train_labels, test_labels = train_test_split(df['review'], df['sentiment'], test_size=0.2, random_state=42)

# Load the trained model
model_nn = load_model('sentiment_model.h5')

# Tokenize the test data
tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(test_texts)

test_sequences = tokenizer.texts_to_sequences(test_texts)
test_padded = pad_sequences(test_sequences, maxlen=200, truncating='post', padding='post')

# Replace NaN values with a default value (0 in this case)
test_labels_int = test_labels.map({'positive': 1, 'negative': 0}).fillna(0)

# Make predictions
predictions_nn = model_nn.predict(test_padded)
predictions_nn = (predictions_nn > 0.5).astype(int)

# Evaluate the model
accuracy_nn = accuracy_score(test_labels_int, predictions_nn)
print(f"Neural Network Model Accuracy: {accuracy_nn}")
print(classification_report(test_labels_int, predictions_nn))
