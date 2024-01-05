import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import make_pipeline
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Step 1: Load and explore the dataset
dataset_path = r'C:\Users\0042H8744\IMDB Dataset.csv'
df = pd.read_csv(dataset_path)

# Step 2: Data Preprocessing
df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})
train_texts, test_texts, train_labels, test_labels = train_test_split(df['review'], df['sentiment'], test_size=0.2, random_state=42)

# Step 3: Build and Train a Simple Naive Bayes Model
model_nb = make_pipeline(CountVectorizer(), MultinomialNB())
model_nb.fit(train_texts, train_labels)
predictions_nb = model_nb.predict(test_texts)
accuracy_nb = accuracy_score(test_labels, predictions_nb)
print(f"Naive Bayes Model Accuracy: {accuracy_nb}")
print(classification_report(test_labels, predictions_nb))

# Step 4: Build and Train a Simple Neural Network Model using TensorFlow
tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(train_texts)

train_sequences = tokenizer.texts_to_sequences(train_texts)
test_sequences = tokenizer.texts_to_sequences(test_texts)

train_padded = pad_sequences(train_sequences, maxlen=200, truncating='post', padding='post')
test_padded = pad_sequences(test_sequences, maxlen=200, truncating='post', padding='post')

model_nn = Sequential([
    Embedding(input_dim=10000, output_dim=16, input_length=200),
    LSTM(64),
    Dense(1, activation='sigmoid')
])

model_nn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model_nn.fit(train_padded, train_labels, epochs=5, validation_data=(test_padded, test_labels))

# Save the trained model
model_nn.save('sentiment_model.h5')

_, accuracy_nn = model_nn.evaluate(test_padded, test_labels)
print(f"Neural Network Model Accuracy: {accuracy_nn}")
