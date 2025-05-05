# Deep_learning_project_2
# 🎬 IMDB Movie Review Sentiment Analysis using Deep Learning

This project applies deep learning techniques to perform sentiment analysis on IMDB movie reviews. It classifies movie reviews as **positive** or **negative** based on their text content, using Natural Language Processing (NLP) and neural networks.

---

## 🔍 Problem Statement

**Goal:**  
Develop a binary classification model that predicts the sentiment of a movie review (Positive or Negative) using the IMDB dataset.

**Why it matters:**  
Sentiment analysis helps platforms understand user feedback, improve recommendations, and monitor public opinion at scale.

---

## 📁 Dataset

- **Source:** [IMDB Large Movie Review Dataset](https://ai.stanford.edu/~amaas/data/sentiment/)
- **Size:** 50,000 labeled reviews (25,000 for training, 25,000 for testing)
- **Labels:**  
  - `1` for Positive  
  - `0` for Negative  

The dataset is balanced and widely used for benchmarking sentiment analysis models.

---

## 🧠 Model Architecture

We use a **deep learning model** built with TensorFlow/Keras:

```python
model = Sequential([
    Embedding(input_dim=10000, output_dim=128, input_length=500),
    LSTM(128, return_sequences=False),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])
