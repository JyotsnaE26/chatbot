# train_chatbot_model.py

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer
import json

# Load training data
training_data = [
    # FAQs
    ("what are your business hours?", "faq"),
    ("how can i contact customer support?", "faq"),
    ("where are you located?", "faq"),
    ("how much time do you take for delivery?", "faq"),
    ("can i pre-order?", "faq"),
    ("do you offer international delivery?", "faq"),
    ("is there a delivery fee?", "faq"),
    ("how can I change my order?", "faq"),
    ("do you have a return policy?", "faq"),
    ("what payment methods do you accept?", "faq"),

    # Greetings
    ("hello", "greeting"),
    ("hi", "greeting"),
    ("hey there", "greeting"),
    ("good morning", "greeting"),
    ("good evening", "greeting"),
    ("what's up?", "greeting"),
    ("hey", "greeting"),
    ("hi there!", "greeting"),
    ("greetings", "greeting"),
    ("hola!", "greeting"),

    # Recipes
    ("how to make pancakes", "recipe"),
    ("ingredients for chocolate cake", "recipe"),
    ("steps for spaghetti", "recipe"),
    ("what's in pasta", "recipe"),
    ("how to cook noodles", "recipe"),
]

# Split data into texts and labels
X = [text for text, label in training_data]
y = [label for text, label in training_data]

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Use SentenceTransformer to create embeddings
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
X_embeddings = embedding_model.encode(X)

# Train classifier
clf = RandomForestClassifier()
clf.fit(X_embeddings, y_encoded)

# Save the classifier, label encoder, and embedding model
joblib.dump(clf, 'intent_classifier.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')
joblib.dump(embedding_model, 'embedding_model.pkl')

print("Training complete. Models saved as:")
print("- intent_classifier.pkl")
print("- label_encoder.pkl")
print("- embedding_model.pkl")
