import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, classification_report

# Set the path to the data files
data_dir = "../data"
pos_file = os.path.join(data_dir, "rt-polarity.pos")
neg_file = os.path.join(data_dir, "rt-polarity.neg")

# Load the data
with open(pos_file, "r", encoding="ISO-8859-1") as f:
    positive_sentences = f.readlines()
with open(neg_file, "r", encoding="ISO-8859-1") as f:
    negative_sentences = f.readlines()

# Labels: positive=1, negative=0
positive_labels = [1] * len(positive_sentences)
negative_labels = [0] * len(negative_sentences)

# Combine data
sentences = positive_sentences + negative_sentences
labels = positive_labels + negative_labels

# Split the data into train (4,000 each), validation (500 each), and test sets (831 each)
X_train = sentences[:4000] + sentences[5331:9331]
y_train = labels[:4000] + labels[5331:9331]

X_val = sentences[4000:4500] + sentences[9331:9831]
y_val = labels[4000:4500] + labels[9331:9831]

X_test = sentences[4500:5331] + sentences[9831:]
y_test = labels[4500:5331] + labels[9831:]

# Convert text data to feature vectors
vectorizer = CountVectorizer(stop_words='english')
X_train_vect = vectorizer.fit_transform(X_train)
X_val_vect = vectorizer.transform(X_val)
X_test_vect = vectorizer.transform(X_test)

# Hyperparameter tuning with GridSearchCV
param_grid = {'alpha': [0.1, 0.5, 1.0, 2.0, 5.0]}
nb_classifier = MultinomialNB()
grid_search = GridSearchCV(nb_classifier, param_grid, cv=5, scoring='accuracy')

# Train with hyperparameter tuning using GridSearchCV
grid_search.fit(X_train_vect, y_train)

# Best model after hyperparameter tuning
best_nb_classifier = grid_search.best_estimator_
print(f"Best Parameters: {grid_search.best_params_}")

# Validation Accuracy (using best estimator)
val_accuracy = best_nb_classifier.score(X_val_vect, y_val)
print(f"Validation Accuracy with Best Estimator: {val_accuracy:.2f}")

# Evaluate the best model on the test set
y_pred = best_nb_classifier.predict(X_test_vect)

# Confusion Matrix and Metrics
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

# Metrics to report
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

print(f"True Positives (TP): {tp}")
print(f"True Negatives (TN): {tn}")
print(f"False Positives (FP): {fp}")
print(f"False Negatives (FN): {fn}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1_score:.2f}")

