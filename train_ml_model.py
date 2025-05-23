import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Data load karein
data = pd.read_csv('ml_data.csv')

# Rows Double (It will solve your issue for now)
#data = pd.concat([data]*2, ignore_index=True)

# Columns swap - symptoms se disease predict karein
X = data['label']  # Symptoms
y = data['text']   # Diseases

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)

# Model banayein
model = make_pipeline(
    TfidfVectorizer(),
    MultinomialNB()
)

# Train karein
model.fit(X_train, y_train)

# Test karein
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)

# Confusion matrix banayein
cm = confusion_matrix(y_test, predictions)

# Confusion matrix ko heatmap se visualize karein
plt.figure(figsize=(10, 8))  # Adjust figure size as needed

#Get all unique names. It will need it for proper visuliaztion
unique_labels = sorted(data['text'].unique())

sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=unique_labels, yticklabels=unique_labels)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.xticks(rotation=45, ha="right") # Rotate x-axis labels for better readability
plt.tight_layout() # Adjust layout to prevent labels from overlapping
plt.show()
# Save figure
plt.savefig('confusion_matrix.png')