import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Define the path to the directory containing the images
data_dir = './train'

# Load the data into a Pandas DataFrame
data = []
species_dirs = os.listdir(data_dir)
for species_dir in species_dirs:
    species_path = os.path.join(data_dir, species_dir)
    if os.path.isdir(species_path):
        for file_name in os.listdir(species_path):
            if file_name.endswith('.jpg'):
                img_path = os.path.join(species_path, file_name)
                data.append((species_dir, img_path))

df = pd.DataFrame(data, columns=['species', 'filepath'])

# Define a function to extract features from the images
def extract_features(filepath):
    img = cv2.imread(filepath)
    img = cv2.resize(img, (100, 100))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist

# Extract features from the images and add them to the DataFrame
df['features'] = df['filepath'].apply(extract_features)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    list(df['features']), list(df['species']), test_size=0.2, random_state=42
)

# Train a logistic regression classifier
clf = LogisticRegression(random_state=42)
clf.fit(X_train, y_train)

# Make predictions on the testing set and compute accuracy
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy}")

# test Accuracy: 0.0
# train Accuracy: 0.01441484019613635
