import tkinter as tk
from tkinter import filedialog
import os
import cv2
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# Define the path to the directory containing the images
data_dir = './test'

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
n_jobs = -1  # Use all available cores
df['features'] = joblib.Parallel(n_jobs=n_jobs)(
    joblib.delayed(extract_features)(row['filepath']) for _, row in df.iterrows()
)

# Train a random forest classifier
if os.path.exists('trained_classifier.joblib'):
    clf = joblib.load('trained_classifier.joblib')
else:
    clf = RandomForestClassifier(random_state=42)
    clf.fit(list(df['features']), list(df['species']))
    # Save the trained classifier to a file
    joblib.dump(clf, 'trained_classifier.joblib')


class BirdSpeciesIdentifier(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.pack()
        self.create_widgets()

    def create_widgets(self):
        self.select_button = tk.Button(self, text="Select Image", command=self.select_image)
        self.select_button.pack()

        self.identify_button = tk.Button(self, text="Identify Species", command=self.identify_species)
        self.identify_button.pack()

        self.quit_button = tk.Button(self, text="Quit", command=self.master.destroy)
        self.quit_button.pack()

        self.result_label = tk.Label(self, text="")
        self.result_label.pack()

    def select_image(self):
        self.file_path = filedialog.askopenfilename()

    def identify_species(self):
        if hasattr(self, 'file_path'):
            img = cv2.imread(self.file_path)
            img = cv2.resize(img, (100, 100))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            species = clf.predict([hist])
            self.result_label.config(text=f"The identified species is {species[0]}")
        else:
            self.result_label.config(text="Please select an image first")


root = tk.Tk()
root.geometry("300x100")
root.title("Bird Identifier GUI")
app = BirdSpeciesIdentifier(master=root)
app.mainloop()