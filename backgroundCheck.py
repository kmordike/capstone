import os
import cv2
import numpy as np
import pandas as pd

# Define the path to the directory containing the test images
test_dir = './test'

# Load the data into a Pandas DataFrame
data = []
species_dirs = os.listdir(test_dir)
for species_dir in species_dirs:
    species_path = os.path.join(test_dir, species_dir)
    if os.path.isdir(species_path):
        for i in range(1, 6):
            img_path = os.path.join(species_path, str(i) + '.jpg')
            if os.path.isfile(img_path):
                data.append((species_dir, img_path))

df = pd.DataFrame(data, columns=['species', 'filepath'])


# Define a function to get the background type of image
def get_background_type(filepath):
    # Load the image
    img = cv2.imread(filepath)
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Threshold the image to get a binary mask
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # Compute the mean color of the background
    mean_color = cv2.mean(img, mask=mask)[:3]
    # Define the different types of backgrounds
    backgrounds = {
        'white': (255, 255, 255),
        'black': (0, 0, 0),
        'gray': (128, 128, 128),
        'green': (0, 128, 0),
        'blue': (0, 0, 128),
        'red': (128, 0, 0),
        'yellow': (255, 255, 0),
        'orange': (255, 165, 0),
        'brown': (165, 42, 42),
        'purple': (128, 0, 128),
    }
    # Find the background type with the closest mean color
    dists = [np.linalg.norm(mean_color - np.array(color)) for color in backgrounds.values()]
    idx = np.argmin(dists)
    return list(backgrounds.keys())[idx]


# Add a column to the DataFrame with the background type of each image
df['background'] = df['filepath'].apply(get_background_type)

# Print the count of each background type
background_count = df['background'].value_counts()
print(background_count)
