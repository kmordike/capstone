import pandas as pd
# import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Check the CSV file and header row
data = pd.read_csv('birds.csv')
print(data.head())

# Check the column names in the DataFrame
print(data.columns)

# Number of bird species
print("Number of bird species:", len(data['labels'].unique()))

# Number of images per species
image_count = data['labels'].value_counts().to_frame().reset_index()
image_count.columns = ['label', 'count']

# Rename the columns
image_count = image_count.rename(columns={'labels': 'label', 'count': 'count'})

image_count.sort_values('count', ascending=False, inplace=True)
print(image_count.head())

# Distribution of images per species
plt.figure(figsize=(10, 6))
sns.histplot(data=image_count, x='count', binwidth=10)
plt.xlabel("Number of Images")
plt.ylabel("Count of Bird Species")
plt.title("Distribution of Images per Bird Species")
plt.show()
