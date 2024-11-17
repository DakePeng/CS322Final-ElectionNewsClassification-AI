# code modified from ChatGPT, https://chatgpt.com/share/673a6474-d5a0-800f-8f90-ffa638288ef0
import pandas as pd
from sklearn.model_selection import train_test_split

training_csv = 'training_data.csv'
test_csv = 'test_data.csv'
validation_cat = 'val_data.csv'

# Load the CSV file into a DataFrame
df = pd.read_csv(training_csv)

# Shuffle the data (important for randomness)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Split the data into 8:1:1 ratio
train, temp = train_test_split(df, test_size=0.2, random_state=42)  # 80% for training
val, test = train_test_split(temp, test_size=0.5, random_state=42)  # 50% of 20% for validation and test

# Save the partitions to separate CSV files
train.to_csv(training_csv, index=False)
val.to_csv(test_csv, index=False)
test.to_csv(validation_cat, index=False)