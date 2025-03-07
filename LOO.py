import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("credit_card_default.csv")

# Remove 'Unnamed' columns if they exist, and rename the target column if needed
df = df.drop(columns=[col for col in df.columns if "Unnamed" in col])  # drop columns with 'Unnamed'
df = df.rename(columns={"Y": "default payment next month"})  # rename the target column

# Separate features (X) and target (y)
X = df.drop(columns=["default payment next month"])  # Features
y = df["default payment next month"]  # Target

# Handle categorical variables
X = pd.get_dummies(X, columns=["SEX", "EDUCATION", "MARRIAGE"], drop_first=True)

# Standardize numerical features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate the model on the full training set
full_train_accuracy = accuracy_score(y_train, model.predict(X_train))

# Select 10 random data points from the training set
random_indices = np.random.choice(range(len(X_train)), size=10, replace=False)
influence_scores = []

# Compute LOO influence for each selected data point
for idx in random_indices:
    # Create a new training set by removing the selected data point
    X_train_loo = np.delete(X_train, idx, axis=0)
    y_train_loo = np.delete(y_train, idx, axis=0)
    
    # Train the model on the new training set
    model.fit(X_train_loo, y_train_loo)
    
    # Compute accuracy on the new training set
    loo_accuracy = accuracy_score(y_train_loo, model.predict(X_train_loo))
    
    # Compute the influence score
    influence_score = full_train_accuracy - loo_accuracy
    influence_scores.append(influence_score)

# Create a DataFrame to display the results
loo_df = pd.DataFrame({
    'Index': random_indices,
    'Influence Score': influence_scores
})

# Display the results
print("Leave-One-Out Influence Scores for 10 Random Data Points:")
print(loo_df)

# Plot the influence scores
plt.figure(figsize=(8, 6))
plt.bar(loo_df['Index'].astype(str), loo_df['Influence Score'], color='blue')
plt.xlabel('Data Point Index')
plt.ylabel('Influence Score')
plt.title('Leave-One-Out Influence Scores')
plt.grid(True)
plt.show()

