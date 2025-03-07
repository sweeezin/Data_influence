import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the dataset
df = pd.read_csv("credit_card_default.csv")

# Remove 'Unnamed' columns if they exist, and rename the target column if needed
df = df.drop(columns=[col for col in df.columns if "Unnamed" in col])  # drop columns with 'Unnamed'
df = df.rename(columns={"Y": "default payment next month"})  # rename the target column

# Now separate features (X) and target (y)
X = df.drop(columns=["default payment next month"])  # We don't need 'default payment next month' here
y = df["default payment next month"]

# Handle categorical variables
# Convert categorical variables to numerical (one-hot encoding)
X = pd.get_dummies(X, columns=["SEX", "EDUCATION", "MARRIAGE"], drop_first=True)

# Standardize numerical features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define Logistic Regression model
model = LogisticRegression(max_iter=1000)

# Train the model on the entire training set
model.fit(X_train, y_train)

# Get the full accuracy on the training set (to compute influence scores)
full_train_accuracy = accuracy_score(y_train, model.predict(X_train))

# Store influence scores
group_influence_scores = []
valid_group_sizes = []

# Define group sizes (10%, 20%, 30%, etc.)
group_sizes = [int(0.1 * len(X_train)), int(0.2 * len(X_train)), int(0.3 * len(X_train)), 
               int(0.4 * len(X_train)), int(0.5 * len(X_train)), int(0.6 * len(X_train)), 
               int(0.7 * len(X_train)), int(0.8 * len(X_train)), int(0.9 * len(X_train)), 
               len(X_train)]  # 100% group size

# Compute Leave-Entire-Group-Out (LEGO) influence for each group
for size in group_sizes:
    # Randomly select a group of data points
    group_indices = np.random.choice(range(len(X_train)), size=size, replace=False)
    
    # Create a new training set by removing the selected group
    X_train_group_out = np.delete(X_train, group_indices, axis=0)
    y_train_group_out = np.delete(y_train, group_indices, axis=0)
    
    # Ensure the new training set is not empty
    if len(X_train_group_out) > 0:
        # Train the model on the new training set
        model.fit(X_train_group_out, y_train_group_out)
        
        # Compute accuracy on the new training set
        group_accuracy = accuracy_score(y_train_group_out, model.predict(X_train_group_out))
        
        # Compute the influence score for the group
        group_influence_score = full_train_accuracy - group_accuracy
        group_influence_scores.append(group_influence_score)
        valid_group_sizes.append(size)
    else:
        print(f"Skipping group of size {size}% because it would leave the training set empty.")

# Plot the influence scores vs group sizes
plt.figure(figsize=(8, 6))
plt.plot(valid_group_sizes, group_influence_scores, marker='o', color='b', label='Group Influence Score')
plt.xlabel('Group Size (%)')
plt.ylabel('Influence Score')
plt.title('Group Size vs Influence Score')
plt.grid(True)
plt.legend(loc='best')
plt.show()

# Report the influence scores in a table
influence_df = pd.DataFrame({
    'Group Size (%)': valid_group_sizes,
    'Influence Score': group_influence_scores
})

print(influence_df)
