import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import learning_curve
from sklearn.metrics import precision_recall_curve, auc

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

# Fit the model (training the model)
model.fit(X_train, y_train)

# Use learning_curve function to calculate training and test scores
train_sizes, train_scores, test_scores = learning_curve(
    model, X_train, y_train, cv=5, n_jobs=-1, train_sizes=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
)

# Calculate the mean and standard deviation of training and test scores
train_mean = train_scores.mean(axis=1)
test_mean = test_scores.mean(axis=1)
train_std = train_scores.std(axis=1)
test_std = test_scores.std(axis=1)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Print results
print(f"Accuracy: {accuracy:.4f}")
print("Confusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Plotting the learning curve
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_mean, color='blue', label='Training Score')
plt.plot(train_sizes, test_mean, color='red', label='Cross-validation Score')
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color='blue', alpha=0.1)
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, color='red', alpha=0.1)

plt.title("Learning Curve (Logistic Regression)")
plt.xlabel("Training Size")
plt.ylabel("Accuracy")
plt.legend(loc='best')
plt.grid(True)
plt.show()  

import numpy as np
import shap

def truncated_monte_carlo_shapley_value(model, X, feature_index, num_permutations=100):
    """
    Compute the Shapley value for a single feature using Truncated Monte Carlo method.
    
    Parameters:
    - model: Trained model (e.g., LogisticRegression).
    - X: Feature matrix (2D array or DataFrame).
    - feature_index: Index of the feature for which we want to compute the Shapley value.
    - num_permutations: Number of random permutations to sample for Monte Carlo estimation.
    
    Returns:
    - Shapley value for the specified feature.
    """
    n_samples, n_features = X.shape
    shapley_value = 0
    
    for _ in range(num_permutations):
        # Randomly sample a subset of features
        perm = np.random.permutation(n_features)
        
        # Create a mask for the subset that includes the feature of interest
        mask_before = np.isin(perm[:np.random.randint(1, n_features)], feature_index)
        mask_after = np.isin(perm[np.random.randint(1, n_features):], feature_index)
        
        # Estimate contribution to Shapley value for the feature of interest
        X_before = X[mask_before]
        X_after = X[mask_after]
        
        prediction_before = model.predict_proba(X_before)[:, 1]
        prediction_after = model.predict_proba(X_after)[:, 1]
        
        shapley_value += prediction_after.mean() - prediction_before.mean()
    
    # Return the average Shapley value over all permutations
    return shapley_value / num_permutations


# Apply the truncated Monte Carlo method to compute Shapley values for each feature
shapley_values = []
for feature_index in range(X_train.shape[1]):
    shapley_values.append(truncated_monte_carlo_shapley_value(model, X_train, feature_index))

# Plot the Shapley values
plt.figure(figsize=(10, 6))
plt.bar(range(X_train.shape[1]), shapley_values)
plt.title('Feature Importance Based on Truncated Monte Carlo Shapley Values')
plt.xlabel('Feature Index')
plt.ylabel('Shapley Value')
plt.xticks(range(X_train.shape[1]), X.columns, rotation=90)
plt.show()


