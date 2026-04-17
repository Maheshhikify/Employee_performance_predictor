# Create folders

import os

folders = [
    "data/raw",
    "data/processed",
    "models",
    "images",
    "outputs"
]

for folder in folders:
    os.makedirs(folder, exist_ok=True)

print("Folders created successfully!")

# STEP 3 — Create Synthetic Dataset

import pandas as pd
import numpy as np

np.random.seed(42)

n = 1000   # Number of employees

data = {

    "employee_id": range(1, n+1),

    "age": np.random.randint(22, 60, n),

    "experience_years":
        np.random.randint(0, 35, n),

    "department":
        np.random.choice(
            ["HR", "IT", "Sales", "Finance"],
            n
        ),

    "education":
        np.random.choice(
            ["Bachelors", "Masters", "PhD"],
            n
        ),

    "job_level":
        np.random.randint(1, 6, n),

    "projects_completed":
        np.random.randint(1, 20, n),

    "training_hours":
        np.random.randint(0, 100, n),

    "avg_task_delay_days":
        np.random.randint(0, 10, n),

    "on_time_delivery_rate":
        np.random.uniform(0.5, 1.0, n),

    "bug_count":
        np.random.randint(0, 50, n),

    "peer_feedback_score":
        np.random.uniform(1, 5, n),

    "manager_score":
        np.random.uniform(1, 5, n),

    "attendance_rate":
        np.random.uniform(0.6, 1.0, n),

    "salary_percentile":
        np.random.randint(1, 100, n)

}

df = pd.DataFrame(data)

# Create Performance Score

performance_score = (
    df["projects_completed"] * 0.3
    + df["training_hours"] * 0.2
    + df["on_time_delivery_rate"] * 30
    + df["manager_score"] * 10
    - df["avg_task_delay_days"] * 2
)

conditions = [
    performance_score > 80,
    performance_score > 60
]

choices = ["High", "Medium"]

df["performance_band"] = np.select(
    conditions,
    choices,
    default="Low"
)

# Save dataset

df.to_csv(
    "data/raw/employee_data.csv",
    index=False
)

print("Dataset created successfully!")

df.head()


# STEP 4 — Data Cleaning

df = pd.read_csv(
    "data/raw/employee_data.csv"
)

print("Before Cleaning:", df.shape)

# Remove duplicates

df = df.drop_duplicates()

# Fill missing values

df = df.fillna(
    df.median(numeric_only=True)
)

print("After Cleaning:", df.shape)

# Save cleaned data

df.to_csv(
    "data/processed/cleaned_employee_data.csv",
    index=False
)

print("Data cleaning completed!")

# STEP 5 — EDA

import matplotlib.pyplot as plt
import seaborn as sns

# Performance Distribution

plt.figure()

sns.countplot(
    x="performance_band",
    data=df
)

plt.title(
    "Performance Band Distribution"
)

plt.savefig(
    "images/performance_distribution.png"
)

plt.show()

# STEP 6 — Correlation Heatmap

plt.figure(figsize=(10,8))

sns.heatmap(
    df.corr(numeric_only=True),
    cmap="coolwarm"
)

plt.title(
    "Correlation Heatmap"
)

plt.savefig(
    "images/correlation_heatmap.png"
)

plt.show()

# STEP 7 — Data Preparation

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Encode target

le = LabelEncoder()

df["performance_band"] = le.fit_transform(
    df["performance_band"]
)

# Separate features

X = df.drop(
    ["performance_band", "employee_id"],
    axis=1
)

y = df["performance_band"]

# Convert categorical to numeric

X = pd.get_dummies(X)

# Train-test split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Data prepared successfully!")

# STEP 8 — Train Model

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()

model.fit(
    X_train,
    y_train
)

print("Model trained successfully!")

# STEP 9 — Model Evaluation

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

y_pred = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("Model Accuracy:", accuracy)

print("\nClassification Report:\n")

print(
    classification_report(
        y_test,
        y_pred
    )
)

cm = confusion_matrix(
    y_test,
    y_pred
)

sns.heatmap(
    cm,
    annot=True,
    fmt="d"
)

plt.title(
    "Confusion Matrix"
)

plt.savefig(
    "images/confusion_matrix.png"
)

plt.show()

# STEP 10 — Feature Importance

import pandas as pd

importances = model.feature_importances_

feature_names = X.columns

feature_importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": importances
})

feature_importance_df = feature_importance_df.sort_values(
    by="importance",
    ascending=False
)

# Plot Top Features

plt.figure(figsize=(10,6))

sns.barplot(
    x="importance",
    y="feature",
    data=feature_importance_df.head(10)
)

plt.title(
    "Top Feature Importance"
)

plt.savefig(
    "images/feature_importance.png"
)

plt.show()

# STEP 11 — Save Model

import joblib

joblib.dump(
    model,
    "models/performance_model.pkl"
)

print("Model saved successfully!")

# STEP 12 — Prediction

sample = X_test.iloc[0:5]

predictions = model.predict(sample)

# Convert numbers back to labels

predictions = le.inverse_transform(
    predictions
)

print("Predictions:\n")

print(predictions)