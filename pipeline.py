import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import os
import sys

# Redirect output to a log file
class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open("pipeline_output.log", "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

sys.stdout = Logger()
sns.set(style="whitegrid", palette="muted")
plt.rcParams['figure.figsize'] = (12, 8)

# Create output directories
os.makedirs('plots', exist_ok=True)

columns = ['user', 'activity', 'timestamp', 'x-axis', 'y-axis', 'z-axis']

def load_and_clean_data(file_path):
    print("Step 1: Loading and Cleaning Data...")
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            if line.endswith(';'): line = line[:-1]
            parts = [p.strip() for p in line.split(',')]
            if len(parts) == 6:
                try:
                    user = int(parts[0])
                    activity = parts[1]
                    timestamp = int(float(parts[2]))
                    x = float(parts[3])
                    y = float(parts[4])
                    z = float(parts[5])
                    data.append([user, activity, timestamp, x, y, z])
                except ValueError: continue
    
    df = pd.DataFrame(data, columns=columns)
    # Basic cleaning: check for nulls (already filtered by loading logic)
    # Check for duplicates
    initial_count = len(df)
    print(f"Initial count: {initial_count}")
    df.drop_duplicates(inplace=True)
    if len(df) < initial_count:
        print(f"Removed {initial_count - len(df)} duplicate rows,Empty lines,Extra semicolons.")
    
    return df

def perform_eda(df):
    print("Step 2: Performing Exploratory Data Analysis...")
    
    # Activity Distribution
    plt.figure()
    sns.countplot(x='activity', data=df, order=df['activity'].value_counts().index)
    plt.title('Distribution of Activities')
    plt.xticks(rotation=45)
    plt.savefig('plots/activity_distribution.png')
    plt.close()

    # Accelerometer Distributions
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    sns.histplot(df['x-axis'], ax=axes[0], kde=True, color='red').set_title('X-axis Distribution')
    sns.histplot(df['y-axis'], ax=axes[1], kde=True, color='green').set_title('Y-axis Distribution')
    sns.histplot(df['z-axis'], ax=axes[2], kde=True, color='blue').set_title('Z-axis Distribution')
    plt.tight_layout()
    plt.savefig('plots/accel_distributions.png')
    plt.close()

    # Boxplots for each axis vs Activity
    fig, axes = plt.subplots(3, 1, figsize=(12, 18))
    sns.boxplot(x='activity', y='x-axis', data=df, ax=axes[0]).set_title('X-axis by Activity')
    sns.boxplot(x='activity', y='y-axis', data=df, ax=axes[1]).set_title('Y-axis by Activity')
    sns.boxplot(x='activity', y='z-axis', data=df, ax=axes[2]).set_title('Z-axis by Activity')
    plt.tight_layout()
    plt.savefig('plots/accel_by_activity.png')
    plt.close()

    # Correlation Matrix
    plt.figure()
    sns.heatmap(df[['x-axis', 'y-axis', 'z-axis']].corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation between Accelerometer Axes')
    plt.savefig('plots/correlation_matrix.png')
    plt.close()

def preprocess_data(df):
    print("Step 3: Preprocessing Data...")
    
    # Categorical Encoding for Target
    le = LabelEncoder()
    df['activity_encoded'] = le.fit_transform(df['activity'])
    
    # Features and Target
    X = df[['x-axis', 'y-axis', 'z-axis']]
    y = df['activity_encoded']
    
    # Standardization
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, le, scaler

def train_and_evaluate(X, y, label_encoder):
    print("Step 4: Model Selection and Training...")
    
    # Since the dataset is large, we'll use a subset for faster model selection
    # but for final training let's use a decent chunk
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Subsample for faster execution in this environment if necessary
    # Let's take 100k rows if total is > 1M
    if len(X_train) > 100000:
        print("Subsampling for faster training...")
        subset_indices = np.random.choice(len(X_train), 100000, replace=False)
        X_train_sub = X_train[subset_indices]
        y_train_sub = y_train.iloc[subset_indices]
    else:
        X_train_sub = X_train
        y_train_sub = y_train

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
    }
    
    best_model = None
    best_f1 = 0
    best_name = ""
    
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train_sub, y_train_sub)
        y_pred = model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')
        
        print(f"{name} Results:")
        print(f"  - Accuracy: {acc:.4f}")
        print(f"  - Macro F1-Score: {f1:.4f} (Primary Metric for Imbalance)")
        
        if f1 > best_f1:
            best_f1 = f1
            best_model = model
            best_name = name

    print(f"\nBest Model Selection based on F1-Score: {best_name}")
    print(f"Final Macro F1-Score: {best_f1:.4f}")
    
    # Final Evaluation on Best Model
    y_pred = best_model.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - {best_name}')
    plt.savefig('plots/confusion_matrix.png')
    plt.close()

if __name__ == "__main__":
    file_path = 'WISDM_ar_v1.1_raw.txt'
    df = load_and_clean_data(file_path)
    perform_eda(df)
    X, y, le, scaler = preprocess_data(df)
    train_and_evaluate(X, y, le)
    print("\nAll steps completed. Plots saved in 'plots/' directory.")
    
    # Automatically open the generated plots (Windows specific)
    plot_files = [
        'plots/activity_distribution.png',
        'plots/accel_distributions.png',
        'plots/accel_by_activity.png',
        'plots/correlation_matrix.png',
        'plots/confusion_matrix.png'
    ]
    
    print("\nOpening generated plots...")
    for plot in plot_files:
        if os.path.exists(plot):
            os.startfile(os.path.abspath(plot))
