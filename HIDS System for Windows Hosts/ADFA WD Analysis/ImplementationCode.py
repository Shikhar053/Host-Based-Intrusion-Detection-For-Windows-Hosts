import os
import numpy as np
import random
from collections import Counter
from sklearn.feature_selection import mutual_info_classif
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
#from sklearn.metrics import recall_score

### 1. Load and Balance ADFA-WD Dataset ###
def load_adfa_wd(root_path):
    normal_dir = os.path.join(root_path, "Full_Trace_Training_Data")
    validation_dir = os.path.join(root_path, "Full_Trace_Validation_Data")
    attack_dir = os.path.join(root_path, "Full_Trace_Attack_Data")

    # Collect normal sequences
    normal_files = [os.path.join(normal_dir, f) for f in os.listdir(normal_dir) if f.endswith(".GHC")]
    validation_files = [os.path.join(validation_dir, f) for f in os.listdir(validation_dir) if f.endswith(".GHC")]
    normal_files.extend(validation_files)

    normal_sequences = [open(f).read().strip() for f in normal_files]
    attack_sequences = []
    
    # Collect attack sequences (randomly selecting 40% of attack data)
    for attack_type in os.listdir(attack_dir):
        attack_path = os.path.join(attack_dir, attack_type)
        if os.path.isdir(attack_path):
            attack_files = [os.path.join(attack_path, f) for f in os.listdir(attack_path) if f.endswith(".GHC")]
            selected_files = random.sample(attack_files, int(0.4 * len(attack_files)))  # Random 40%
            attack_sequences.extend([open(f).read().strip() for f in selected_files])

    # Balance dataset (2182 normal + 2217 attack)
    min_size = min(len(normal_sequences), len(attack_sequences))
    normal_sequences = random.sample(normal_sequences, min_size)
    attack_sequences = random.sample(attack_sequences, min_size)

    X_data = normal_sequences + attack_sequences
    y_labels = [0] * len(normal_sequences) + [1] * len(attack_sequences)  # 0 = Normal, 1 = Attack
    return X_data, y_labels

### 2. Extract First N System Calls ###
def select_first_n_calls(sequences, N=100):
    return [" ".join(seq.split()[:N]) for seq in sequences]

### 3. Convert to 5-Gram BoW Representation ###
def extract_features_bow(sequences):
    ngram_size = 5
    all_ngrams = []
    
    for seq in sequences:
        calls = seq.split()
        ngrams = [" ".join(calls[i:i + ngram_size]) for i in range(len(calls) - ngram_size + 1)]
        all_ngrams.append(ngrams)

    unique_ngrams = set(ngram for seq in all_ngrams for ngram in seq)
    vocab = {ngram: idx for idx, ngram in enumerate(unique_ngrams)}
    
    X = np.zeros((len(sequences), len(vocab)))

    for i, ngrams in enumerate(all_ngrams):
        ngram_counts = Counter(ngrams)
        for ngram, count in ngram_counts.items():
            X[i, vocab[ngram]] = count  

    return X, list(vocab.keys())

### 4. Feature Selection using Mutual Information (Algorithm 1) ###
def feature_selection_mi(X, y):
    # Step 2: Calculate MI values
    mi_values = mutual_info_classif(X, y)
    
    # Step 3: K-Means clustering on MI values
    kmeans = KMeans(n_clusters=2, random_state=42)
    clusters = kmeans.fit_predict(mi_values.reshape(-1, 1))
    
    # Step 4: Select features from the cluster with high MI values
    selected_features = np.where(clusters == np.argmax(kmeans.cluster_centers_))[0]
    
    # Step 5: Reduce dimensionality
    return X[:, selected_features]

### 5. Train & Evaluate Machine Learning Models ###
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import classification_report, accuracy_score

def train_evaluate_models(X_train, X_test, y_train, y_test):
    models = {
        "KNN": KNeighborsClassifier(),
        "Decision Tree": DecisionTreeClassifier(),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(),
        "XGBoost": XGBClassifier(),
        "AdaBoost": AdaBoostClassifier(algorithm="SAMME")
    }

    results = {}

    for model_name, model in models.items():
        print("=" * 80)
        print(f" Training and Evaluating: {model_name}...\n")

        # Train model
        model.fit(X_train, y_train)

        # Predictions
        y_pred = model.predict(X_test)

        # Accuracy
        accuracy = accuracy_score(y_test, y_pred)

        # Classification Report
        report = classification_report(y_test, y_pred, digits=6, output_dict=True)

        print(f" Classification Report for {model_name}:\n")
        print(classification_report(y_test, y_pred, digits=6))

        print(f" Accuracy: {accuracy:.6f}")
        print("=" * 80, "\n")

        # Store results
        results[model_name] = {
            "accuracy": accuracy,
            "classification_report": report
        }

    return results

### MAIN EXECUTION ###
root_path = "Full_Process_Traces"  # Change this to your dataset path
X_data, y_labels = load_adfa_wd(root_path)

for N in [100, 200, 300, 400, 500]:
    print(f"\nRunning for N = {N} system calls...")
    
    # Step 1: Select First N Calls
    X_selected = select_first_n_calls(X_data, N)
    
    # Step 2: Extract 5-Gram BoW Features
    X_bow, vocab = extract_features_bow(X_selected)
    
    # Step 3: Feature Selection via MI + K-Means
    X_selected_features = feature_selection_mi(X_bow, y_labels)
    
    # Step 4: Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X_selected_features, y_labels, test_size=0.25, random_state=None)
    
    # Step 5: Train & Evaluate Models
    results = train_evaluate_models(X_train, X_test, y_train, y_test)
    
   # print(f"Results for N={N}: {results}")
