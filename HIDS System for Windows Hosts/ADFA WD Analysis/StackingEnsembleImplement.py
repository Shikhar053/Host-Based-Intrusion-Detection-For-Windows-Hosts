import os
import numpy as np
import random
import warnings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import mutual_info_classif
from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")
random.seed(42)
np.random.seed(42)

# 1. Load and balance ADFA-WD
def load_adfa_wd_exact_balanced(root_path):
    normal_dir = os.path.join(root_path, "Full_Trace_Training_Data")
    validation_dir = os.path.join(root_path, "Full_Trace_Validation_Data")
    attack_dir = os.path.join(root_path, "Full_Trace_Attack_Data")

    # Normal Sequences: Training + Validation = 2182
    normal_files = [os.path.join(normal_dir, f) for f in os.listdir(normal_dir) if f.endswith(".GHC")]
    validation_files = [os.path.join(validation_dir, f) for f in os.listdir(validation_dir) if f.endswith(".GHC")]
    normal_files.extend(validation_files)  # total 2182 normal
    normal_sequences = [open(f).read().strip() for f in normal_files]

    # Attack Sequences: sample exactly 2182
    attack_sequences = []
    for attack_type in os.listdir(attack_dir):
        attack_path = os.path.join(attack_dir, attack_type)
        if os.path.isdir(attack_path):
            attack_files = [os.path.join(attack_path, f) for f in os.listdir(attack_path) if f.endswith(".GHC")]
            attack_sequences.extend(attack_files)

    # Random sample of 2182 attack files
    selected_attack_files = random.sample(attack_sequences, 2182)
    attack_sequences = [open(f).read().strip() for f in selected_attack_files]

    # Labels
    X_data = normal_sequences + attack_sequences
    y_labels = [0] * len(normal_sequences) + [1] * len(attack_sequences)
    return X_data, y_labels

# 2. Use first N system calls
def select_first_n_calls(sequences, N=500):
    return [" ".join(seq.split()[:N]) for seq in sequences]

# 3. TF-IDF 5-gram features
def extract_features_tfidf(sequences):
    tfidf = TfidfVectorizer(ngram_range=(5,5))
    X = tfidf.fit_transform(sequences)
    return X, tfidf.get_feature_names_out()

# 4. Feature selection with MI + KMeans
def feature_selection_mi(X, y):
    mi = mutual_info_classif(X, y)
    kmeans = KMeans(n_clusters=2, random_state=42)
    clusters = kmeans.fit_predict(mi.reshape(-1, 1))
    selected = np.where(clusters == np.argmax(kmeans.cluster_centers_))[0]
    return X[:, selected]

# 5. Final Stacking Ensemble
def stacking_ensemble(X, y):
    base_models = {
    "KNN": KNeighborsClassifier(),
    "DT": DecisionTreeClassifier(random_state=42),
    "RF": RandomForestClassifier(random_state=42)
    }
    meta_models = {
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
        "AdaBoost": AdaBoostClassifier(random_state=42),
        "LightGBM": LGBMClassifier(verbose=-1, random_state=42)
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # To collect predictions
    base_preds_all = {name: [] for name in base_models}
    meta_preds_all = {name: [] for name in meta_models}
    true_labels = []

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = np.array(y)[train_idx], np.array(y)[test_idx]
        true_labels.extend(y_test)

        base_outputs = []

        for name, model in base_models.items():
            model.fit(X_train, y_train)
            probs = model.predict_proba(X_test)
            # Keep only the probability of class 1 (Attack)
            attack_probs = probs[:, 1]
            base_preds_all[name].extend(np.round(attack_probs))
            base_outputs.append(attack_probs.reshape(-1, 1))


        stacked_X = np.hstack(base_outputs)

        for name, meta in meta_models.items():
            meta.fit(stacked_X, y_test)
            meta_preds = meta.predict(stacked_X)
            meta_preds_all[name].extend(meta_preds)

    # ==== Print base model reports ====
    print("\n\n===== BASE MODEL RESULTS =====")
    for name in base_models:
        print(f"\n=== Base Model: {name} ===")
        print(classification_report(true_labels, base_preds_all[name], target_names=["Normal", "Attack"], digits=4))

    # ==== Print meta model reports ====
    print("\n\n===== STACKING META MODEL RESULTS =====")
    for name in meta_models:
        print(f"\n=== Meta Model: {name} ===")
        print(classification_report(true_labels, meta_preds_all[name], target_names=["Normal", "Attack"], digits=4))

# === FINAL RUN: COMBINED FEATURE MATRIX FROM MULTIPLE N ===
if __name__ == "__main__":
    path = "Full_Process_Traces"  # <-- Your dataset path
    X_data, y_labels = load_adfa_wd_exact_balanced(path)

    X_all = []
    print("\nExtracting and combining features from multiple N values...")

    for N in [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]:
        print(f"Processing N = {N}")
        X_n = select_first_n_calls(X_data, N)
        X_tfidf_n, _ = extract_features_tfidf(X_n)
        X_all.append(X_tfidf_n)

    # Stack everything into one big feature matrix
    from scipy.sparse import hstack
    X_combined = hstack(X_all)

    print("\nRunning Stacking Ensemble on Combined Feature Set...")
    stacking_ensemble(X_combined, y_labels)
