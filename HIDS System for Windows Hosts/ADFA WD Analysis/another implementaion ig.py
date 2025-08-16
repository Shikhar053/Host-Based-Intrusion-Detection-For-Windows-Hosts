import os
import numpy as np
import random
import warnings
from collections import defaultdict
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, precision_recall_fscore_support, accuracy_score
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

    normal_files = [os.path.join(normal_dir, f) for f in os.listdir(normal_dir) if f.endswith(".GHC")]
    validation_files = [os.path.join(validation_dir, f) for f in os.listdir(validation_dir) if f.endswith(".GHC")]
    normal_files.extend(validation_files)
    normal_sequences = [open(f).read().strip() for f in normal_files]

    attack_sequences = []
    for attack_type in os.listdir(attack_dir):
        attack_path = os.path.join(attack_dir, attack_type)
        if os.path.isdir(attack_path):
            attack_files = [os.path.join(attack_path, f) for f in os.listdir(attack_path) if f.endswith(".GHC")]
            attack_sequences.extend(attack_files)

    selected_attack_files = random.sample(attack_sequences, 2182)
    attack_sequences = [open(f).read().strip() for f in selected_attack_files]

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

# 4. Final Stacking Ensemble
def stacking_ensemble(X, y):
    base_models = {
        "KNN": KNeighborsClassifier(),
        "DT": DecisionTreeClassifier(random_state=42),
        "RF": RandomForestClassifier(random_state=42)
    }

    meta_model_variants = {
        "XGBoost_default": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
        "XGBoost_tuned": XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1, subsample=0.8,
                                       colsample_bytree=0.8, use_label_encoder=False, eval_metric='logloss', random_state=42),

        "AdaBoost_default": AdaBoostClassifier(random_state=42),
        "AdaBoost_tuned": AdaBoostClassifier(n_estimators=150, learning_rate=0.5, random_state=42),

        "LightGBM_default": LGBMClassifier(verbose=-1, random_state=42),
        "LightGBM_tuned": LGBMClassifier(n_estimators=200, num_leaves=40, learning_rate=0.1, max_depth=5,
                                         verbose=-1, random_state=42)
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    base_preds_all = {name: [] for name in base_models}
    meta_preds_all = {name: [] for name in meta_model_variants}
    meta_metrics = defaultdict(list)
    true_labels = []

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = np.array(y)[train_idx], np.array(y)[test_idx]
        true_labels.extend(y_test)

        base_outputs = []

        for name, model in base_models.items():
            model.fit(X_train, y_train)
            probs = model.predict_proba(X_test)
            attack_probs = probs[:, 1]
            base_preds_all[name].extend(np.round(attack_probs))
            base_outputs.append(attack_probs.reshape(-1, 1))

        stacked_X = np.hstack(base_outputs)

        for name, meta in meta_model_variants.items():
            meta.fit(stacked_X, y_test)
            meta_preds = meta.predict(stacked_X)
            meta_preds_all[name].extend(meta_preds)

            precision, recall, f1, _ = precision_recall_fscore_support(y_test, meta_preds, average='binary')
            acc = accuracy_score(y_test, meta_preds)
            meta_metrics[name].append((precision, recall, f1, acc))

    print("\n\n===== BASE MODEL RESULTS =====")
    for name in base_models:
        print(f"\n=== Base Model: {name} ===")
        print(classification_report(true_labels, base_preds_all[name], target_names=["Normal", "Attack"], digits=4))

    print("\n\n===== STACKING META MODEL RESULTS (All Variants) =====")
    for name in meta_model_variants:
        print(f"\n=== Meta Model: {name} ===")
        print(classification_report(true_labels, meta_preds_all[name], target_names=["Normal", "Attack"], digits=4))

    print("\n\n===== AVERAGED META MODEL SCORES =====")
    for name, scores in meta_metrics.items():
        avg = np.mean(scores, axis=0)
        print(f"{name}: Precision={avg[0]:.4f}, Recall={avg[1]:.4f}, F1={avg[2]:.4f}, Accuracy={avg[3]:.4f}")

# === FINAL RUN: COMBINED FEATURE MATRIX FROM MULTIPLE N ===
if __name__ == "__main__":
    path = "Full_Process_Traces"  # <-- Update if needed
    X_data, y_labels = load_adfa_wd_exact_balanced(path)

    X_all = []
    print("\nExtracting and combining features from multiple N values...")

    for N in [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]:
        print(f"Processing N = {N}")
        X_n = select_first_n_calls(X_data, N)
        X_tfidf_n, _ = extract_features_tfidf(X_n)
        X_all.append(X_tfidf_n)

    X_combined = hstack(X_all)
    print("\nRunning Stacking Ensemble on Combined Feature Set...")
    stacking_ensemble(X_combined, y_labels)
