import pandas as pd
import joblib
import os
import numpy as np

from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score
)

from config.config import TRAIT_COLUMNS, MODEL_PATH


# =====================================================
# 🔥 TRAIN + SAVE LINEAR SVM MODEL
# =====================================================
def train_and_save_model(train_path="train_dataset.csv", test_path="test_dataset.csv"):
    """
    Trains ONLY a Linear SVM model using 5-Fold CV.
    Saves the SVM model + CV metrics + feature list.
    """

    print("Loading training and testing datasets...")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    print(f"✅ Training data shape: {train_df.shape}")
    print(f"✅ Testing data shape: {test_df.shape}")

    # Clean datasets
    for df in [train_df, test_df]:
        if "Video_ID" in df.columns:
            df.drop(columns=["Video_ID"], inplace=True)
        df.dropna(subset=["is_autistic"], inplace=True)

    # Feature split
    X_train = train_df.drop(columns=["is_autistic"])
    y_train = train_df["is_autistic"]
    X_test = test_df.drop(columns=["is_autistic"])
    y_test = test_df["is_autistic"]

    # Align feature columns
    shared_features = X_train.columns.intersection(X_test.columns).tolist()
    X_train = X_train[shared_features]
    X_test = X_test[shared_features]

    print(f"✅ Using {len(shared_features)} shared features")

    # =====================================================
    # 🔥 LINEAR SVM + 5-FOLD CROSS VALIDATION
    # =====================================================
    print("\nTraining Linear SVM with 5-Fold Cross Validation...")

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    acc_list, prec_list, rec_list, f1_list = [], [], [], []

    fold = 1
    best_model = None  # we will save the last trained full model

    for train_idx, valid_idx in kf.split(X_train, y_train):

        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[valid_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[valid_idx]

        svm_model = SVC(kernel='linear', probability=True)
        svm_model.fit(X_tr, y_tr)

        y_val_pred = svm_model.predict(X_val)

        acc_list.append(accuracy_score(y_val, y_val_pred))
        prec_list.append(precision_score(y_val, y_val_pred, zero_division=0))
        rec_list.append(recall_score(y_val, y_val_pred, zero_division=0))
        f1_list.append(f1_score(y_val, y_val_pred, zero_division=0))

        print(f"Fold {fold}: ACC={acc_list[-1]:.4f}, F1={f1_list[-1]:.4f}")
        fold += 1

        best_model = svm_model  # save last model

    # CV metrics summary
    svm_cv_metrics = {
        "cv_accuracy_mean": float(np.mean(acc_list)),
        "cv_precision_mean": float(np.mean(prec_list)),
        "cv_recall_mean": float(np.mean(rec_list)),
        "cv_f1_mean": float(np.mean(f1_list)),
        "cv_accuracy_all": acc_list,
        "cv_f1_all": f1_list
    }

    print("\n--- LINEAR SVM 5-FOLD CV RESULTS ---")
    print(svm_cv_metrics)

    # =====================================================
    # 🔥 SAVE ONLY SVM MODEL
    # =====================================================
    joblib.dump({
        "model": best_model,
        "metrics": svm_cv_metrics,
        "features": shared_features
    }, MODEL_PATH)

    print(f"\n💾 Linear SVM model + metrics saved to '{MODEL_PATH}'")

    print(svm_cv_metrics)

    return best_model, svm_cv_metrics



# =====================================================
# 🔥 LOAD MODEL
# =====================================================
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Model not found. Please train a model first.")

    print(f"Loading SVM model from '{MODEL_PATH}'...")
    data = joblib.load(MODEL_PATH)

    return data["model"], data.get("metrics", {})



# =====================================================
# 🔥 GENERATE PATIENT PREDICTION REPORT
# =====================================================
def generate_prediction_and_report(features, model_data=None):

    print("\n--- Generating Patient Report ---")

    if model_data is None:
        data = joblib.load(MODEL_PATH)
        model = data["model"]
        metrics = data.get("metrics", {})
        model_features = data.get("features", [])
    else:
        model, metrics = model_data
        saved = joblib.load(MODEL_PATH)
        model_features = saved.get("features", [])

    # Convert to DataFrame
    input_df = pd.DataFrame([features])

    # Align missing features
    for f in model_features:
        if f not in input_df:
            input_df[f] = 0

    input_df = input_df[model_features]

    # Predict
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)[0]

    predicted_label = "AUTISTIC" if prediction[0] == 1 else "NOT AUTISTIC"
    confidence = prediction_proba[1] if prediction[0] == 1 else prediction_proba[0]

    report = {
        "detected_traits": features,
        "final_prediction": {
            "label": predicted_label,
            "confidence": f"{confidence:.2%}",
            "likelihood_score": round(float(confidence), 4)
        },
        "model_info": {
            "used_features": model_features,
            "performance": metrics
        }
    }

    print(f"✅ Prediction: {predicted_label} (Confidence: {confidence:.2%})")
    return report
