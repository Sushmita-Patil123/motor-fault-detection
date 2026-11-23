import os
import sys
import json
import argparse
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

FEATURE_COLS = ['voltage', 'current_r', 'current_y', 'current_b', 'rpm', 'power_input', 'power_factor']

def main():
    parser = argparse.ArgumentParser(description="Train ML Models on 3-Phase Induction Motor Electrical Parameters")
    parser.add_argument("dataset", nargs="?", default=r"c:\FAULT_DETECTION\dataset\induction_motor_electrical_dataset.csv", help="Path to dataset")
    args = parser.parse_args()
    
    dataset_path = args.dataset.strip().strip('<>').strip('"').strip("'")
    workspace = r"c:\FAULT_DETECTION"
    models_dir = os.path.join(workspace, "models")
    os.makedirs(models_dir, exist_ok=True)
    
    if not os.path.exists(dataset_path):
        fallback = os.path.join(workspace, "dataset", "induction_motor_electrical_dataset.csv")
        if os.path.exists(fallback):
            dataset_path = fallback
        else:
            raise FileNotFoundError(f"Dataset not found at {dataset_path}")
            
    print(f"Loading dataset: {dataset_path}")
    df = pd.read_csv(dataset_path)
    print(f"Dataset shape: {df.shape}")
    
    target_col = 'fault' if 'fault' in df.columns else df.columns[-1]
    avail_features = [c for c in FEATURE_COLS if c in df.columns]
    
    print(f"\nTraining on Features ({len(avail_features)}): {avail_features}")
    X = df[avail_features]
    y = df[target_col]
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    class_names = le.classes_.tolist()
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.20, random_state=42, stratify=y_encoded
    )
    
    # Pipelines with standard scaling
    pipelines = {
        "Random Forest": Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1))
        ]),
        "Gradient Boosting": Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42))
        ]),
        "Support Vector Machine": Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', SVC(kernel='rbf', C=10.0, probability=True, random_state=42))
        ]),
        "K-Nearest Neighbors": Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', KNeighborsClassifier(n_neighbors=5, weights='distance'))
        ])
    }
    
    results = []
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    print("\n" + "="*70)
    print(f"{'Model':<26} | {'Test Acc':<10} | {'5-Fold CV Acc':<14} | {'F1 Score':<10}")
    print("="*70)
    
    best_score = -1.0
    best_model_name = None
    best_pipeline = None
    
    for name, pipe in pipelines.items():
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring='accuracy')
        
        test_acc = accuracy_score(y_test, y_pred)
        mean_cv = np.mean(cv_scores)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        results.append({
            "Model": name,
            "Test Accuracy": round(test_acc * 100, 2),
            "CV Accuracy (5-Fold)": round(mean_cv * 100, 2),
            "F1-Score": round(f1 * 100, 2)
        })
        
        print(f"{name:<26} | {test_acc*100:>8.2f}% | {mean_cv*100:>12.2f}% | {f1*100:>8.2f}%")
        
        if test_acc > best_score:
            best_score = test_acc
            best_model_name = name
            best_pipeline = pipe
            
    print("="*70)
    print(f"\nBest Performing Model: {best_model_name} ({best_score*100:.2f}%)")
    
    # Feature importances from Random Forest
    rf_pipe = pipelines["Random Forest"]
    rf_clf = rf_pipe.named_steps['classifier']
    feature_importance_dict = {}
    importances = rf_clf.feature_importances_
    sorted_indices = np.argsort(importances)[::-1]
    print("\nRandom Forest Feature Importance Ranking:")
    for idx in sorted_indices:
        feat_name = avail_features[idx]
        imp_val = round(float(importances[idx]) * 100, 2)
        feature_importance_dict[feat_name] = imp_val
        print(f"  • {feat_name:<22}: {imp_val:.2f}%")
        
    # Save Model Artifacts
    best_path = os.path.join(models_dir, "best_model.joblib")
    scaler_path = os.path.join(models_dir, "scaler.joblib")
    le_path = os.path.join(models_dir, "label_encoder.joblib")
    metrics_path = os.path.join(models_dir, "evaluation_metrics.json")
    comp_csv_path = os.path.join(models_dir, "model_comparison.csv")
    
    joblib.dump(best_pipeline, best_path)
    joblib.dump(best_pipeline.named_steps['scaler'], scaler_path)
    joblib.dump(le, le_path)
    
    df_results = pd.DataFrame(results)
    df_results.to_csv(comp_csv_path, index=False)
    
    y_best_pred = best_pipeline.predict(X_test)
    cls_report = classification_report(y_test, y_best_pred, target_names=class_names, output_dict=True)
    conf_mat = confusion_matrix(y_test, y_best_pred).tolist()
    
    metrics = {
        "best_model": best_model_name,
        "test_accuracy": round(best_score * 100, 2),
        "classes": class_names,
        "feature_cols": avail_features,
        "feature_importances": feature_importance_dict,
        "classification_report": cls_report,
        "confusion_matrix": conf_mat,
        "comparison": results
    }
    
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
        
    print(f"\nAll artifacts saved successfully in {models_dir}.")

if __name__ == "__main__":
    main()
