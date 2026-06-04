import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    ConfusionMatrixDisplay)

# 1. Load Dataset
print("Breast Cancer Classification")
print("\nLoading dataset from UCI repository...")

cancer = fetch_ucirepo(id=17)
X = cancer.data.features
y = cancer.data.targets

print(f"\nDataset info:")
print(f"  Samples  : {X.shape[0]}")
print(f"  Features : {X.shape[1]}")
print(f"\nTarget distribution:\n{y.value_counts().to_string()}")

# 2. Encode Target
le = LabelEncoder()
y  = le.fit_transform(y.values.ravel())
print(f"\nEncoded: B=0 (Benign={( y==0).sum()})  M=1 (Malignant={(y==1).sum()})")

# 3. Feature Engineering
X = X.copy()

mean_cols  = [c for c in X.columns if "mean"  in c]
worst_cols = [c for c in X.columns if "worst" in c]

for mc, wc in zip(sorted(mean_cols), sorted(worst_cols)):
    base = mc.replace("_mean", "")
    X[f"{base}_irregularity"] = X[wc] / (X[mc] + 1e-6)

print(f"\nAfter feature engineering: {X.shape[1]} features")

# 4. Train / Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 5. Define Models
def make_pipeline(model):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", model)])

models = {
    "Logistic Regression": make_pipeline(LogisticRegression(max_iter=1000, C=1.0, random_state=42)),
    "KNN":                 make_pipeline(KNeighborsClassifier(n_neighbors=5)),
    "SVM":                 make_pipeline(SVC(kernel="rbf", C=10, probability=True, random_state=42)),
    "Random Forest":       make_pipeline(RandomForestClassifier(n_estimators=200, random_state=42)),
    "XGBoost":             make_pipeline(XGBClassifier(n_estimators=100, use_label_encoder=False,
                                                        eval_metric="logloss", random_state=42,
                                                        verbosity=0))}

# 6. Train and Evaluate
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = {}

print("\n" + "=" * 55)
print(" MODEL COMPARISON")
print("=" * 55)

for name, pipeline in models.items():
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="accuracy")
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_proba)

    results[name] = {
        "cv_mean":  cv_scores.mean(),
        "cv_std":   cv_scores.std(),
        "test_acc": pipeline.score(X_test, y_test),
        "auc":      auc,
        "pipeline": pipeline,
        "y_pred":   y_pred,
        "y_proba":  y_proba}

    print(f"\n{name}")
    print(f" CV Accuracy : {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    print(f" Test Acc : {pipeline.score(X_test, y_test):.3f}")
    print(f" AUC-ROC : {auc:.3f}")
    print(classification_report(y_test, y_pred,target_names=["Benign", "Malignant"],zero_division=0))

best_name = max(results, key=lambda k: results[k]["auc"])
print(f"\n★  Best model by AUC: {best_name} ({results[best_name]['auc']:.3f})")

# 7. Plots
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Project — Breast Cancer Classification", fontsize=14, fontweight="bold")

# Plot 1: AUC bar chart
names = list(results.keys())
auc_vals = [results[n]["auc"] for n in names]
colors = ["#e74c3c" if n == best_name else "#f1948a" for n in names]
axes[0].barh(names, auc_vals, color=colors, edgecolor="white", height=0.6)
axes[0].set_xlabel("AUC-ROC Score")
axes[0].set_title("AUC-ROC Comparison")
axes[0].set_xlim(0.8, 1.01)
for i, v in enumerate(auc_vals):
    axes[0].text(v + 0.001, i, f"{v:.3f}", va="center", fontsize=9)

# Plot 2: Confusion matrix of best model
cm = confusion_matrix(y_test, results[best_name]["y_pred"])
ConfusionMatrixDisplay(cm, display_labels=["Benign", "Malignant"]).plot(
    ax=axes[1], colorbar=False, cmap="Reds")
axes[1].set_title(f"Confusion Matrix — {best_name}")

# Plot 3: ROC curves
for name, res in results.items():
    fpr, tpr, _ = roc_curve(y_test, res["y_proba"])
    axes[2].plot(fpr, tpr, label=f"{name} ({res['auc']:.2f})", linewidth=1.5)
axes[2].plot([0, 1], [0, 1], "k--", linewidth=0.8)
axes[2].set_xlabel("False Positive Rate")
axes[2].set_ylabel("True Positive Rate")
axes[2].set_title("ROC Curves")
axes[2].legend(fontsize=8)
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("project_results.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nPlot saved as project_results.png")

# 8. Correlation Heatmap (top 10 features)
rf_model    = models["Random Forest"].named_steps["model"]
feat_names  = X.columns.tolist()
importances = pd.Series(rf_model.feature_importances_, index=feat_names)
top10       = importances.nlargest(10).index.tolist()

fig, ax = plt.subplots(figsize=(10, 8))
corr = X[top10].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
            linewidths=0.5, ax=ax, square=True)
ax.set_title("Top 10 Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("project_correlation.png", dpi=150, bbox_inches="tight")
plt.show()
print("Correlation heatmap saved as project_correlation.png")

# 9. SHAP Explanation (XGBoost)
try:
    import shap

    print("\nGenerating SHAP explanation...")
    xgb_pipeline = models["XGBoost"]

    preprocessor = Pipeline([
        ("imputer", xgb_pipeline.named_steps["imputer"]),
        ("scaler",  xgb_pipeline.named_steps["scaler"])])
    X_test_transformed = preprocessor.transform(X_test)
    X_test_df = pd.DataFrame(X_test_transformed, columns=X.columns)

    xgb_model = xgb_pipeline.named_steps["model"]
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_test_df)

    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test_df, plot_type="bar", show=False)
    plt.title("SHAP Feature Importance — XGBoost")
    plt.tight_layout()
    plt.savefig("project_shap.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("SHAP plot saved as project_shap.png")

except ImportError:
    print("\nSHAP not installed.")