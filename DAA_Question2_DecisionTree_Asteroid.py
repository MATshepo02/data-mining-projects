# =============================================================================
# ITDAA4-12 Project | Question 2 (35 Marks)
# Topic: Predicting Asteroid Diameter — Decision Tree Regression
# Dataset: asteroid.csv
# Author: Matshepo Tshabangu
# =============================================================================

# pip install pandas numpy matplotlib seaborn scikit-learn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeRegressor, export_text, plot_tree
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import mutual_info_regression, SelectKBest
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                              r2_score, mean_absolute_percentage_error)
import warnings
warnings.filterwarnings("ignore")

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "#f9f9f9",
    "axes.grid":        True,
    "grid.alpha":       0.35,
    "font.size":        11,
})

TARGET = "diameter"   # target column — update if named differently in your CSV


# =============================================================================
# LOAD DATASET
# =============================================================================

print("Loading asteroid.csv ...")
df = pd.read_csv("asteroid.csv", low_memory=False)
print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
print("\nColumns:\n", df.columns.tolist())
print("\nFirst 3 rows:")
print(df.head(3))
print("\nData types:\n", df.dtypes)


# =============================================================================
# DATA CLEANING  (5 Marks)
# =============================================================================

print("\n" + "=" * 60)
print("DATA CLEANING")
print("=" * 60)

print(f"\nInitial shape: {df.shape}")

# --- Missing values ---
print("\nMissing values per column (top 20):")
missing = df.isnull().sum().sort_values(ascending=False)
print(missing[missing > 0].head(20))

# Drop columns where >60% of values are missing
threshold = 0.60
cols_to_drop = [col for col in df.columns
                if df[col].isnull().mean() > threshold]
print(f"\nDropping {len(cols_to_drop)} columns with >60% missing: {cols_to_drop}")
df.drop(columns=cols_to_drop, inplace=True)

# Drop rows missing the target
df.dropna(subset=[TARGET], inplace=True)
print(f"After dropping missing target rows: {df.shape}")

# Fill remaining numeric missing values with median
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
for col in numeric_cols:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].median(), inplace=True)

# Fill remaining categorical missing with mode
cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
for col in cat_cols:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].mode()[0], inplace=True)

print(f"Missing values remaining: {df.isnull().sum().sum()}")

# --- Duplicate rows ---
dupes = df.duplicated().sum()
print(f"Duplicate rows: {dupes}")
df.drop_duplicates(inplace=True)
print(f"Shape after deduplication: {df.shape}")

# --- Target variable distribution ---
print(f"\nTarget '{TARGET}' statistics:")
print(df[TARGET].describe().round(4))

# Remove extreme outliers in target (IQR method)
Q1 = df[TARGET].quantile(0.01)
Q3 = df[TARGET].quantile(0.99)
df = df[(df[TARGET] >= Q1) & (df[TARGET] <= Q3)]
print(f"After outlier removal (1st-99th percentile): {df.shape}")


# =============================================================================
# CATEGORICAL ENCODING  (3 Marks)
# =============================================================================

print("\n" + "=" * 60)
print("CATEGORICAL ENCODING")
print("=" * 60)

cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
if TARGET in cat_cols:
    cat_cols.remove(TARGET)

print(f"Categorical columns to encode: {cat_cols}")

le = LabelEncoder()
for col in cat_cols:
    n_unique = df[col].nunique()
    if n_unique <= 30:
        # Label encode low-cardinality columns
        df[col] = le.fit_transform(df[col].astype(str))
        print(f"  Label-encoded '{col}' ({n_unique} unique values)")
    else:
        # Drop very high cardinality string columns (e.g., names, IDs)
        df.drop(columns=[col], inplace=True)
        print(f"  Dropped high-cardinality '{col}' ({n_unique} unique values)")

print(f"\nShape after encoding: {df.shape}")


# =============================================================================
# FEATURE SELECTION  (5 Marks)
# =============================================================================

print("\n" + "=" * 60)
print("FEATURE SELECTION")
print("=" * 60)

# Separate features and target
X = df.drop(columns=[TARGET])
y = df[TARGET]

# Remove near-zero variance columns
from sklearn.feature_selection import VarianceThreshold
selector_var = VarianceThreshold(threshold=0.01)
selector_var.fit(X)
low_var_cols = X.columns[~selector_var.get_support()].tolist()
if low_var_cols:
    print(f"Dropping {len(low_var_cols)} near-zero variance columns: {low_var_cols}")
    X.drop(columns=low_var_cols, inplace=True)

# Mutual Information — rank features by relevance to target
print("\nComputing Mutual Information scores ...")
mi_scores = mutual_info_regression(X, y, random_state=42)
mi_df = pd.DataFrame({"Feature": X.columns, "MI_Score": mi_scores})
mi_df.sort_values("MI_Score", ascending=False, inplace=True)
print(f"\nTop 20 features by Mutual Information:\n{mi_df.head(20).to_string(index=False)}")

# Select top K features
K_FEATURES = min(15, len(mi_df))
top_features = mi_df.head(K_FEATURES)["Feature"].tolist()
print(f"\nSelected top {K_FEATURES} features: {top_features}")

X_selected = X[top_features]

# --- Feature Importance Plot ---
fig, ax = plt.subplots(figsize=(10, 6))
colours = plt.cm.viridis(np.linspace(0.2, 0.9, K_FEATURES))
bars = ax.barh(
    mi_df["Feature"].head(K_FEATURES)[::-1],
    mi_df["MI_Score"].head(K_FEATURES)[::-1],
    colour=colours[::-1], alpha=0.85
)
ax.set_title(f"Top {K_FEATURES} Features — Mutual Information Score",
             fontweight="bold")
ax.set_xlabel("Mutual Information Score")
ax.set_ylabel("Feature")
plt.tight_layout()
plt.savefig("plot_10_feature_importance_MI.png", dpi=150)
plt.show()
print("Saved: plot_10_feature_importance_MI.png")

# Correlation heatmap of selected features
fig, ax = plt.subplots(figsize=(12, 10))
corr_matrix = X_selected.corr()
sns.heatmap(corr_matrix, annot=False, cmap="coolwarm",
            vmin=-1, vmax=1, ax=ax, linewidths=0.3)
ax.set_title("Feature Correlation Heatmap (Selected Features)", fontweight="bold")
plt.tight_layout()
plt.savefig("plot_11_feature_correlation.png", dpi=150)
plt.show()
print("Saved: plot_11_feature_correlation.png")


# =============================================================================
# FEATURE SCALING  (4 Marks)
# =============================================================================

print("\n" + "=" * 60)
print("FEATURE SCALING")
print("=" * 60)

scaler = StandardScaler()
X_scaled = pd.DataFrame(
    scaler.fit_transform(X_selected),
    columns=X_selected.columns,
    index=X_selected.index
)
print("StandardScaler applied (zero mean, unit variance) ✓")
print(f"\nScaled feature stats (mean ≈ 0, std ≈ 1):")
print(X_scaled.describe().round(3).loc[["mean", "std"]])


# =============================================================================
# TRAIN / TEST SPLIT  (2 Marks)
# =============================================================================

print("\n" + "=" * 60)
print("TRAIN / TEST SPLIT (80/20)")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.20, random_state=42
)
print(f"Training samples : {len(X_train)}")
print(f"Test samples     : {len(X_test)}")
print(f"Train target mean: {y_train.mean():.4f}")
print(f"Test  target mean: {y_test.mean():.4f}")


# =============================================================================
# FIT DECISION TREE MODEL  (4 Marks)
# =============================================================================

print("\n" + "=" * 60)
print("DECISION TREE MODEL FITTING")
print("=" * 60)

# Hyperparameter tuning via cross-validation over max_depth
print("Tuning max_depth via 5-fold cross-validation ...")
depth_scores = {}
for depth in range(2, 16):
    dt = DecisionTreeRegressor(max_depth=depth, random_state=42)
    cv_scores = cross_val_score(dt, X_train, y_train, cv=5, scoring="r2")
    depth_scores[depth] = cv_scores.mean()
    print(f"  max_depth={depth:2d}  CV R² = {cv_scores.mean():.4f}")

best_depth = max(depth_scores, key=depth_scores.get)
print(f"\n★ Best max_depth: {best_depth}  (CV R² = {depth_scores[best_depth]:.4f})")

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(list(depth_scores.keys()), list(depth_scores.values()),
        marker="o", colour="#e74c3c", linewidth=2)
ax.axvline(x=best_depth, colour="green", linestyle="--",
           label=f"Best depth = {best_depth}")
ax.set_title("Decision Tree Max Depth vs CV R² Score", fontweight="bold")
ax.set_xlabel("max_depth")
ax.set_ylabel("Mean CV R²")
ax.legend()
plt.tight_layout()
plt.savefig("plot_12_depth_tuning.png", dpi=150)
plt.show()
print("Saved: plot_12_depth_tuning.png")

# Fit final model
dt_model = DecisionTreeRegressor(
    max_depth=best_depth,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42
)
dt_model.fit(X_train, y_train)
print(f"\nDecision Tree fitted ✓")
print(f"Nodes in tree : {dt_model.tree_.node_count}")
print(f"Tree depth    : {dt_model.get_depth()}")
print(f"Leaves        : {dt_model.get_n_leaves()}")


# =============================================================================
# EVALUATE MODEL PERFORMANCE  (3 Marks)
# =============================================================================

print("\n" + "=" * 60)
print("MODEL EVALUATION")
print("=" * 60)

y_train_pred = dt_model.predict(X_train)
y_test_pred  = dt_model.predict(X_test)

def evaluate(actual, predicted, label):
    mae  = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    r2   = r2_score(actual, predicted)
    mape = mean_absolute_percentage_error(actual, predicted) * 100
    print(f"\n  {label}:")
    print(f"    MAE   : {mae:.4f}")
    print(f"    RMSE  : {rmse:.4f}")
    print(f"    R²    : {r2:.4f}")
    print(f"    MAPE  : {mape:.2f}%")
    return {"MAE": mae, "RMSE": rmse, "R2": r2, "MAPE": mape}

train_metrics = evaluate(y_train, y_train_pred, "Training Set")
test_metrics  = evaluate(y_test,  y_test_pred,  "Test Set")

r2_gap = train_metrics["R2"] - test_metrics["R2"]
print(f"\n  R² gap (train - test): {r2_gap:.4f}")
if r2_gap > 0.15:
    print("  ⚠ Possible overfitting — consider pruning or min_samples tuning.")
else:
    print("  ✓ Train/test R² are close — model generalises well.")

# Actual vs Predicted scatter
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, actual, pred, label, colour in [
    (axes[0], y_train, y_train_pred, "Training Set", "#2980b9"),
    (axes[1], y_test,  y_test_pred,  "Test Set",     "#e74c3c"),
]:
    ax.scatter(actual, pred, alpha=0.4, s=10, colour=colour)
    lims = [min(actual.min(), pred.min()), max(actual.max(), pred.max())]
    ax.plot(lims, lims, "k--", linewidth=1.2)
    r2 = r2_score(actual, pred)
    ax.set_title(f"Actual vs Predicted — {label}\nR² = {r2:.4f}", fontweight="bold")
    ax.set_xlabel("Actual Diameter (km)")
    ax.set_ylabel("Predicted Diameter (km)")

plt.tight_layout()
plt.savefig("plot_13_actual_vs_predicted.png", dpi=150)
plt.show()
print("Saved: plot_13_actual_vs_predicted.png")

# Residuals
residuals = y_test - y_test_pred
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
axes[0].scatter(y_test_pred, residuals, alpha=0.4, s=10, colour="#8e44ad")
axes[0].axhline(0, colour="red", linewidth=1, linestyle="--")
axes[0].set_title("Residuals vs Predicted", fontweight="bold")
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("Residual")
axes[1].hist(residuals, bins=40, colour="#8e44ad", alpha=0.8)
axes[1].set_title("Residual Distribution", fontweight="bold")
axes[1].set_xlabel("Residual")
plt.tight_layout()
plt.savefig("plot_14_residuals.png", dpi=150)
plt.show()
print("Saved: plot_14_residuals.png")

# Decision Tree feature importances
feat_imp = pd.DataFrame({
    "Feature":   X_selected.columns,
    "Importance": dt_model.feature_importances_
}).sort_values("Importance", ascending=False)

fig, ax = plt.subplots(figsize=(10, 5))
ax.barh(feat_imp["Feature"][::-1], feat_imp["Importance"][::-1],
        colour="#27ae60", alpha=0.85)
ax.set_title("Decision Tree Feature Importances", fontweight="bold")
ax.set_xlabel("Importance Score")
plt.tight_layout()
plt.savefig("plot_15_dt_feature_importance.png", dpi=150)
plt.show()
print("Saved: plot_15_dt_feature_importance.png")


# =============================================================================
# PLOT DECISION TREE  (4 Marks)
# =============================================================================

print("\n" + "=" * 60)
print("DECISION TREE VISUALISATION")
print("=" * 60)

# Text representation (first 5 levels)
tree_rules = export_text(dt_model,
                         feature_names=list(X_selected.columns),
                         max_depth=4)
print("\nTree structure (first 4 levels):\n")
print(tree_rules[:2000])

# Graphical tree (capped at depth 4 for readability)
PLOT_DEPTH = min(4, best_depth)
fig, ax = plt.subplots(figsize=(24, 10))
plot_tree(
    dt_model,
    feature_names=list(X_selected.columns),
    filled=True,
    rounded=True,
    fontsize=7,
    max_depth=PLOT_DEPTH,
    ax=ax,
    impurity=True
)
ax.set_title(
    f"Decision Tree — Asteroid Diameter Prediction\n"
    f"(Showing first {PLOT_DEPTH} levels of {best_depth} total | "
    f"R²={test_metrics['R2']:.4f} | RMSE={test_metrics['RMSE']:.4f})",
    fontsize=12, fontweight="bold"
)
plt.tight_layout()
plt.savefig("plot_16_decision_tree.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: plot_16_decision_tree.png")

# Summary table
print("\n" + "=" * 60)
print("FINAL MODEL PERFORMANCE SUMMARY")
print("=" * 60)
summary = pd.DataFrame([train_metrics, test_metrics],
                        index=["Training", "Test"])
summary = summary.round(4)
print(summary.to_string())
print(f"\nModel: DecisionTreeRegressor(max_depth={best_depth})")
print(f"Features used: {list(X_selected.columns)}")
print("\nQ2 Complete ✓")
