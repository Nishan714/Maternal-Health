# Install Required Libraries
!pip install imbalanced-learn xgboost scikit-learn==1.5.0 seaborn matplotlib shap --quiet

# Import Packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score, 
                            roc_curve, auc, accuracy_score, f1_score, precision_score, 
                            recall_score)
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier, 
                             AdaBoostClassifier)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.preprocessing import label_binarize
import warnings
import os

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100

plot_dir = "/content/plots"
os.makedirs(plot_dir, exist_ok=True)

# Load Dataset
df = pd.read_csv("/content/Link1.csv")
print(f"Dataset shape: {df.shape}\n")
print("Columns:", list(df.columns))

# Identify Target Column
possible_targets = ['RiskLevel', 'Risk Level', 'risk_level', 'Risk', 'risk',
                   'target', 'Target', 'label', 'Label', 'class', 'Class']
target_col = next((col for col in possible_targets if col in df.columns), df.columns[-1])
print(f"\nTarget column: '{target_col}'")

# Encode target if needed
if df[target_col].dtype == 'object':
    le = LabelEncoder()
    df[target_col] = le.fit_transform(df[target_col])
    print(f"Target encoded: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# EDA
print(f"\n{'='*60}\nEXPLORATORY DATA ANALYSIS\n{'='*60}")
display(df.head(10))
print(f"\nShape: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"\nMissing values:\n{df.isnull().sum()[df.isnull().sum() > 0] if df.isnull().sum().sum() > 0 else 'None'}")
display(df.describe())
print(f"\n{target_col} distribution:\n{df[target_col].value_counts()}")

# Visualizations - Target Distribution
plt.figure(figsize=(10, 6))
target_counts = df[target_col].value_counts().sort_index()
plt.bar(target_counts.index, target_counts.values, color=sns.color_palette("viridis", len(target_counts)))
plt.xlabel('Risk Level')
plt.ylabel('Count')
plt.title('Target Distribution')
for i, v in enumerate(target_counts.values):
    plt.text(target_counts.index[i], v + 10, str(v), ha='center', fontweight='bold')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f"{plot_dir}/01_target_distribution.png", dpi=300, bbox_inches='tight')
plt.show()

# Feature Distributions
numeric_features = [col for col in df.select_dtypes(include=['int64', 'float64']).columns 
                   if col != target_col]

if numeric_features:
    n_cols = 3
    n_rows = (len(numeric_features) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 4))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
    
    for idx, col in enumerate(numeric_features):
        sns.histplot(data=df, x=col, hue=target_col, kde=True, ax=axes[idx], palette='viridis')
        axes[idx].set_title(f'{col}')
    
    for idx in range(len(numeric_features), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/02_feature_distributions.png", dpi=300, bbox_inches='tight')
    plt.show()

# Correlation Heatmap
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
if len(numeric_cols) > 1:
    plt.figure(figsize=(12, 10))
    sns.heatmap(df[numeric_cols].corr(), annot=True, fmt='.2f', cmap='viridis',
                center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title('Correlation Heatmap', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/03_correlation_heatmap.png", dpi=300, bbox_inches='tight')
    plt.show()

# Box Plots
if numeric_features:
    n_cols = 3
    n_rows = (len(numeric_features) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 4))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
    
    for idx, col in enumerate(numeric_features):
        sns.boxplot(data=df, x=target_col, y=col, ax=axes[idx], palette='Set2')
        axes[idx].set_title(f'{col} by Risk Level')
    
    for idx in range(len(numeric_features), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/04_boxplots.png", dpi=300, bbox_inches='tight')
    plt.show()

# Prepare Data
y = df[target_col].values
X = df.drop(columns=[target_col])
categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

n_classes = len(np.unique(y))
print(f"\nFeatures: {len(numeric_cols)} numeric, {len(categorical_cols)} categorical")
print(f"Classes: {n_classes}, Distribution: {np.bincount(y)}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, 
                                                    random_state=42, stratify=y)
print(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

# Preprocessing Pipeline
try:
    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
except TypeError:
    ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)

numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', ohe)
])

transformers = [('num', numeric_transformer, numeric_cols)]
if categorical_cols:
    transformers.append(('cat', categorical_transformer, categorical_cols))

preprocessor = ColumnTransformer(transformers=transformers)

# Define Models
models = {
    "Logistic Regression": LogisticRegression(max_iter=2000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "SVM": SVC(probability=True, kernel='rbf', random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=7),
    "Naive Bayes": GaussianNB(),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "AdaBoost": AdaBoostClassifier(algorithm='SAMME', random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss',
                            learning_rate=0.05, n_estimators=200, max_depth=4, 
                            random_state=42),
    "ANN": MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu',
                        solver='adam', max_iter=500, random_state=42)
}

# K-Fold Cross-Validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring_metric = 'roc_auc' if n_classes == 2 else 'f1_weighted'
metric_name = "ROC-AUC" if n_classes == 2 else "F1-Score (Weighted)"

results = {}
all_metrics = []

print(f"\n{'='*60}\nMODEL PERFORMANCE (5-Fold CV)\n{'='*60}\n")

for name, model in models.items():
    try:
        pipe = ImbPipeline([
            ('preprocessor', preprocessor),
            ('smote', SMOTE(random_state=42)),
            ('model', model)
        ])
        
        scores = cross_val_score(pipe, X, y, cv=cv, scoring=scoring_metric)
        results[name] = scores
        
        all_metrics.append({
            'Model': name,
            'Mean Score': scores.mean(),
            'Std Dev': scores.std()
        })
        
        print(f"{name:25s} {metric_name} = {scores.mean():.4f} (± {scores.std():.4f})")
    except Exception as e:
        print(f"{name:25s} Failed: {str(e)[:50]}")

# Model Comparison Table
if all_metrics:
    metrics_df = pd.DataFrame(all_metrics).sort_values('Mean Score', ascending=False)
    print(f"\n{'='*60}\nMODEL COMPARISON\n{'='*60}")
    display(metrics_df)
    metrics_df.to_csv("/content/model_comparison_table.csv", index=False)

# Model Comparison Visualizations
if results:
    results_df = pd.DataFrame(results)
    mean_scores = results_df.mean().sort_values(ascending=False)
    
    # Bar Chart
    plt.figure(figsize=(12, 8))
    colors = plt.cm.viridis(np.linspace(0, 1, len(mean_scores)))
    plt.barh(range(len(mean_scores)), mean_scores.values, color=colors)
    plt.yticks(range(len(mean_scores)), mean_scores.index)
    plt.xlabel(f"Mean {metric_name}", fontsize=12)
    plt.title(f"Model Comparison - {metric_name}", fontsize=14, fontweight='bold')
    for i, v in enumerate(mean_scores.values):
        plt.text(v + 0.01, i, f"{v:.3f}", va='center', fontsize=10, fontweight='bold')
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/05_model_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # CV Heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(results_df[mean_scores.index].T, annot=True, fmt='.3f', cmap='RdYlGn',
                cbar_kws={'label': metric_name}, linewidths=0.5)
    plt.title(f'Cross-Validation Results - {metric_name}', fontsize=14, fontweight='bold')
    plt.xlabel('Fold')
    plt.ylabel('Model')
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/06_cv_heatmap.png", dpi=300, bbox_inches='tight')
    plt.show()

# Train Best Model
if results:
    best_model_name = mean_scores.idxmax()
    best_model = models[best_model_name]
    
    print(f"\n{'='*60}\nBEST MODEL: {best_model_name}\n{'='*60}")
    
    final_pipe = ImbPipeline([
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('model', best_model)
    ])
    
    final_pipe.fit(X_train, y_train)
    y_pred = final_pipe.predict(X_test)
    y_prob = final_pipe.predict_proba(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"\nTest Performance:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}\n")
    print(classification_report(y_test, y_pred))
    
    # Confusion Matrix
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0], linewidths=1)
    axes[0].set_title(f"{best_model_name} - Confusion Matrix", fontsize=14, fontweight='bold')
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")
    
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_norm * 100, annot=True, fmt='.2f', cmap='Greens', ax=axes[1], linewidths=1)
    axes[1].set_title(f"{best_model_name} - Normalized (%)", fontsize=14, fontweight='bold')
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")
    
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/07_confusion_matrix.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # ROC Curves
    if n_classes == 2:
        fpr, tpr, _ = roc_curve(y_test, y_prob[:, 1])
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, label=f"{best_model_name} (AUC = {roc_auc:.3f})",
                linewidth=3, color='#2E86AB')
        plt.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=2)
        plt.fill_between(fpr, tpr, alpha=0.2, color='#2E86AB')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve - {best_model_name}", fontsize=14, fontweight='bold')
        plt.legend(loc='lower right')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{plot_dir}/08_roc_curve.png", dpi=300, bbox_inches='tight')
        plt.show()
    else:
        y_test_bin = label_binarize(y_test, classes=np.unique(y))
        
        plt.figure(figsize=(10, 8))
        colors = plt.cm.Set1(np.linspace(0, 1, n_classes))
        
        for i, color in enumerate(colors):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, color=color, linewidth=2,
                    label=f'Class {i} (AUC = {roc_auc:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curves - {best_model_name}", fontsize=14, fontweight='bold')
        plt.legend(loc='lower right')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{plot_dir}/08_roc_curve.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    # Feature Importance
    feature_names = numeric_cols.copy()
    if categorical_cols:
        try:
            encoder = final_pipe.named_steps['preprocessor'].named_transformers_['cat'].named_steps['encoder']
            feature_names.extend(encoder.get_feature_names_out(categorical_cols))
        except:
            feature_names.extend(categorical_cols)
    
    X_train_transformed = final_pipe.named_steps['preprocessor'].transform(X_train)
    X_test_transformed = final_pipe.named_steps['preprocessor'].transform(X_test)
    
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_transformed, y_train)
    
    # Tree-based Feature Importance
    if best_model_name in ['Decision Tree', 'Random Forest', 'Gradient Boosting', 'AdaBoost', 'XGBoost']:
        model = final_pipe.named_steps['model']
        
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            top_n = min(20, len(feature_names))
            top_indices = indices[:top_n]
            
            importance_df = pd.DataFrame({
                'Feature': [feature_names[i] for i in top_indices],
                'Importance': importances[top_indices]
            })
            
            print(f"\nTop {top_n} Important Features:")
            display(importance_df)
            importance_df.to_csv("/content/feature_importance.csv", index=False)
            
            plt.figure(figsize=(10, 8))
            colors = plt.cm.viridis(np.linspace(0, 1, len(top_indices)))
            plt.barh(range(len(top_indices)), importances[top_indices], color=colors)
            plt.yticks(range(len(top_indices)), [feature_names[i] for i in top_indices])
            plt.xlabel('Importance')
            plt.title(f'Top {top_n} Features - {best_model_name}', fontsize=14, fontweight='bold')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig(f"{plot_dir}/10_feature_importance.png", dpi=300, bbox_inches='tight')
            plt.show()
    
    # Permutation Importance
    print(f"\n{'='*60}\nPERMUTATION IMPORTANCE\n{'='*60}")
    
    trained_model = final_pipe.named_steps['model']
    perm_importance = permutation_importance(trained_model, X_test_transformed, y_test,
                                            n_repeats=10, random_state=42, n_jobs=-1)
    
    perm_indices = np.argsort(perm_importance.importances_mean)[::-1]
    top_n_perm = min(20, len(feature_names))
    top_perm_indices = perm_indices[:top_n_perm]
    
    perm_df = pd.DataFrame({
        'Feature': [feature_names[i] for i in top_perm_indices],
        'Importance': perm_importance.importances_mean[top_perm_indices]
    })
    
    display(perm_df)
    perm_df.to_csv("/content/permutation_importance.csv", index=False)
    
    plt.figure(figsize=(10, 8))
    colors = plt.cm.plasma(np.linspace(0, 1, len(top_perm_indices)))
    plt.barh(range(len(top_perm_indices)), perm_importance.importances_mean[top_perm_indices],
            xerr=perm_importance.importances_std[top_perm_indices],
            color=colors, alpha=0.8, ecolor='black', capsize=5)
    plt.yticks(range(len(top_perm_indices)), [feature_names[i] for i in top_perm_indices])
    plt.xlabel('Permutation Importance')
    plt.title(f'Top {top_n_perm} Features - Permutation Importance', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/12_permutation_importance.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # SHAP Analysis
    print(f"\n{'='*60}\nSHAP ANALYSIS\n{'='*60}")
    
    try:
        max_samples = min(100, len(X_test_transformed))
        X_shap = X_test_transformed[:max_samples]
        
        if best_model_name in ['Random Forest', 'XGBoost', 'Gradient Boosting']:
            explainer = shap.TreeExplainer(trained_model)
            shap_values = explainer.shap_values(X_shap)
        else:
            explainer = shap.KernelExplainer(trained_model.predict_proba,
                                            shap.sample(X_train_resampled, 100))
            shap_values = explainer.shap_values(X_shap)
        
        shap_values_combined = shap_values[0] if isinstance(shap_values, list) else shap_values
        
        # SHAP Summary
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values_combined, X_shap, feature_names=feature_names, show=False)
        plt.title('SHAP Summary', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"{plot_dir}/13_shap_summary.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        # SHAP Bar
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values_combined, X_shap, feature_names=feature_names,
                         plot_type="bar", show=False)
        plt.title('SHAP Importance', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"{plot_dir}/14_shap_bar.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        shap_importance = np.abs(shap_values_combined).mean(axis=0)
        shap_indices = np.argsort(shap_importance)[::-1][:20]
        
        shap_df = pd.DataFrame({
            'Feature': [feature_names[i] for i in shap_indices],
            'SHAP Importance': shap_importance[shap_indices]
        })
        
        display(shap_df)
        shap_df.to_csv("/content/shap_importance.csv", index=False)
        
    except Exception as e:
        print(f"SHAP failed: {str(e)}")

print(f"\n{'='*60}\nANALYSIS COMPLETE\n{'='*60}")
print(f"\nPlots saved to: {plot_dir}/")
print("CSV files saved to: /content/")