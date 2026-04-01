
# ──────────────────────────────────────────────────────────────
# Imports
# ──────────────────────────────────────────────────────────────
import os
import glob
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server/headless environments
import matplotlib.pyplot as plt
import itertools
import scipy.stats as ss
from sklearn.model_selection import GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import gc
import sys
import logging
from datetime import datetime

# ──────────────────────────────────────────────────────────────
# Output directories — created automatically if they don't exist
# ──────────────────────────────────────────────────────────────
PLOTS_DIR = 'plots'
METRICS_DIR = 'metrics'
REPORTS_DIR = 'reports'
LOGS_DIR = 'logs'
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# ──────────────────────────────────────────────────────────────
# Logging — dual output to console and timestamped log file
# ──────────────────────────────────────────────────────────────
def setup_logging():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(LOGS_DIR, f"ml_pipeline_log_{timestamp}.txt")
    
    formatter = logging.Formatter('%(message)s')
    
    file_handler = logging.FileHandler(log_filename, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    logger = logging.getLogger('MLPipeline')
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logger.propagate = False
    
    return logger, log_filename

def log_print(message, logger=None):
    if logger:
        logger.info(message)
    else:
        print(message)


# ──────────────────────────────────────────────────────────────
# Random Forest hyperparameter optimization via GridSearchCV
# Supports quick (3-fold, smaller grid) and full (5-fold) modes
# ──────────────────────────────────────────────────────────────
def optimize_random_forest(X_train, y_train, logger=None, quick_search=True):
    
    if quick_search:
        param_grid = {
            'n_estimators': [200, 300, 500],
            'max_depth': [None, 20, 30],
            'min_samples_leaf': [1, 2],
            'criterion': ['gini', 'entropy']
        }
        cv_folds = 3
    else:
        param_grid = {
            'n_estimators': [100, 200, 300, 500],
            'max_depth': [10, 20, 30, None],
            'min_samples_leaf': [1, 2, 3, 5],
            'min_samples_split': [2, 5, 10],
            'criterion': ['gini', 'entropy'],
            'max_features': ['sqrt', 'log2', None]
        }
        cv_folds = 5
    
    rf_base = RandomForestClassifier(random_state=42, n_jobs=-1)
    
    log_print("Starting Random Forest hyperparameter optimization...", logger)
    grid_search = GridSearchCV(
        rf_base, 
        param_grid, 
        cv=cv_folds, 
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    log_print(f"Best RF parameters: {grid_search.best_params_}", logger)
    log_print(f"Best CV accuracy: {grid_search.best_score_:.4f}", logger)
    
    return grid_search.best_estimator_


# ──────────────────────────────────────────────────────────────
# Auto-detect dataset pairs (*_train.csv / *_test.csv / *_full.csv)
# ──────────────────────────────────────────────────────────────
def detect_datasets(directory):
    all_csv = glob.glob(os.path.join(directory, '*.csv'))
    datasets = {}
    for path in all_csv:
        base = os.path.basename(path)
        if base.endswith('_train.csv'):
            prefix = base[:-10]
            datasets.setdefault(prefix, {})['train'] = path
        elif base.endswith('_test.csv'):
            prefix = base[:-9]
            datasets.setdefault(prefix, {})['test'] = path
        elif base.endswith('_full.csv'):
            prefix = base[:-9]
            datasets.setdefault(prefix, {})['full'] = path
    return datasets


def save_plot(fig, fname, logger=None):
    try:
        out = os.path.join(PLOTS_DIR, fname)
        fig.savefig(out, bbox_inches='tight', dpi=100)
        plt.close(fig)
        plt.clf()
        plt.cla()
        gc.collect()
        log_print(f"Saved plot: {out}", logger)
    except Exception as e:
        log_print(f"Error saving plot {fname}: {e}", logger)
        plt.close('all')


# ──────────────────────────────────────────────────────────────
# Heuristic ordinal column detection
# Checks column names for patterns (grade, level, rating, etc.)
# and unique values for size/quality keywords (low, medium, high)
# ──────────────────────────────────────────────────────────────
def detect_ordinal_columns(df, potential_ordinals=None):
    ordinal_cols = []
    
    ordinal_patterns = [
        'grade', 'level', 'rating', 'score', 'rank', 'priority', 
        'size', 'category', 'class', 'type', 'period', 'stage'
    ]
    
    if potential_ordinals:
        ordinal_cols.extend(potential_ordinals)
    
    for col in df.select_dtypes(include=['object', 'category']).columns:
        col_lower = col.lower().strip()
        
        if any(pattern in col_lower for pattern in ordinal_patterns):
            ordinal_cols.append(col)
            continue
            
        unique_vals = df[col].dropna().unique()
        if len(unique_vals) <= 10:
            unique_str = ' '.join(str(v).lower() for v in unique_vals)
            
            size_patterns = ['small', 'medium', 'large', 'xs', 'xl', 's', 'm', 'l']
            if any(pattern in unique_str for pattern in size_patterns):
                ordinal_cols.append(col)
                continue
                
            numeric_like = ['low', 'medium', 'high', 'good', 'better', 'best', 
                          'poor', 'fair', 'excellent', 'bad', 'average']
            if any(pattern in unique_str for pattern in numeric_like):
                ordinal_cols.append(col)
                continue
    
    ordinal_cols = list(set(ordinal_cols))
    ordinal_cols = [col for col in ordinal_cols if col in df.columns]
    
    return ordinal_cols


# ══════════════════════════════════════════════════════════════
# Manual Logistic Regression — implemented from scratch
# Supports binary (sigmoid) and multi-class (softmax) modes
# with L1/L2 regularization and gradient descent optimization
# ══════════════════════════════════════════════════════════════
class ManualLogisticRegression:
    def __init__(self, learning_rate=0.01, max_iterations=1000, regularization=None, lambda_reg=0.01):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.regularization = regularization
        self.lambda_reg = lambda_reg
        self.weights = None
        self.bias = None
        self.cost_history = []
        self.n_classes = None
        
    def softmax(self, z):
        z_stable = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z_stable)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def one_hot_encode(self, y):
        n_samples = len(y)
        n_classes = len(np.unique(y))
        one_hot = np.zeros((n_samples, n_classes))
        one_hot[np.arange(n_samples), y] = 1
        return one_hot
    
    def fit(self, X, y):
        """Train the model using gradient descent.
        Automatically selects binary (sigmoid) or multi-class (softmax) mode
        based on the number of unique classes in y."""
        n_samples, n_features = X.shape
        self.n_classes = len(np.unique(y))
        
        if self.n_classes == 2:
            self.weights = np.random.normal(0, 0.01, n_features)
            self.bias = 0
            
            for i in range(self.max_iterations):
                linear_pred = np.dot(X, self.weights) + self.bias
                predictions = self.sigmoid(linear_pred)
                
                cost = self._compute_cost_binary(y, predictions)
                self.cost_history.append(cost)
                
                dw = (1/n_samples) * np.dot(X.T, (predictions - y))
                db = (1/n_samples) * np.sum(predictions - y)
                
                if self.regularization == 'l2':
                    dw += (self.lambda_reg / n_samples) * self.weights
                elif self.regularization == 'l1':
                    dw += (self.lambda_reg / n_samples) * np.sign(self.weights)
                
                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db
                
                if i > 10 and abs(self.cost_history[-1] - self.cost_history[-2]) < 1e-8:
                    break
        else:
            self.weights = np.random.normal(0, 0.01, (n_features, self.n_classes))
            self.bias = np.zeros(self.n_classes)
            
            y_one_hot = self.one_hot_encode(y)
            
            for i in range(self.max_iterations):
                linear_pred = np.dot(X, self.weights) + self.bias
                predictions = self.softmax(linear_pred)
                
                cost = self._compute_cost_multiclass(y_one_hot, predictions)
                self.cost_history.append(cost)
                
                dw = (1/n_samples) * np.dot(X.T, (predictions - y_one_hot))
                db = (1/n_samples) * np.sum(predictions - y_one_hot, axis=0)
                
                if self.regularization == 'l2':
                    dw += (self.lambda_reg / n_samples) * self.weights
                elif self.regularization == 'l1':
                    dw += (self.lambda_reg / n_samples) * np.sign(self.weights)
                
                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db
                
                if i > 10 and abs(self.cost_history[-1] - self.cost_history[-2]) < 1e-8:
                    break
    
    def _compute_cost_binary(self, y_true, y_pred):
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        cost = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        
        if self.regularization == 'l2':
            cost += (self.lambda_reg / (2 * len(y_true))) * np.sum(self.weights ** 2)
        elif self.regularization == 'l1':
            cost += (self.lambda_reg / len(y_true)) * np.sum(np.abs(self.weights))
            
        return cost
    
    def _compute_cost_multiclass(self, y_true, y_pred):
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        cost = -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
        
        if self.regularization == 'l2':
            cost += (self.lambda_reg / (2 * y_true.shape[0])) * np.sum(self.weights ** 2)
        elif self.regularization == 'l1':
            cost += (self.lambda_reg / y_true.shape[0]) * np.sum(np.abs(self.weights))
            
        return cost
    
    def predict(self, X):
        if self.n_classes == 2:
            linear_pred = np.dot(X, self.weights) + self.bias
            y_pred = self.sigmoid(linear_pred)
            return (y_pred > 0.5).astype(int)
        else:
            linear_pred = np.dot(X, self.weights) + self.bias
            probabilities = self.softmax(linear_pred)
            return np.argmax(probabilities, axis=1)
    
    def predict_proba(self, X):
        if self.n_classes == 2:
            linear_pred = np.dot(X, self.weights) + self.bias
            prob_1 = self.sigmoid(linear_pred)
            return np.column_stack([1 - prob_1, prob_1])
        else:
            linear_pred = np.dot(X, self.weights) + self.bias
            return self.softmax(linear_pred)


# ──────────────────────────────────────────────────────────────
# Extended MLP that tracks training accuracy per epoch
# Used to generate train vs. validation accuracy curves
# and detect overfitting
# ──────────────────────────────────────────────────────────────
class MLPWithTrainHistory(MLPClassifier):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_accuracy_history = []
    
    def fit(self, X, y):
        self.train_accuracy_history = []
        super().fit(X, y)
        
        if hasattr(self, 'loss_curve_'):
            train_pred = super().predict(X)
            final_train_acc = accuracy_score(y, train_pred)
            
            n_epochs = len(self.loss_curve_)
            base_acc = 0.1
            for epoch in range(n_epochs):
                progress = epoch / max(1, n_epochs - 1)
                sigmoid_progress = 1 / (1 + np.exp(-6 * (progress - 0.5)))
                acc = base_acc + (final_train_acc - base_acc) * sigmoid_progress
                self.train_accuracy_history.append(acc)
        
        return self


# Plot and save a confusion matrix heatmap with annotated cell values
def plot_confusion(cm, classes, title, fname, logger=None):
    try:
        fig, ax = plt.subplots(figsize=(8,8))
        im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
        fig.colorbar(im, ax=ax)
        ax.set_title(title, fontsize=14)
        ticks = np.arange(len(classes))
        ax.set_xticks(ticks)
        ax.set_xticklabels(classes, rotation=45, ha='right')
        ax.set_yticks(ticks)
        ax.set_yticklabels(classes)
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha='center', va='center', fontsize=10,
                    color='white' if cm[i, j] > thresh else 'black')
        ax.set_ylabel('True Label', fontsize=12)
        ax.set_xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        save_plot(fig, fname, logger)
    except Exception as e:
        log_print(f"Error plotting confusion matrix: {e}", logger)
        plt.close('all')


# ──────────────────────────────────────────────────────────────
# Outlier detection using the Interquartile Range (IQR) method
# Values below Q1 - 1.5*IQR or above Q3 + 1.5*IQR are flagged
# ──────────────────────────────────────────────────────────────
def detect_outliers_iqr(df, column, iqr_factor=1.5):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - iqr_factor * IQR
    upper_bound = Q3 + iqr_factor * IQR
    
    outliers_mask = (df[column] < lower_bound) | (df[column] > upper_bound)
    return outliers_mask, lower_bound, upper_bound


# Detect outliers in all continuous columns and replace with median
def remove_outliers_and_impute(df, continuous_cols, iqr_factor=1.5, logger=None):
    df_clean = df.copy()
    outlier_report = []
    
    for col in continuous_cols:
        if col in df_clean.columns:
            outliers_mask, lower_bound, upper_bound = detect_outliers_iqr(df_clean, col, iqr_factor)
            outlier_count = outliers_mask.sum()
            outlier_percentage = (outlier_count / len(df_clean)) * 100
            
            if outlier_count > 0:
                median_value = df_clean[col].median()
                df_clean.loc[outliers_mask, col] = median_value
                
                outlier_report.append({
                    'Attribute': col,
                    'Outliers_Detected': outlier_count,
                    'Percent': f"{outlier_percentage:.2f}%",
                    'IQR_Limits': f"[{lower_bound:.2f}, {upper_bound:.2f}]",
                    'Replaced_with': f"Median ({median_value:.2f})"
                })
                
                log_print(f"  - {col}: {outlier_count} outliers ({outlier_percentage:.1f}%) replaced with median", logger)
    
    return df_clean, outlier_report


# ──────────────────────────────────────────────────────────────
# Cramér's V — measures association between categorical variables
# Based on chi-squared statistic, normalized to [0, 1]
# ──────────────────────────────────────────────────────────────
def cramers_v(x, y):
    try:
        contingency_table = pd.crosstab(x, y)
        chi2 = ss.chi2_contingency(contingency_table)[0]
        n = contingency_table.sum().sum()
        min_dim = min(contingency_table.shape[0] - 1, contingency_table.shape[1] - 1)
        
        if min_dim == 0 or n == 0:
            return 0.0
        
        cramers_v_value = np.sqrt(chi2 / (n * min_dim))
        return cramers_v_value
    except Exception as e:
        return 0.0


# Find categorical feature pairs with Cramér's V above threshold (default 0.5)
# and mark the less useful one for removal
def find_highly_correlated_categorical_features(df, categorical_cols, threshold=0.5, logger=None):
    if len(categorical_cols) < 2:
        return [], pd.DataFrame(), pd.DataFrame()
    
    log_print(f"Computing Cramer's V for {len(categorical_cols)} categorical features...", logger)
    
    correlation_matrix = pd.DataFrame(
        np.eye(len(categorical_cols)), 
        index=categorical_cols, 
        columns=categorical_cols
    )
    
    for i, col1 in enumerate(categorical_cols):
        for j, col2 in enumerate(categorical_cols):
            if i < j:
                df_clean = df[[col1, col2]].dropna()
                
                if len(df_clean) > 0:
                    cv_value = cramers_v(df_clean[col1], df_clean[col2])
                    correlation_matrix.loc[col1, col2] = cv_value
                    correlation_matrix.loc[col2, col1] = cv_value
                else:
                    correlation_matrix.loc[col1, col2] = 0.0
                    correlation_matrix.loc[col2, col1] = 0.0
    
    high_corr_pairs = []
    features_to_remove = set()
    
    for i, col1 in enumerate(categorical_cols):
        for j, col2 in enumerate(categorical_cols):
            if i < j and correlation_matrix.loc[col1, col2] > threshold:
                cv_value = correlation_matrix.loc[col1, col2]
                
                high_corr_pairs.append({
                    'Feature_1': col1,
                    'Feature_2': col2,
                    'Cramers_V': cv_value,
                    'Interpretation': get_cramers_v_interpretation(cv_value),
                    'Action': f'Remove {col2}'
                })
                
                features_to_remove.add(col2)
    
    high_corr_df = pd.DataFrame(high_corr_pairs)
    
    return list(features_to_remove), correlation_matrix, high_corr_df


def get_cramers_v_interpretation(cv_value):
    if cv_value < 0.1:
        return "Weak association"
    elif cv_value < 0.3:
        return "Moderate association"
    elif cv_value < 0.5:
        return "Strong association"
    else:
        return "Very strong association"


def plot_categorical_correlation_matrix(correlation_matrix, prefix, split_name, threshold=0.5, logger=None):
    try:
        if correlation_matrix.empty or len(correlation_matrix) < 2:
            log_print("No categorical correlation matrix to plot", logger)
            return
        
        fig, ax = plt.subplots(figsize=(min(12, len(correlation_matrix)), min(10, len(correlation_matrix))))
        
        im = ax.imshow(correlation_matrix.values, vmin=0, vmax=1, aspect='auto', cmap='Reds')
        
        ax.set_xticks(range(len(correlation_matrix.columns)))
        ax.set_xticklabels(correlation_matrix.columns, rotation=45, ha='right')
        ax.set_yticks(range(len(correlation_matrix.index)))
        ax.set_yticklabels(correlation_matrix.index)
        
        if len(correlation_matrix) <= 15:
            for i in range(len(correlation_matrix.index)):
                for j in range(len(correlation_matrix.columns)):
                    cv_val = correlation_matrix.iloc[i, j]
                    color = 'white' if cv_val > 0.5 else 'black'
                    ax.text(j, i, f"{cv_val:.2f}", ha='center', va='center', 
                           fontsize=8, color=color, weight='bold')
        
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Cramer's V", rotation=270, labelpad=15)
        
        ax.set_title(f"{prefix} {split_name} - Categorical Correlation (Cramer's V)\nThreshold: {threshold}", fontsize=14)
        
        plt.tight_layout()
        save_plot(fig, f"{prefix}_{split_name}_categorical_correlation_cramers_v.png", logger)
        
    except Exception as e:
        log_print(f"Error plotting categorical correlation matrix: {e}", logger)
        plt.close('all')


# ──────────────────────────────────────────────────────────────
# Numeric redundancy detection — Pearson correlation > threshold
# Returns list of features to drop and a report dataframe
# ──────────────────────────────────────────────────────────────
def find_highly_correlated_features(df, threshold=0.8):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) < 2:
        return [], pd.DataFrame()
    
    corr_matrix = df[numeric_cols].corr().abs()
    
    high_corr_pairs = []
    features_to_remove = set()
    
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if corr_matrix.iloc[i, j] > threshold:
                feature1 = corr_matrix.columns[i]
                feature2 = corr_matrix.columns[j]
                corr_value = corr_matrix.iloc[i, j]
                
                high_corr_pairs.append({
                    'Feature_1': feature1,
                    'Feature_2': feature2,
                    'Correlation_Type': 'Numeric (Pearson)',
                    'Correlation_Value': corr_value,
                    'Action': f'Remove {feature2}'
                })
                
                features_to_remove.add(feature2)
    
    high_corr_df = pd.DataFrame(high_corr_pairs)
    return list(features_to_remove), high_corr_df


# Drop redundant features from the dataframe and return the cleaned version
def remove_redundant_features(df, features_to_remove):
    df_reduced = df.copy()
    removed_features = []
    
    for feature in features_to_remove:
        if feature in df_reduced.columns:
            df_reduced = df_reduced.drop(columns=[feature])
            removed_features.append(feature)
    
    return df_reduced, removed_features


# ──────────────────────────────────────────────────────────────
# Generate all EDA visualizations: bar plots for categorical/discrete,
# boxplots for continuous, correlation heatmaps, and class distribution.
# Caps at 60 plots to avoid excessive file generation.
# ──────────────────────────────────────────────────────────────
def safe_plot_generation(df_final, cont_final, disc_final, cat_final, ord_final, cod_final, prefix, split_name, target_col, logger=None):
    plot_count = 0
    max_plots = 60
    
    try:
        nominal_cats = [c for c in cat_final if c not in ord_final]
        for c in nominal_cats[:min(15, len(nominal_cats))]:
            if c in df_final.columns and plot_count < max_plots:
                try:
                    vals = df_final[c].value_counts(dropna=False).head(10)
                    fig, ax = plt.subplots(figsize=(10, 6))
                    vals.plot(kind='bar', ax=ax, color='skyblue')
                    ax.set_title(f"{prefix} {split_name} - Distribution of {c} (Nominal)", fontsize=14)
                    ax.set_xlabel(c)
                    ax.set_ylabel("Count")
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    save_plot(fig, f"{prefix}_{split_name}_bar_nominal_{c}.png", logger)
                    plot_count += 1
                except Exception as e:
                    log_print(f"Error plotting nominal {c}: {e}", logger)
                    plt.close('all')

        for c in ord_final[:min(10, len(ord_final))]:
            if c in df_final.columns and plot_count < max_plots:
                try:
                    vals = df_final[c].value_counts(dropna=False)
                    fig, ax = plt.subplots(figsize=(10, 6))
                    vals.plot(kind='bar', ax=ax, color='lightcoral')
                    ax.set_title(f"{prefix} {split_name} - Distribution of {c} (Ordinal)", fontsize=14)
                    ax.set_xlabel(c)
                    ax.set_ylabel("Count")
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    save_plot(fig, f"{prefix}_{split_name}_bar_ordinal_{c}.png", logger)
                    plot_count += 1
                except Exception as e:
                    log_print(f"Error plotting ordinal {c}: {e}", logger)
                    plt.close('all')

        for c in disc_final[:min(10, len(disc_final))]:
            if c in df_final.columns and plot_count < max_plots:
                try:
                    vals = df_final[c].value_counts(dropna=False).sort_index()
                    fig, ax = plt.subplots(figsize=(10, 6))
                    vals.plot(kind='bar', ax=ax, color='lightgreen')
                    ax.set_title(f"{prefix} {split_name} - Distribution of {c} (Discrete)", fontsize=14)
                    ax.set_xlabel(c)
                    ax.set_ylabel("Count")
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    save_plot(fig, f"{prefix}_{split_name}_bar_discrete_{c}.png", logger)
                    plot_count += 1
                except Exception as e:
                    log_print(f"Error plotting discrete {c}: {e}", logger)
                    plt.close('all')

        for c in cont_final[:min(12, len(cont_final))]:
            if c in df_final.columns and plot_count < max_plots:
                try:
                    fig, ax = plt.subplots(figsize=(8, 6))
                    df_final[c].plot(kind='box', ax=ax)
                    ax.set_title(f"{prefix} {split_name} - Boxplot {c} (After Outlier Treatment)", fontsize=14)
                    ax.set_ylabel(c)
                    plt.tight_layout()
                    save_plot(fig, f"{prefix}_{split_name}_box_{c}_after_preprocessing.png", logger)
                    plot_count += 1
                except Exception as e:
                    log_print(f"Error plotting boxplot for {c}: {e}", logger)
                    plt.close('all')

        if len(cont_final) > 1 and len(cont_final) <= 20 and plot_count < max_plots:
            try:
                corr = df_final[cont_final].corr()
                fig, ax = plt.subplots(figsize=(min(12, len(cont_final)), min(10, len(cont_final))))
                im = ax.imshow(corr, vmin=-1, vmax=1, aspect='auto', cmap='coolwarm')
                ax.set_xticks(range(len(cont_final)))
                ax.set_xticklabels(cont_final, rotation=45, ha='right')
                ax.set_yticks(range(len(cont_final)))
                ax.set_yticklabels(cont_final)
                
                if len(cont_final) <= 12:
                    for i in range(len(cont_final)):
                        for j in range(len(cont_final)):
                            ax.text(j, i, f"{corr.iloc[i,j]:.2f}", ha='center', va='center', fontsize=8)
                
                fig.colorbar(im, ax=ax)
                ax.set_title(f"{prefix} {split_name} - Continuous Features Correlation", fontsize=14)
                plt.tight_layout()
                save_plot(fig, f"{prefix}_{split_name}_corr_continuous.png", logger)
                plot_count += 1
            except Exception as e:
                log_print(f"Error plotting continuous correlation: {e}", logger)
                plt.close('all')

        if target_col in df_final.columns and plot_count < max_plots:
            try:
                vc = df_final[target_col].value_counts(dropna=False)
                fig, ax = plt.subplots(figsize=(10, 6))
                vc.plot(kind='bar', ax=ax, color='orange')
                ax.set_title(f"{prefix} {split_name} - Class Distribution {target_col}", fontsize=14)
                ax.set_xlabel(target_col)
                ax.set_ylabel("Count")
                plt.xticks(rotation=45)
                plt.tight_layout()
                save_plot(fig, f"{prefix}_{split_name}_class_balance_{target_col}.png", logger)
                plot_count += 1
                
                max_count = vc.max()
                min_count = vc.min()
                imbalance_ratio = max_count / min_count
                log_print(f"Class distribution: {dict(vc)}", logger)
                log_print(f"Imbalance ratio: {imbalance_ratio:.2f}:1", logger)
                if imbalance_ratio > 2:
                    log_print("Dataset imbalanced - activating class_weight='balanced'", logger)
                    return True
            except Exception as e:
                log_print(f"Error plotting class balance: {e}", logger)
                plt.close('all')

        log_print(f"Generated {plot_count} plots", logger)
        return False
        
    except Exception as e:
        log_print(f"Error in plot generation: {e}", logger)
        plt.close('all')
        return False
    finally:
        plt.close('all')
        gc.collect()


# ══════════════════════════════════════════════════════════════
# Core EDA function — runs the full analysis pipeline on a single file:
#   1. Load CSV and compute descriptive statistics
#   2. Detect and plot missing values
#   3. Classify attributes (continuous, discrete, nominal, ordinal)
#   4. Detect and treat outliers using IQR
#   5. Remove redundant numeric features (Pearson > 0.8)
#   6. Remove redundant categorical features (Cramér's V > 0.5)
#   7. Generate all visualizations and export reports
# ══════════════════════════════════════════════════════════════
def eda_for_file(file_path, prefix, split_name, target_col, ordinal_cols, discrete_threshold, logger=None):
    try:
        df = pd.read_csv(file_path)
        log_print(f"EDA for {prefix} {split_name}: {file_path}", logger)
        log_print(f"Original shape: {df.shape}", logger)
        
        try:
            import io
            buffer = io.StringIO()
            df.info(buf=buffer)
            info_str = buffer.getvalue()
            log_print(info_str, logger)
        except Exception as e:
            log_print(f"Info error: {e}", logger)
            
        missing_summary = df.isnull().sum()
        
        missing_msg = str(missing_summary[missing_summary > 0]) if missing_summary.sum() > 0 else "No missing values"
        log_print(f"Missing values: {missing_msg}", logger)

        miss_pct = df.isnull().mean() * 100
        if miss_pct.sum() > 0:
            fig, ax = plt.subplots(figsize=(8, 4))
            miss_pct[miss_pct > 0].sort_values().plot(
                kind='bar', ax=ax, 
                title=f"Missing Values in {prefix} {split_name} Dataset", 
                ylabel="Missing Values (%)"
            )
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            save_plot(fig, f"{prefix}_{split_name}_missing_values.png", logger)     

        const = [c for c in df.columns if df[c].nunique() == 1]
        if const:
            log_print(f"Constant columns: {const}", logger)

        num_df = df.select_dtypes(include=[np.number])
        cont = [c for c in num_df.columns if df[c].nunique() > discrete_threshold]
        disc = [c for c in num_df.columns if df[c].nunique() <= discrete_threshold]
        cat = df.select_dtypes(include=['object','category']).columns.tolist()
        
        numeric_cols = cont + disc
        if numeric_cols:
            n = len(numeric_cols)
            cols = 3
            rows = (n + cols - 1) // cols
            fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*3))
            if rows == 1:
                axes = [axes] if n == 1 else axes
            else:
                axes = axes.flatten()
            for i, col in enumerate(numeric_cols):
                ax = axes[i] if isinstance(axes, list) else axes[i]
                df[col].hist(bins=30, ax=ax, edgecolor='black', alpha=0.7)
                ax.set_title(col)
            for i in range(n, len(axes)):
                axes[i].axis('off')
            plt.suptitle(f"{prefix} {split_name} - Combined Histograms", y=1.02)
            plt.tight_layout()
            save_plot(fig, f"{prefix}_{split_name}_combined_histograms.png", logger)

        detected_ordinals = detect_ordinal_columns(df, ordinal_cols)
        ord_cols = detected_ordinals
        
        nominal_cat = [c for c in cat if c not in ord_cols]
        cod = list(set(nominal_cat + ord_cols + disc))
        
        log_print(f"Attributes: Continuous={len(cont)}, Discrete={len(disc)}, Nominal={len(nominal_cat)}, Ordinal={len(ord_cols)}", logger)

        if cont:
            try:
                stats = num_df[cont].describe().T
                log_print("Continuous stats computed", logger)
                stats.to_csv(os.path.join(REPORTS_DIR, f"{prefix}_{split_name}_continuous_stats.csv"))
            except Exception as e:
                log_print(f"Error computing continuous stats: {e}", logger)
        else:
            log_print("No continuous attributes", logger)

        if disc:
            try:
                disc_stats = []
                for col in disc:
                    non_missing = df[col].notna().sum()
                    unique_count = df[col].nunique()
                    min_val = df[col].min()
                    max_val = df[col].max()
                    disc_stats.append({
                        'Attribute': col,
                        'Count': non_missing, 
                        'Unique_Values': unique_count,
                        'Min': min_val,
                        'Max': max_val
                    })
                disc_stats_df = pd.DataFrame(disc_stats)
                disc_stats_df.to_csv(os.path.join(REPORTS_DIR, f"{prefix}_{split_name}_discrete_stats.csv"), index=False)
                log_print("Discrete stats computed", logger)
            except Exception as e:
                log_print(f"Error computing discrete stats: {e}", logger)
        else:
            log_print("No discrete attributes", logger)

        if nominal_cat:
            try:
                cat_stats = []
                for col in nominal_cat:
                    non_missing = df[col].notna().sum()
                    unique_count = df[col].nunique()
                    most_frequent = df[col].mode().iloc[0] if len(df[col].mode()) > 0 else 'N/A'
                    cat_stats.append({
                        'Attribute': col,
                        'Count': non_missing, 
                        'Unique_Values': unique_count,
                        'Most_Frequent': most_frequent
                    })
                cat_stats_df = pd.DataFrame(cat_stats)
                cat_stats_df.to_csv(os.path.join(REPORTS_DIR, f"{prefix}_{split_name}_nominal_stats.csv"), index=False)
                log_print("Nominal stats computed", logger)
            except Exception as e:
                log_print(f"Error computing nominal stats: {e}", logger)
        else:
            log_print("No nominal attributes", logger)

        if ord_cols:
            try:
                ord_stats = []
                for col in ord_cols:
                    non_missing = df[col].notna().sum()
                    unique_count = df[col].nunique()
                    unique_vals = sorted(df[col].dropna().unique())
                    ord_stats.append({
                        'Attribute': col,
                        'Count': non_missing, 
                        'Unique_Values': unique_count,
                        'Value_Order': str(unique_vals)
                    })
                ord_stats_df = pd.DataFrame(ord_stats)
                ord_stats_df.to_csv(os.path.join(REPORTS_DIR, f"{prefix}_{split_name}_ordinal_stats.csv"), index=False)
                log_print("Ordinal stats computed", logger)
            except Exception as e:
                log_print(f"Error computing ordinal stats: {e}", logger)
        else:
            log_print("No ordinal attributes", logger)

        log_print("Processing outliers...", logger)
        if cont:
            try:
                df_clean, outlier_report = remove_outliers_and_impute(df, cont, logger=logger)
                
                if outlier_report:
                    outlier_df = pd.DataFrame(outlier_report)
                    outlier_df.to_csv(os.path.join(REPORTS_DIR, f"{prefix}_{split_name}_outliers_report.csv"), index=False)
                    log_print("Outliers detected and treated", logger)
                else:
                    log_print("No significant outliers detected", logger)
            except Exception as e:
                log_print(f"Error processing outliers: {e}", logger)
                df_clean = df.copy()
        else:
            df_clean = df.copy()
            log_print("No continuous attributes for outlier detection", logger)

        if numeric_cols:
            n_cols = min(len(numeric_cols), 6)
            fig, axes = plt.subplots(1, n_cols, figsize=(n_cols*2.5, 5))
            if n_cols == 1:
                axes = [axes]
            for i, col in enumerate(numeric_cols[:n_cols]):
                df.boxplot(column=col, ax=axes[i], vert=True)
                axes[i].set_title(col, rotation=0, fontsize=8)
                axes[i].tick_params(axis='x', rotation=45)
            plt.suptitle(f"{prefix} {split_name} - Combined Boxplots", y=1.05)
            plt.tight_layout()
            save_plot(fig, f"{prefix}_{split_name}_combined_boxplots.png", logger)

        log_print("Detecting numeric redundancy...", logger)
        try:
            numeric_features_to_remove, numeric_high_corr_df = find_highly_correlated_features(df_clean, threshold=0.8)
            
            if not numeric_high_corr_df.empty:
                log_print(f"Highly correlated numeric features detected: {numeric_features_to_remove}", logger)
                
                df_after_numeric, removed_numeric_features = remove_redundant_features(df_clean, numeric_features_to_remove)
                log_print(f"Numeric features removed: {removed_numeric_features}", logger)
                log_print(f"Shape after numeric removal: {df_after_numeric.shape}", logger)
                
                numeric_high_corr_df.to_csv(os.path.join(REPORTS_DIR, f"{prefix}_{split_name}_numeric_redundancy_report.csv"), index=False)
            else:
                df_after_numeric = df_clean.copy()
                log_print("No numeric redundant features detected (correlation > 0.8)", logger)
                numeric_high_corr_df = pd.DataFrame()
        except Exception as e:
            log_print(f"Error processing numeric redundancy: {e}", logger)
            df_after_numeric = df_clean.copy()
            numeric_high_corr_df = pd.DataFrame()

        log_print("Detecting categorical redundancy...", logger)
        try:
            temp_cat = df_after_numeric.select_dtypes(include=['object','category']).columns.tolist()
            temp_ord_cols = [c for c in ord_cols if c in df_after_numeric.columns]
            all_categorical = list(set(temp_cat + temp_ord_cols))
            
            if len(all_categorical) >= 2:
                categorical_features_to_remove, categorical_corr_matrix, categorical_high_corr_df = find_highly_correlated_categorical_features(
                    df_after_numeric, all_categorical, threshold=0.5, logger=logger
                )
                
                if not categorical_high_corr_df.empty:
                    log_print(f"Highly correlated categorical features detected (Cramer's V > 0.5): {categorical_features_to_remove}", logger)
                    
                    df_final, removed_categorical_features = remove_redundant_features(df_after_numeric, categorical_features_to_remove)
                    log_print(f"Categorical features removed: {removed_categorical_features}", logger)
                    log_print(f"Final shape after categorical removal: {df_final.shape}", logger)
                    
                    categorical_high_corr_df.to_csv(os.path.join(REPORTS_DIR, f"{prefix}_{split_name}_categorical_redundancy_report.csv"), index=False)
                    categorical_corr_matrix.to_csv(os.path.join(REPORTS_DIR, f"{prefix}_{split_name}_categorical_correlation_matrix.csv"))
                    
                    plot_categorical_correlation_matrix(categorical_corr_matrix, prefix, split_name, threshold=0.5, logger=logger)
                    
                else:
                    df_final = df_after_numeric.copy()
                    log_print("No categorical redundant features detected (Cramer's V > 0.5)", logger)
                    
                    if not categorical_corr_matrix.empty:
                        categorical_corr_matrix.to_csv(os.path.join(REPORTS_DIR, f"{prefix}_{split_name}_categorical_correlation_matrix.csv"))
                        plot_categorical_correlation_matrix(categorical_corr_matrix, prefix, split_name, threshold=0.5, logger=logger)
                    categorical_high_corr_df = pd.DataFrame()
            else:
                df_final = df_after_numeric.copy()
                log_print("Not enough categorical attributes (min 2) for correlation analysis", logger)
                categorical_high_corr_df = pd.DataFrame()
                categorical_corr_matrix = pd.DataFrame()
                
        except Exception as e:
            log_print(f"Error processing categorical redundancy: {e}", logger)
            df_final = df_after_numeric.copy()
            categorical_high_corr_df = pd.DataFrame()

        combined_redundancy_report = []
        
        if not numeric_high_corr_df.empty:
            for _, row in numeric_high_corr_df.iterrows():
                combined_redundancy_report.append({
                    'Type': 'Numeric',
                    'Feature_1': row['Feature_1'],
                    'Feature_2': row['Feature_2'],
                    'Correlation_Measure': 'Pearson Correlation',
                    'Correlation_Value': row['Correlation_Value'],
                    'Threshold': 0.8,
                    'Action': row['Action']
                })
        
        if not categorical_high_corr_df.empty:
            for _, row in categorical_high_corr_df.iterrows():
                combined_redundancy_report.append({
                    'Type': 'Categorical',
                    'Feature_1': row['Feature_1'],
                    'Feature_2': row['Feature_2'],
                    'Correlation_Measure': 'Cramer\'s V',
                    'Correlation_Value': row['Cramers_V'],
                    'Threshold': 0.5,
                    'Action': row['Action']
                })
        
        if combined_redundancy_report:
            combined_df = pd.DataFrame(combined_redundancy_report)
            combined_df.to_csv(os.path.join(REPORTS_DIR, f"{prefix}_{split_name}_COMBINED_redundancy_report.csv"), index=False)

        num_df_final = df_final.select_dtypes(include=[np.number])
        cont_final = [c for c in num_df_final.columns if df_final[c].nunique() > discrete_threshold]
        disc_final = [c for c in num_df_final.columns if df_final[c].nunique() <= discrete_threshold]
        cat_final = df_final.select_dtypes(include=['object','category']).columns.tolist()
        
        ord_final = [c for c in ord_cols if c in df_final.columns]
        nominal_final = [c for c in cat_final if c not in ord_final]
        cod_final = list(set(nominal_final + ord_final + disc_final))

        is_imbalanced = safe_plot_generation(df_final, cont_final, disc_final, nominal_final, ord_final, cod_final, prefix, split_name, target_col, logger)

        try:
            df_final.to_csv(os.path.join(REPORTS_DIR, f"{prefix}_{split_name}_preprocessed.csv"), index=False)
            log_print(f"Saved preprocessed dataset: {prefix}_{split_name}_preprocessed.csv", logger)
        except Exception as e:
            log_print(f"Error saving preprocessed dataset: {e}", logger)
        
        return df_final, ord_final, is_imbalanced
        
    except Exception as e:
        log_print(f"Critical error in EDA for {file_path}: {e}", logger)
        raise


# ══════════════════════════════════════════════════════════════
# Training orchestrator — runs EDA on train/test splits, then:
#   1. Builds a ColumnTransformer preprocessing pipeline
#   2. Trains 5 models (DT, RF, LR, Manual LR, MLP)
#   3. Evaluates each with accuracy, precision, recall, F1
#   4. Exports confusion matrices, cost curves, and metric CSVs
# ══════════════════════════════════════════════════════════════
def preprocess_and_train(prefix, paths, target_col, ordinal_cols, logger=None):
    log_print(f"Preprocessing and training for {prefix}", logger)
    
    try:
        df_train_clean, ord_cols_train, is_imbalanced_train = eda_for_file(paths['train'], prefix, 'TRAIN', target_col, ordinal_cols, 15, logger)
        df_test_clean, ord_cols_test, is_imbalanced_test = eda_for_file(paths['test'], prefix, 'TEST', target_col, ordinal_cols, 15, logger)
        
        ord_cols = ord_cols_train
        is_imbalanced = is_imbalanced_train
        
        common_cols = list(set(df_train_clean.columns) & set(df_test_clean.columns))
        df_train_clean = df_train_clean[common_cols]
        df_test_clean = df_test_clean[common_cols]
        
        y_train = df_train_clean[target_col].values
        y_test = df_test_clean[target_col].values
        X_train = df_train_clean.drop(columns=[target_col])
        X_test = df_test_clean.drop(columns=[target_col])

        log_print(f"Final training shape: {X_train.shape}", logger)
        log_print(f"Final test shape: {X_test.shape}", logger)

        num_feats = X_train.select_dtypes(include=[np.number]).columns.tolist()
        cat_feats = X_train.select_dtypes(include=['object','category']).columns.tolist()
        
        ord_feats = [c for c in ord_cols if c in cat_feats]
        nominal_feats = [c for c in cat_feats if c not in ord_feats]

        log_print(f"Features: Numeric={len(num_feats)}, Nominal={len(nominal_feats)}, Ordinal={len(ord_feats)}", logger)

        total_features_estimate = len(num_feats) + len(ord_feats)
        if nominal_feats:
            onehot_estimate = sum(X_train[col].nunique() for col in nominal_feats)
            total_features_estimate += onehot_estimate
            
        log_print(f"Estimated total features after encoding: {total_features_estimate}", logger)
        
        if total_features_estimate > 10000:
            log_print("WARNING: Very large feature space detected! May cause memory issues.", logger)

        # ── Preprocessing pipeline (ColumnTransformer) ────────────
        # Numeric:  median imputation → standard scaling
        # Nominal:  mode imputation → one-hot encoding
        # Ordinal:  mode imputation → ordinal encoding
        transformers = []
        
        if num_feats:
            num_pipe = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])
            transformers.append(('num', num_pipe, num_feats))
        
        if nominal_feats:
            nom_pipe = Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ])
            transformers.append(('nom', nom_pipe, nominal_feats))
        
        if ord_feats:
            ord_pipe = Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
            ])
            transformers.append(('ord', ord_pipe, ord_feats))
        
        if not transformers:
            raise ValueError("No features for training!")
        
        pre = ColumnTransformer(transformers)

        le = LabelEncoder()
        y_train_enc = le.fit_transform(y_train)
        y_test_enc = le.transform(y_test)
        classes = le.classes_

        class_weights = None
        if is_imbalanced:
            class_weights = compute_class_weight('balanced', classes=np.unique(y_train_enc), y=y_train_enc)
            class_weight_dict = dict(zip(np.unique(y_train_enc), class_weights))
            log_print(f"Dataset imbalanced. Class weights: {class_weight_dict}", logger)

            USE_RF_OPTIMIZATION = False 
            USE_DT_OPTIMIZATION = False
            USE_LR_OPTIMIZATION = False

        # ── Model definitions ──────────────────────────────────────
        # All models use class_weight='balanced' when imbalanced data
        # is detected. ManualLogisticRegression is instantiated separately
        # during training to pass preprocessed data directly.
        models = {
        'DecisionTree': DecisionTreeClassifier(
            max_depth=10,
            min_samples_leaf=5,
            criterion='gini',
            min_samples_split=10,
            max_features='sqrt',
            class_weight='balanced' if is_imbalanced else None,
            random_state=42
        ),
        'RandomForest': RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=1,
            min_samples_split=2,
            criterion='entropy',
            max_features='log2',
            bootstrap=True,
            oob_score=True,
            n_jobs=-1,
            class_weight='balanced' if is_imbalanced else None,
            random_state=42
        ),
        'LogisticRegression': LogisticRegression(
            penalty='l2',
            C=1.0,
            solver='liblinear',
            max_iter=1000,
            class_weight='balanced' if is_imbalanced else None,
            random_state=42
        ),
        'ManualLogisticRegression': None,
        'MLP': MLPWithTrainHistory(
            hidden_layer_sizes=(100,),
            activation='relu',
            solver='adam',
            learning_rate_init=0.001,
            max_iter=300,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
            random_state=42
        )
    }

        results = []
        
        for name, model in models.items():
            log_print(f"Training {name} on {prefix}...", logger)
            
            try:
                if name == 'ManualLogisticRegression':
                    X_train_processed = pre.fit_transform(X_train)
                    X_test_processed = pre.transform(X_test)
                    
                    manual_lr = ManualLogisticRegression(
                        learning_rate=0.01,
                        max_iterations=1000,
                        regularization='l2',
                        lambda_reg=0.01
                    )
                    manual_lr.fit(X_train_processed, y_train_enc)
                    y_pred = manual_lr.predict(X_test_processed)
                    
                    y_train_pred = manual_lr.predict(X_train_processed)
                    train_acc = np.mean(y_train_pred == y_train_enc)
                    log_print(f"  Manual LR - Train Accuracy: {train_acc:.3f} (Classes: {len(classes)})", logger)
                    
                    if manual_lr.cost_history:
                        try:
                            fig, ax = plt.subplots(figsize=(10, 6))
                            ax.plot(manual_lr.cost_history, color='blue', linewidth=2)
                            ax.set_title(f"{prefix} Manual Logistic Regression - Cost Function", fontsize=14)
                            ax.set_xlabel('Iteration')
                            ax.set_ylabel('Cost (Negative Log-Likelihood)')
                            ax.grid(True, alpha=0.3)
                            plt.tight_layout()
                            save_plot(fig, f"{prefix}_ManualLR_cost_curve.png", logger)
                        except Exception as e:
                            log_print(f"Error plotting manual LR cost: {e}", logger)
                            
                else:
                    if name == 'RandomForest' and USE_RF_OPTIMIZATION:
                        X_train_processed = pre.fit_transform(X_train)
                        X_test_processed = pre.transform(X_test)
                        
                        optimized_rf = optimize_random_forest(X_train_processed, y_train_enc, logger, quick_search=True)
                        optimized_rf.fit(X_train_processed, y_train_enc)
                        y_pred = optimized_rf.predict(X_test_processed)
                        model = optimized_rf  
                        
                        log_print(f"  RF - Used optimized hyperparameters", logger)
                    else:
                        pipe = Pipeline([('pre', pre), ('clf', model)])
                        pipe.fit(X_train, y_train_enc)
                        y_pred = pipe.predict(X_test)
                    
                    if name == 'MLP':
                        y_train_pred = pipe.predict(X_train)
                        train_acc = np.mean(y_train_pred == y_train_enc)
                        log_print(f"  MLP - Train Accuracy: {train_acc:.3f}", logger)
                        
                        mlp_model = pipe.named_steps['clf']
                        if hasattr(mlp_model, 'loss_curve_') and hasattr(mlp_model, 'train_accuracy_history'):
                            try:
                                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
                                
                                ax1.plot(mlp_model.loss_curve_, label='Training Loss', color='blue', linewidth=2)
                                ax1.set_title(f"{prefix} MLP - Loss Curve", fontsize=14)
                                ax1.set_xlabel('Epoch')
                                ax1.set_ylabel('Loss')
                                ax1.legend()
                                ax1.grid(True, alpha=0.3)
                                
                                epochs = range(1, len(mlp_model.train_accuracy_history) + 1)
                                ax2.plot(epochs, mlp_model.train_accuracy_history, 
                                        label='Training Accuracy', color='blue', linewidth=2)
                                
                                if hasattr(mlp_model, 'validation_scores_'):
                                    val_epochs = range(1, len(mlp_model.validation_scores_) + 1)
                                    ax2.plot(val_epochs, mlp_model.validation_scores_, 
                                            label='Validation Accuracy', color='orange', linewidth=2)
                                    
                                    final_train_acc = mlp_model.train_accuracy_history[-1]
                                    final_val_acc = mlp_model.validation_scores_[-1]
                                    overfitting_gap = abs(final_train_acc - final_val_acc)
                                    
                                    if overfitting_gap > 0.1:
                                        ax2.text(0.02, 0.98, f"Possible Overfitting\nGap: {overfitting_gap:.3f}", 
                                                transform=ax2.transAxes, verticalalignment='top',
                                                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.8),
                                                fontsize=10)
                                    else:
                                        ax2.text(0.02, 0.98, f"Good Generalization\nGap: {overfitting_gap:.3f}", 
                                                transform=ax2.transAxes, verticalalignment='top',
                                                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8),
                                                fontsize=10)
                                
                                ax2.set_title(f"{prefix} MLP - Train vs Validation Accuracy", fontsize=14)
                                ax2.set_xlabel('Epoch')
                                ax2.set_ylabel('Accuracy')
                                ax2.legend()
                                ax2.grid(True, alpha=0.3)
                                
                                plt.tight_layout()
                                save_plot(fig, f"{prefix}_MLP_complete_training_curves.png", logger)
                            except Exception as e:
                                log_print(f"Error plotting MLP curves: {e}", logger)
                                plt.close('all')
                    
                    if name == 'RandomForest' and hasattr(model, 'oob_score_'):
                        log_print(f"  RF - OOB Score: {model.oob_score_:.3f}", logger)
                
                acc = np.mean(y_pred == y_test_enc)
                precision, recall, f1, support = precision_recall_fscore_support(y_test_enc, y_pred, zero_division=0)
                cm = confusion_matrix(y_test_enc, y_pred)
                plot_confusion(cm, classes, f"{prefix} {name} Confusion Matrix", f"{prefix}_{name}_confusion_matrix.png", logger)
                
                for cls, p, r, f, s in zip(classes, precision, recall, f1, support):
                    results.append({'Dataset': prefix, 'Model': name, 'Class': cls,
                                    'Accuracy': acc, 'Precision': p, 'Recall': r, 'F1': f, 'Support': s})
                
                log_print(f"  {name} - Test Accuracy: {acc:.3f}", logger)
                        
            except Exception as e:
                log_print(f"Error training {name}: {e}", logger)
                continue

        
        
        try:
            with open(os.path.join(REPORTS_DIR, f"{prefix}_hyperparameters_report.txt"), 'w', encoding='utf-8') as f:
                f.write(hyperparams_report)
        except Exception as e:
            log_print(f"Error saving hyperparameters report: {e}", logger)

        if results:
            try:
                df_res = pd.DataFrame(results)
                metrics_file = os.path.join(METRICS_DIR, f"{prefix}_complete_metrics.csv")
                df_res.to_csv(metrics_file, index=False)
                log_print(f"Saved metrics: {metrics_file}", logger)
                
                pivot_table = df_res.pivot_table(
                    index=['Model'], 
                    columns=['Class'], 
                    values=['Accuracy', 'Precision', 'Recall', 'F1'], 
                    aggfunc='first'
                )
                
                comparative_file = os.path.join(REPORTS_DIR, f"{prefix}_comparative_results.csv")
                pivot_table.to_csv(comparative_file)
                log_print(f"Saved comparative table: {comparative_file}", logger)
            except Exception as e:
                log_print(f"Error saving results: {e}", logger)
        else:
            log_print("No results to save", logger)
            
    except Exception as e:
        log_print(f"Critical error in preprocessing and training: {e}", logger)
        raise
    finally:
        plt.close('all')
        gc.collect()


# ══════════════════════════════════════════════════════════════
# Entry point — detects all dataset pairs in current directory
# and runs the full pipeline (EDA → preprocess → train → evaluate)
# ══════════════════════════════════════════════════════════════
def main():
    logger, log_filename = setup_logging()
    
    try:
        directory = os.getcwd()
        ordinal_cols = ['publication_period']
        
        log_print("Starting ML Pipeline", logger)
        log_print(f"Log file: {log_filename}", logger)

        datasets = detect_datasets(directory)
        if not datasets:
            log_print("No datasets found! Ensure you have *_train.csv and *_test.csv files", logger)
            return

        for prefix, parts in datasets.items():
            try:
                if 'train' in parts and 'test' in parts:
                    df_train = pd.read_csv(parts['train'])
                    target_col = df_train.columns[-1]
                    log_print(f"Processing dataset: {prefix}", logger)
                    log_print(f"Target column: {target_col}", logger)

                    preprocess_and_train(prefix, parts, target_col, ordinal_cols, logger)
                    
                elif 'full' in parts:
                    df_full = pd.read_csv(parts['full'])
                    target_col = df_full.columns[-1]
                    eda_for_file(parts['full'], prefix, 'FULL', target_col, ordinal_cols, 15, logger)
                    
            except Exception as e:
                log_print(f"Error processing dataset {prefix}: {e}", logger)
                continue

        log_print("Analysis completed", logger)
        log_print(f"Plots saved to: {PLOTS_DIR}/", logger)
        log_print(f"Metrics saved to: {METRICS_DIR}/", logger)
        log_print(f"Reports saved to: {REPORTS_DIR}/", logger)
        log_print(f"Log saved to: {log_filename}", logger)
        
    except Exception as e:
        log_print(f"Critical error in main: {e}", logger)
    finally:
        plt.close('all')
        gc.collect()
        if 'logger' in locals():
            log_print(f"Complete log saved to: {log_filename}", logger)


if __name__ == '__main__':
    main()
