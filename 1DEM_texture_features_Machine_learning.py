import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import optuna
import seaborn as sns
import shap

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


output_dir = ''
os.makedirs(output_dir, exist_ok=True)


df = pd.read_csv(os.path.join(output_dir, 'multi_geomorph_features.csv'))
X = df.drop(columns=['label', 'filename', 'geomorph'])
y = df['label']
feature_names = X.columns.tolist()


X = X.dropna()
y = y.loc[X.index]


scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

def optimize_model(trial, model_name):
    if model_name == 'Random Forest':
        return RandomForestClassifier(
            n_estimators=trial.suggest_int('n_estimators', 50, 300),
            max_depth=trial.suggest_int('max_depth', 3, 20),
            min_samples_split=trial.suggest_int('min_samples_split', 2, 10),
            random_state=42
        )
    elif model_name == 'Logistic Regression':
        return LogisticRegression(
            penalty='l1',
            solver='liblinear',
            C=trial.suggest_loguniform('C', 1e-3, 10),
            max_iter=1000
        )
    elif model_name == 'Linear SVM':
        return SVC(
            kernel='linear',
            C=trial.suggest_loguniform('C', 1e-3, 10),
            probability=True
        )
    elif model_name == 'XGBoost':
        return XGBClassifier(
            n_estimators=trial.suggest_int('n_estimators', 50, 300),
            max_depth=trial.suggest_int('max_depth', 3, 15),
            learning_rate=trial.suggest_loguniform('learning_rate', 0.01, 0.3),
            subsample=trial.suggest_float('subsample', 0.5, 1.0),
            colsample_bytree=trial.suggest_float('colsample_bytree', 0.5, 1.0),
            gamma=trial.suggest_float('gamma', 0, 5),
            reg_alpha=trial.suggest_float('reg_alpha', 0, 5),
            reg_lambda=trial.suggest_float('reg_lambda', 0, 5),
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=42
        )
    elif model_name == 'LightGBM':
        return LGBMClassifier(
            n_estimators=trial.suggest_int('n_estimators', 50, 300),
            max_depth=trial.suggest_int('max_depth', 3, 15),
            learning_rate=trial.suggest_loguniform('learning_rate', 0.01, 0.3),
            subsample=trial.suggest_float('subsample', 0.5, 1.0),
            colsample_bytree=trial.suggest_float('colsample_bytree', 0.5, 1.0),
            reg_alpha=trial.suggest_float('reg_alpha', 0.0, 5.0),
            reg_lambda=trial.suggest_float('reg_lambda', 0.0, 5.0),
            random_state=42
        )
    elif model_name == 'CatBoost':
        return CatBoostClassifier(
            iterations=trial.suggest_int('iterations', 50, 300),
            depth=trial.suggest_int('depth', 3, 10),
            learning_rate=trial.suggest_loguniform('learning_rate', 0.01, 0.3),
            l2_leaf_reg=trial.suggest_float('l2_leaf_reg', 1.0, 10.0),
            verbose=0,
            random_state=42
        )
    else:
        raise ValueError("Unknown model")


model_names = ['Random Forest', 'Logistic Regression', 'Linear SVM', 'XGBoost', 'LightGBM', 'CatBoost']
best_model_dict = {}

# 超参数搜索
for model_name in model_names:
    print(f"\n  {model_name} best parameters")

    def objective(trial):
        model = optimize_model(trial, model_name)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        return acc

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=300)

    best_params = study.best_params
    print(f"best parameters {model_name}：{best_params}")
    best_model = optimize_model(optuna.trial.FixedTrial(best_params), model_name)
    best_model_dict[model_name] = best_model


print("\n StackingClassifier...")
stacking_estimators = [(name.replace(" ", "_"), model) for name, model in best_model_dict.items()]
meta_model =  XGBClassifier(random_state=42)
stacking_clf = StackingClassifier(estimators=stacking_estimators, final_estimator=meta_model, passthrough=True)
best_model_dict['Stacking Ensemble'] = stacking_clf


for model_name, model in best_model_dict.items():
    print(f"\n：{model_name}")
    try:
        model.fit(X_train, y_train)
    except:
        model.fit(X_train.astype(np.float32), y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"{acc:.4f}")

    model_save_path = os.path.join(output_dir, f'{model_name.replace(" ", "_")}_best_model.pkl')
    joblib.dump(model, model_save_path)
    print(f"{model_save_path}")


    try:
        top_n = 10
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_).flatten()
        else:
            importances = None

        if importances is not None:
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            }).sort_values(by='Importance', ascending=False)

            importance_csv = os.path.join(output_dir, f'{model_name.replace(" ", "_")}_feature_importance.csv')
            importance_df.to_csv(importance_csv, index=False)

            importance_df_top = importance_df.head(top_n)
            plt.figure(figsize=(10, 6))
            plt.barh(importance_df_top['Feature'], importance_df_top['Importance'], color='skyblue')
            plt.xlabel('Importance')
            plt.title(f'{model_name} Top {top_n} Feature Importance')
            plt.gca().invert_yaxis()
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.tight_layout()
            fig_path = os.path.join(output_dir, f'{model_name}_feature_importance_top{top_n}.svg')
            plt.savefig(fig_path)
            plt.close()
            print(f" {model_name} {fig_path}")
    except Exception as e:
        print(f" {model_name} {e}")


    try:
        explainer = shap.Explainer(model.predict, X_train)
        shap_values = explainer(X_test)

        shap.summary_plot(shap_values, X_test, feature_names=feature_names, plot_type='bar', show=False)
        plt.tight_layout()
        shap_bar_path = os.path.join(output_dir, f'{model_name}_shap_summary_bar.svg')
        plt.savefig(shap_bar_path)
        plt.close()
        print(f" {model_name} {shap_bar_path}")


        shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
        plt.tight_layout()
        shap_beeswarm_path = os.path.join(output_dir, f'{model_name}_shap_summary_beeswarm.svg')
        plt.savefig(shap_beeswarm_path)
        plt.close()
        print(f" {model_name} SHAP beeswarm ：{shap_beeswarm_path}")
    except Exception as e:
        print(f" {model_name} SHAP {e}")



def cross_val_accuracy(model, X, y, cv=5):
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    return cross_val_score(model, X, y, cv=skf, scoring='accuracy', n_jobs=-1)

cv_scores_dict = {}
for model_name, model in best_model_dict.items():
    print(f" {model_name} ")
    try:
        scores = cross_val_accuracy(model, X_scaled, y, cv=5)
    except:
        scores = cross_val_accuracy(model, X_scaled.astype(np.float32), y, cv=5)
    cv_scores_dict[model_name] = scores
    print(f"：{scores.mean():.4f} ± {scores.std():.4f}")

cv_df = pd.DataFrame({k: pd.Series(v) for k, v in cv_scores_dict.items()})
plt.figure(figsize=(10, 6))
sns.boxplot(data=cv_df, palette='Set2')
plt.ylabel('Cross-Validation Accuracy')
plt.title('Model Accuracy Comparison (5-Fold CV)')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'model_cv_accuracy_boxplot.svg'))
plt.close()
