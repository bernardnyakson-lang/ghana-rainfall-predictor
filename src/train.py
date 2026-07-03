import os
import sys
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, classification_report
from xgboost import XGBClassifier
import lightgbm as lgb
from scipy.stats import randint, loguniform, uniform

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from preprocessing import load_data, prepare_features
from config import MODEL_PATH, RANDOM_STATE, TARGET_CLASSES

def get_models_and_params(numerical_cols):
    """
    Define all models. Use class_weight='balanced' for all
    to handle the severe class imbalance (87.9% NORAIN).
    """
    models = {
        'Decision Tree': (
            DecisionTreeClassifier(class_weight='balanced', random_state=RANDOM_STATE),
            {
                'max_depth':        randint(3, 15),
                'min_samples_leaf': randint(1, 20)
            }
        ),
        'Random Forest': (
            RandomForestClassifier(class_weight='balanced', random_state=RANDOM_STATE),
            {
                'n_estimators':     randint(50, 200),
                'max_depth':        randint(3, 15),
                'min_samples_leaf': randint(1, 20)
            }
        ),
        'XGBoost': (
            XGBClassifier(scale_pos_weight=1, random_state=RANDOM_STATE, eval_metric='mlogloss'),
            {
                'n_estimators':  randint(50, 200),
                'max_depth':     randint(3, 10),
                'learning_rate': uniform(0.01, 0.3)
            }
        ),
        'LightGBM': (
            lgb.LGBMClassifier(class_weight='balanced', random_state=RANDOM_STATE),
            {
                'n_estimators':  randint(50, 200),
                'max_depth':     randint(3, 10),
                'learning_rate': uniform(0.01, 0.3)
            }
        ),
        # Add Logistic Regression
        'Logistic Regression': (
            Pipeline([
                ('preprocessor', ColumnTransformer(
                    transformers=[
                        ('scaler', StandardScaler(), NUMERICAL_COLS)
                    ],
                    remainder='passthrough'
                )),
                ('model', LogisticRegression(
                    class_weight='balanced',
                    random_state=RANDOM_STATE,
                    max_iter=2000,
                    
                ))
            ]),
            {
                'model__C': loguniform(0.01, 10)
            }
        ),
    }
    
    return models


def tune_and_compare(models, X_train, y_train, X_test, y_test, n_iter=20, cv=5):
    """
    Run RandomizedSearchCV for each model.
    Use scoring='f1_macro' because this is a multi-class problem.
    """
    results = []
    best_models = {}

    for name, (model, params) in models.items():
        print(f'Tuning {name}...')

        search = RandomizedSearchCV(
            model,
            params,
            n_iter=n_iter,
            scoring='f1_macro',
            cv=cv,
            n_jobs=-1,
            random_state=RANDOM_STATE
        )
        search.fit(X_train, y_train)

        y_pred = search.predict(X_test)
        best_model = search.best_estimator_

        f1 = f1_score(y_test, y_pred, average='macro')
        print(f"{name} F1 Macro: {f1:.4f}")

        print(classification_report(y_test, y_pred, target_names=TARGET_CLASSES))

        results.append({
            'Model': name,
            'CV F1 Macro': search.best_score_,
            'Test F1 Macro': f1,
            'Best_Params': search.best_params_
        })
        best_models[name] = best_model

    results_df = pd.DataFrame(results).sort_values('Test F1 Macro', ascending=False)
    return results_df, best_models


def save_model(model, path=MODEL_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(model, f)
    print(f'Model saved: {path}')


if __name__ == '__main__':
    df = load_data()
    print(f'Data loaded: {df.shape}')

    X_train, X_test, y_train, y_test = prepare_features(df)
    print(f'Train: {X_train.shape}  |  Test: {X_test.shape}')
    print(f'Target classes: {TARGET_CLASSES}')

    models = get_models_and_params()

    print('\nRunning RandomizedSearchCV...\n')
    results_df, best_models = tune_and_compare(models, X_train, y_train, X_test, y_test)

    print('\n--- Model Comparison ---')
    print(results_df[['Model', 'CV F1 Macro', 'Test F1 Macro']].to_string(index=False))

    best_name  = results_df.iloc[0]['Model']
    best_model = best_models[best_name]
    print(f'\nBest model: {best_name}')

    save_model(best_model)
    print('Training complete.')
