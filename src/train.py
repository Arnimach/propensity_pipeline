import os
import numpy as np
import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import fbeta_score
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.base import clone

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Point to local MLFlow server
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment('Propensity_Model_production')


# custom MLFlow Wrapper for Rank-Averaging
class RankAverageEnsemblePyfunc(mlflow.pyfunc.PythonModel):
    """
     mlflow wrapper class for ensemble
     rank averaging for customer propensity
    """
    def __init__(self, selected_models, best_threshold, selected_pipelines):
        self.selected_models = selected_models
        self.best_threshold = best_threshold
        self.selected_pipelines = selected_pipelines

    def predict(self, context, model_input):
        base_pred = {}

        for model_name in self.selected_models:
            pipeline = self.selected_pipelines[model_name]
            pred_prob = pipeline.predict_proba(model_input)[:, 1]
            base_pred[model_name] = pred_prob

        df_pred = pd.DataFrame(base_pred)
        blend_pred_prob = df_pred.mean(axis=1).values

        return (blend_pred_prob >= self.best_threshold).astype(int)


def create_models(n_seeds=5):
    all_models = []

    rf_params = [
        {'n_estimators': 100, 'max_depth': 10, 'min_samples_split': 5},
        {'n_estimators': 300, 'max_depth': 20, 'min_samples_split': 2},
        {'n_estimators': 500, 'max_depth': 30, 'min_samples_split': 5}
    ]

    xgb_params = [
        {'n_estimators': 1000, 'learning_rate': 0.03, 'max_depth': 5},
        {'n_estimators': 3000, 'learning_rate': 0.01, 'max_depth': 7},
        {'n_estimators': 10000, 'learning_rate': 0.001, 'max_depth': 10}

    ]

    lgbm_params = [
        {'learning_rate': 0.02, 'n_estimators': 3000, 'num_leaves': 63, 'max_depth': 8},
        {'learning_rate': 0.03, 'n_estimators': 2000, 'num_leaves': 31, 'max_depth': 6},
        {'learning_rate': 0.05, 'n_estimators': 1500, 'num_leaves': 15, 'max_depth': 4}
    ]

    seeds = range(n_seeds)

    for seed in seeds:
        for j, params in enumerate(rf_params):
            rf_model = (f"rf_{j}_{seed}", RandomForestClassifier(class_weight="balanced", random_state=seed, **params))
            all_models.append(rf_model)

        for k, params in enumerate(xgb_params):
            xgb_model = (f"xgb_{k}_{seed}", XGBClassifier(objective='binary:logistic', random_state=seed, **params))
            all_models.append(xgb_model)

        for l, params in enumerate(lgbm_params):
            lgb_model = (f"lgb_{l}_{seed}",
                         LGBMClassifier(objective='binary', is_unbalance=True, verbosity=-1, random_state=seed, **params))
            all_models.append(lgb_model)

    print(f"We have set up a total of {len(all_models)} models.")

    return all_models


def creat_oof_preds(X, y, models, cat_features, num_features):
    # Dictionary to store each model's full OOF predictions array
    oof_preds = {}

    # k-fold object for cross-validation
    skf = StratifiedKFold(shuffle=True, n_splits=5, random_state=42)

    # preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_features),
                      ('num', RobustScaler(), num_features)
                      ]
    )

    for name, base_model in models:
        oof_model = np.zeros(len(X))

        for train_idx, val_idx in skf.split(X, y):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            pipeline = Pipeline(steps=
                                [('pre', preprocessor),
                                 ('model', clone(base_model))
                                 ])

            pipeline.fit(X_train, y_train)
            oof_model[val_idx] = pipeline.predict_proba(X_val)[:, 1]

        oof_preds[name] = oof_model

    return oof_preds


def get_ranked_predictions(oof_predictions, y, top_k=30, corr_threshold=0.95):
    # Step 1: Score all  models using F2 Score (tuning threshold)
    print(f"Scoring all candidate models using f2 score...")
    model_metrics = []

    # Ensure y is converted to a clean NumPy array to avoid Pandas index alignment issues
    y_arr = y.values if hasattr(y, 'values') else y

    for model, y_prob in oof_predictions.items():
        best_f2_score = 0
        best_threshold = 0.5

        for threshold in np.linspace(0.05, 0.95, 19):
            y_pred = (y_prob >= threshold).astype(int)
            f2_score = fbeta_score(y_arr, y_pred, beta=2, zero_division=0)

            if f2_score > best_f2_score:
                best_f2_score = f2_score
                best_threshold = threshold

        model_metrics.append({
            "model_name": model,
            "f2_score": best_f2_score,
            "threshold": best_threshold
        })

    df_scores = \
        pd.DataFrame(model_metrics).sort_values(by=["f2_score"], ascending=False)

    # Step 2: Filter out redundant (highly correlated) models
    print("Filtering for prediction diversity...")
    df_oofs = pd.DataFrame(oof_predictions)
    selected_models = []

    for model_name in df_scores["model_name"]:

        if len(selected_models) >= top_k:
            break

        if not selected_models:
            selected_models.append(model_name)

        else:
            corr = df_oofs[selected_models].corrwith(df_oofs[model_name])

            if corr.max() < corr_threshold:
                selected_models.append(model_name)

    print(f"Selected {len(selected_models)} diverse top-performing models!")

    ranked_oofs = df_oofs[selected_models].rank(pct=True)
    final_blend_oof = ranked_oofs.mean(axis=1)

    return selected_models, final_blend_oof, df_scores


def train():

    # step#1 load the clean data file
    df = pd.read_csv("data/clean_data.csv")

    # define categorical and numerical columns
    cat_features = [
        'OperatingSystems',
        'Browser',
        'Region',
        'TrafficType',
        'Month',
        'VisitorType',
        'Weekend'
    ]

    num_features = [
        'Administrative',
        'Administrative_Duration',
        'Informational',
        'Informational_Duration',
        'ProductRelated',
        'ProductRelated_Duration',
        'BounceRates',
        'ExitRates',
        'PageValues',
        'SpecialDay'

    ]
    X = df[cat_features + num_features]
    y = df['Revenue']

    # step# 2 Get all candidate models
    all_models = create_models(n_seeds=5)

    # step#3 Get out-of-fold-predictions
    oof_predictions = creat_oof_preds(X, y, all_models, cat_features, num_features)

    # step#4 Get blended predictions
    selected_models,  final_blend_oofs,  _ = get_ranked_predictions(oof_predictions, y)

    # step#5 define pre-processor for the prediction pipeline
    preprocessor = ColumnTransformer(
        transformers=[('pre', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_features),
                      ('num', RobustScaler(), num_features)
                      ]
    )

    # define final model name
    model_name = "PropensityModel"

    with mlflow.start_run(run_name='Rank_Average_Ensemble_Run') as run:

        # find the optimal threshold on blended OOF scores
        # convert y to np array
        y_arr = y.values if hasattr(y, 'values') else y
        # initialize the best score and best thr
        best_f2_score, best_threshold = 0, 0.5

        for thr in np.linspace(0.05, 0.95, 91):
            y_pred = (final_blend_oofs >= thr).astype(int)
            f2_score = fbeta_score(y_arr, y_pred, beta=2, zero_division=0)

            # save the best threshold and score
            if f2_score > best_f2_score:
                best_f2_score = f2_score
                best_threshold = thr

        # fit selected model pipelines on the entire data
        selected_pipelines = {}

        for name, model in all_models:
            if name in selected_models:
                base_model = model
                full_pipeline = Pipeline(steps=[
                ('pre', clone(preprocessor)),
                ('model', clone(base_model))
                ])

                full_pipeline.fit(X, y)

                selected_pipelines[name] = full_pipeline

        # log final model parameters and metrics
        mlflow.log_params({
            "best_threshold" : best_threshold,
            "total_candidate_models": len(all_models),
            "total_selected_models": len(selected_models)

        })

        mlflow.log_metric("model_f2_score", best_f2_score)

        # build pyfunc and log to mlflow
        ensemble_model = RankAverageEnsemblePyfunc(
            selected_models=selected_models,
            best_threshold=best_threshold,
            selected_pipelines=selected_pipelines)

        result = mlflow.pyfunc.log_model(
            artifact_path='model',
            python_model=ensemble_model,
            registered_model_name=model_name
        )

        print(f"Rank ensemble logged! Best f2 score {best_f2_score} @ threshold value {best_threshold}")

    # set production alias
    client = MlflowClient()
    latest_version = result.registered_model_version
    client.set_registered_model_alias(
        name=model_name,
        alias="Production",
        version=latest_version
    )
    print(f"Registered model version {latest_version} promoted to @Production!")


if __name__ == "__main__":
    train()
