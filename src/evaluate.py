"""
Module: Evaluation
------------------
Role: Generate metrics and plots for model performance.
Input: Trained Model + Test Data.
Output: Metrics dictionary and plots saved to `reports/`.
"""

"""
Educational Goal:
- Why this module exists in an MLOps system: Ensures objective, automated assessment of model performance before deployment.
- Responsibility (separation of concerns): Scoring the fitted model on held-out test data.
- Pipeline contract (inputs and outputs): Takes a fitted model and test data, returns a single float metric.

TODO: Replace print statements with standard library logging in a later session
TODO: Any temporary or hardcoded variable or parameter will be imported from config.yml in a later session
"""

from sklearn.metrics import mean_squared_error, f1_score, roc_auc_score
import pandas as pd


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series, problem_type: str) -> float:
    """
    Inputs:
    - model: The fitted scikit-learn Pipeline.
    - X_test: Test features DataFrame.
    - y_test: Test target Series.
    - problem_type: "regression" or "classification".
    Outputs:
    - score: RMSE (regression) or ROC AUC (classification) as a float.
    Why this contract matters for reliable ML delivery:
    - A stable evaluation contract makes experimetns comparable and reduces accidental metric drift.
    """
    print(f"[evaluate.evaluate_model Evaluating model for problem_type={problem_type}")  # TODO: replace with logging later

    if not isinstance(y_test, pd.Series):
        y_test = pd.Series(y_test)

    # --------------------------------------------------------
    # START STUDENT CODE
    # --------------------------------------------------------
    # TODO_STUDENT: Paste your notebook logic here to replace or extend the baseline
    # Why: Different business problems require different optimization metrics (e.g., Precision, Recall, MAE).
    # Examples:
    # 1. return float(roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))
    # 2. return float(mean_absolute_error(y_test, predictions))
    #
    # Optional forcing function (leave commented)
    # raise NotImplementedError("Student: You must implement this logic to proceed!")
    #
    # Placeholder (Remove this after implementing your code):
    print("Warning: Student has not implemented this section yet")
    # --------------------------------------------------------
    # END STUDENT CODE
    # --------------------------------------------------------

    if problem_type == "regression":
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = float(mse ** 0.5)
        return rmse

    if problem_type == "classification":
        # Preferred path: AUC from predicted probabilities
        try:
            y_proba = model.predict_proba(X_test)

            # Binary classification: proba shape (n_samples, 2)
            if hasattr(y_proba, "shape") and len(y_proba.shape) == 2 and y_proba.shape[1] == 2:
                auc = float(roc_auc_score(y_test, y_proba[:, 1]))
                return auc

            # Multiclass: proba shape (n_samples, n_classes)
            auc = float(roc_auc_score(y_test, y_proba, multi_class="ovr", average="weighted"))
            return auc

        except Exception as e_prob:
            print(
                f"[evaluate.evaluate_model] predict_proba failed ({e_prob}). Trying decision_function..."
            )  # TODO: replace with logging later

            # Some estimators expose decision_function instead of predict_proba
            try:
                y_score = model.decision_function(X_test)

                # Binary: decision_function shape (n_samples,)
                if hasattr(y_score, "ndim") and y_score.ndim == 1:
                    auc = float(roc_auc_score(y_test, y_score))
                    return auc

                # Multiclass: decision_function shape (n_samples, n_classes)
                auc = float(roc_auc_score(y_test, y_score, multi_class="ovr", average="weighted"))
                return auc

            except Exception as e_df:
                print(
                    f"[evaluate.evaluate_model] decision_function failed ({e_df}). Falling back to weighted F1."
                )  # TODO: replace with logging later

                # Final fallback: return weighted F1 from predicted labels
                y_pred = model.predict(X_test)
                f1 = float(f1_score(y_test, y_pred, average="weighted"))
                return f1

    raise ValueError("problem_type must be either 'regression' or 'classification'")
