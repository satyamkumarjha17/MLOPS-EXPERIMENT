import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

df = pd.read_csv("EXPERIMENT-18/Student_Performance_DT - Student_Performance_DT.csv")

print("Dataset Shape:", df.shape)
print("\nFirst 5 Rows:")
print(df.head())

print("\nMissing Values:")
print(df.isnull().sum())

X = df.drop("Final_Result", axis=1)
y = df["Final_Result"]

categorical_cols = X.select_dtypes(include=["object", "str"]).columns
numeric_cols = X.select_dtypes(exclude=["object"]).columns

preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Samples:", len(X_train))
print("Testing Samples:", len(X_test))

default_model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", DecisionTreeClassifier(random_state=42))
])

default_model.fit(X_train, y_train)

y_pred_default = default_model.predict(X_test)

def evaluate_model(y_true, y_pred):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(
            y_true, y_pred,
            average="weighted",
            zero_division=0
        ),
        "Recall": recall_score(
            y_true, y_pred,
            average="weighted",
            zero_division=0
        ),
        "F1-score": f1_score(
            y_true, y_pred,
            average="weighted",
            zero_division=0
        )
    }

default_metrics = evaluate_model(y_test, y_pred_default)

param_grid = {
    "classifier__criterion": [
        "gini",
        "entropy",
        "log_loss"
    ],
    "classifier__max_depth": [
        None,
        3,
        5,
        7,
        10
    ],
    "classifier__min_samples_split": [
        2,
        5,
        10
    ],
    "classifier__min_samples_leaf": [
        1,
        2,
        4
    ],
    "classifier__class_weight": [
        None,
        "balanced"
    ]
}

grid_search = GridSearchCV(
    estimator=Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", DecisionTreeClassifier(random_state=42))
    ]),
    param_grid=param_grid,
    cv=5,
    scoring="f1_weighted",
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print("\nBest Hyperparameters:")
print(grid_search.best_params_)

print("\nBest Cross-Validation F1-score:")
print(grid_search.best_score_)

tuned_model = grid_search.best_estimator_

y_pred_tuned = tuned_model.predict(X_test)

tuned_metrics = evaluate_model(y_test, y_pred_tuned)

print("\nDEFAULT DECISION TREE")

for metric, value in default_metrics.items():
    print(f"{metric}: {value:.4f}")

print("\nTUNED DECISION TREE")

for metric, value in tuned_metrics.items():
    print(f"{metric}: {value:.4f}")

print("\nTuned Model Classification Report:")
print(
    classification_report(
        y_test,
        y_pred_tuned,
        zero_division=0
    )
)

comparison = pd.DataFrame(
    [default_metrics, tuned_metrics],
    index=[
        "Default Decision Tree",
        "Tuned Decision Tree"
    ]
)

print("\nMODEL COMPARISON")
print(comparison.round(4))

difference = (
    comparison.loc["Tuned Decision Tree"]
    - comparison.loc["Default Decision Tree"]
)

print("\nPerformance Change:")
print(difference.round(4))

if tuned_metrics["F1-score"] > default_metrics["F1-score"]:
    print("\nHyperparameter tuning IMPROVED the model.")
else:
    print("\nHyperparameter tuning DID NOT IMPROVE the model.")