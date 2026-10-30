import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, classification_report

data = pd.read_csv("EXPERIMENT-17/Titanic-Dataset - Titanic-Dataset.csv")

X = data.drop(columns=["Survived", "PassengerId", "Name", "Ticket", "Cabin"])
y = data["Survived"]

categorical_features = ["Gender", "Embarked"]
numerical_features = ["Pclass", "Age", "SibSp", "Parch", "Fare"]

numerical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", drop="first"))
])

preprocessor = ColumnTransformer([
    ("num", numerical_pipeline, numerical_features),
    ("cat", categorical_pipeline, categorical_features)
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", DecisionTreeClassifier(
        criterion="gini",
        max_depth=5,
        random_state=42
    ))
])

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print("Decision Tree Classification")
print("--------------------------------")
print("Training Samples:", len(X_train))
print("Testing Samples:", len(X_test))

print("\nAccuracy:", accuracy)

print("\nConfusion Matrix:")
print(confusion)

print("\nPrecision:", precision)

print("Recall:", recall)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

new_passenger = pd.DataFrame({
    "Pclass": [3],
    "Gender": ["male"],
    "Age": [25],
    "SibSp": [0],
    "Parch": [0],
    "Fare": [10.0],
    "Embarked": ["S"]
})

prediction = model.predict(new_passenger)

if prediction[0] == 1:
    print("\nPrediction: Passenger Survived")
else:
    print("\nPrediction: Passenger Did Not Survive")