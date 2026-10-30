import pandas as pd
import joblib
from sklearn.metrics import accuracy_score

model = joblib.load('model.pkl')

test_df = pd.read_csv('Iris - Iris.csv')
df = test_df.drop("Id", axis=1)

X = df.drop("Species", axis=1)
y = df["Species"]
pred = model.predict(X)
accuracy = accuracy_score(pred, y)
print("Accuracy:", accuracy)

# This is the part of software testing
assert accuracy >= 0.81
print("Test case 1 passed")
assert len(pred) == len(X)
print("Test case 2 passed")