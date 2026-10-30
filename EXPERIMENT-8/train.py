import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

df = pd.read_csv('Iris - Iris.csv')
df = df.drop("Id", axis=1)

y = df["Species"]
X = df.drop("Species", axis=1)

x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=200)
model.fit(x_train, y_train)

pred = model.predict(x_test)
accuracy = accuracy_score(y_test, pred)

print("Accuracy:", accuracy)

joblib.dump(model, 'model.pkl')
print("Model saved successfully!")
