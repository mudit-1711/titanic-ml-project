import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

df = pd.read_csv("data/train.csv")

X = df[["Pclass", "Age", "Fare"]]

y = df["Survived"]

X = X.fillna(X.mean())

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = LogisticRegression()

model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)

print("Accuracy:", accuracy)

pickle.dump(model, open("model.pkl", "wb"))

print("Model saved")