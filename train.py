import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

df = pd.read_csv("data/train.csv")

X = df[["Pclass", "Age", "Fare","Sex"]]
X["Sex"] = X["Sex"].map({
    "male": 1,
    "female": 0
})
y = df["Survived"]
X["Age"] = X["Age"].fillna(X["Age"].mean())
X["Fare"] = X["Fare"].fillna(X["Fare"].mean())

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = RandomForestClassifier( n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)

print("Accuracy:", accuracy)

pickle.dump(model, open("model.pkl", "wb"))

print("Model saved")