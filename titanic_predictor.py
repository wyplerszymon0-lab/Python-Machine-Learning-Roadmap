import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

df = df[['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch']].dropna()
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

X = df.drop('Survived', axis=1)
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print(f"Dataset size: {len(df)} passengers")
print(f"Model Accuracy: {accuracy * 100:.2f}%")

test_passengers = [
    [3, 0, 22, 1, 0],
    [1, 1, 35, 0, 0]
]

for p in test_passengers:
    res = model.predict([p])
    status = "Survived" if res[0] == 1 else "Did not survive"
    print(f"Input {p} | Prediction: {status}")
