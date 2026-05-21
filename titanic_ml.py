import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv("/Users/apple/Desktop/titanic_project_final/titanic_clean.csv")

#CONVERTING SEX TO NUMERIC VALUES, X MATRIX WITH FEATURE AND Y WITH ANSWERS
df["Sex"]= df["Sex"].map({"male": 0,"female":1})
features = ["Pclass", "Sex", "Age", "Fare"]
X = df[features]
y = df["Survived"]

#SPLITING THE DATASET INTO TRAINING AND TESTING SETS
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f" Train rows :{len(X_train)} | Test rows :{len(X_test)}")

#TRAINING THE MODEL
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)

#MAKING PREDICTIONS, COMPARING THEM TO ANSWERS AND MAKING REPORT
lr_predictions = lr.predict(X_test)
print(f" Accuracy: {accuracy_score(y_test, lr_predictions):.1%}")
print(classification_report(y_test, lr_predictions))

#PREDICTING SURVIVAL FOR A NEW PASSENGER
passenger = [[3, 0, 22, 8]]  # 3rd class male, 22 yrs, cheap fare
result = lr.predict(passenger)[0]
prob   = lr.predict_proba(passenger)[0][1]
print(f"\n3rd class male 22: {'Survived' if result==1 else 'Did not survive'} ({prob:.0%} chance)")