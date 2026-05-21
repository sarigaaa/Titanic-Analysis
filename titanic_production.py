import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# This magic line pulls your setup data directly from your first file!
from titanic_ml import X_train, X_test, y_train, y_test, features, lr_predictions

# Training the Random Forest model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)

# Compare accuracy side-by-side
print(f"Random Forest accuracy:      {accuracy_score(y_test, rf_preds):.1%}")
print(f"Logistic Regression accuracy:{accuracy_score(y_test, lr_predictions):.1%}")

# Feature importance — what actually predicted survival most?
imp = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)
print("\nFeature importance:")
print(imp)

# Plot and save the graph
imp.plot(kind="bar", color="#4A90D9", title="What predicted survival most?")
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.show()