from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from preprocess import load_and_preprocess_data

# Load and preprocess data
X_train, X_test, y_train, y_test = load_and_preprocess_data("data/churn_data.csv")

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Save model and accuracy
accuracy = accuracy_score(y_test, model.predict(X_test))
print(f"Model Accuracy: {accuracy:.2f}")

# Save model using joblib
import joblib
joblib.dump(model, "churn_model.pkl")
