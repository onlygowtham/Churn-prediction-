import joblib
from sklearn.metrics import classification_report
from preprocess import load_and_preprocess_data

# Load pre-trained model
model = joblib.load("churn_model.pkl")

# Load test data
_, X_test, _, y_test = load_and_preprocess_data("data/churn_data.csv")

# Evaluate model
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
