import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data(file_path):
    # Load dataset
    data = pd.read_csv(file_path)
    
    # Select features and target
    X = data.iloc[:, 1:-1]  # All columns except CustomerID and Exited
    y = data.iloc[:, -1]    # 'Exited' column is the target
    
    # One-hot encode categorical features
    X = pd.get_dummies(X, drop_first=True)
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Standardize numerical features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test
