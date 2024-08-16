import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

# Define a list of models to evaluate
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(random_state=42),
    'Support Vector Regression': SVR(kernel='rbf')
}

# Hyperparameter grids for each model
param_grids = {
    'Linear Regression': {},  # No hyperparameters to tune for Linear Regression
    'Random Forest': {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    'Gradient Boosting': {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    },
    'Support Vector Regression': {
        'C': [0.1, 1, 10],
        'epsilon': [0.01, 0.1, 0.2]
    }
}

# Load the dataset
df = pd.read_csv('../static/possum.csv')

df_cleaned = df.dropna()
# One-hot encode the 'Pop' column
df_encoded = pd.get_dummies(df_cleaned, columns=['Pop', 'sex'], drop_first=True)

# Display the first few rows of the encoded dataframe
print(df_encoded.head())

# Features and target variable
X = df_encoded.drop(columns=['hdlngth', 'case'])
y = df_encoded['hdlngth']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the scaler
joblib.dump(scaler, 'data/scaler_hd.pkl')

# Dictionary to store the results
results = {}

# Evaluate each model
best_models = {}
for name, model in models.items():
    print(f"Training {name} model with the following features:")
    print(X_train.columns.tolist())  # Print the feature columns

    if name in param_grids:
        # Use GridSearchCV for hyperparameter tuning
        grid_search = GridSearchCV(estimator=model, param_grid=param_grids[name], cv=5, n_jobs=-1, scoring='r2')
        grid_search.fit(X_train_scaled, y_train)
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test_scaled)
        best_models[name] = best_model  # Save the best model found by GridSearchCV
    else:
        # Train the model and make predictions
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        best_models[name] = model  # Save the fitted model

    # Evaluate the model
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    
    # Store the results
    results[name] = {
        'R-squared': r2,
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse
    }
    
    # Print the results for each model
    print(f"{name}:")
    print(f"  R-squared: {r2:.2f}")
    print(f"  Mean Absolute Error: {mae:.2f}")
    print(f"  Mean Squared Error: {mse:.2f}")
    print(f"  Root Mean Squared Error: {rmse:.2f}")
    print()

# Save each best model found
joblib.dump(best_models['Linear Regression'], 'data/linear_regression_hd_model.pkl')
joblib.dump(best_models['Random Forest'], 'data/random_forest_hd_model.pkl')
joblib.dump(best_models['Gradient Boosting'], 'data/gradient_boosting_hd_model.pkl')
joblib.dump(best_models['Support Vector Regression'], 'data/svr_hd_model.pkl')

# Find the best model based on R-squared
best_model_name = max(results, key=lambda name: results[name]['R-squared'])
best_model_results = results[best_model_name]

print(f"Best model: {best_model_name}")
print(f"R-squared: {best_model_results['R-squared']:.2f}")
print(f"Mean Absolute Error: {best_model_results['MAE']:.2f}")
print(f"Mean Squared Error: {best_model_results['MSE']:.2f}")
print(f"Root Mean Squared Error: {best_model_results['RMSE']:.2f}")
