import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn import metrics
import matplotlib.pyplot as plt
sns.set_theme(style='darkgrid')

# Loading the dataset
file_path = 'Advertising.csv'
df = pd.read_csv(file_path)

# Selecting features and target variables
features = ['TV', 'Radio', 'Newspaper']
X = df[features]
y = df['Sales']

# Adding Polynomial Features
degree = 2
poly = PolynomialFeatures(degree, include_bias=False)
X_poly = poly.fit_transform(X)

# Splitting the data into training and testing cases
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Applying Polynomial Features to both training and test sets
X_train_poly = poly.transform(X_train)
X_test_poly = poly.transform(X_test)

# Creating a Random Forest Regressor model with feature scaling
model = make_pipeline(StandardScaler(), RandomForestRegressor(random_state=42))

# Hyperparameter tuning using GridSearchCV
param_grid = {
    'randomforestregressor__n_estimators': [50, 100, 150],
    'randomforestregressor__max_depth': [None, 10, 20, 30],
    'randomforestregressor__min_samples_split': [2, 5, 10],
    'randomforestregressor__min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train_poly, y_train)

# Best hyperparameters
best_params = grid_search.best_params_
print('Best Hyperparameters:', best_params)

# Training the model with the best hyperparameters on the training set
best_model = grid_search.best_estimator_
best_model.fit(X_train_poly, y_train)

# Making predictions on the test set
y_pred = best_model.predict(X_test_poly)

# Evaluation of the model
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred, squared=False))
print('R-squared:', metrics.r2_score(y_test, y_pred))

## Residual Plot
residuals = y_test - y_pred
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_pred, y=residuals)
plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.show()

# Cross-Validation to evaluate the model on multiple splits
cv_scores = cross_val_score(best_model, X_poly, y, cv=5, scoring='neg_mean_squared_error')
cv_rmse_scores = (-cv_scores)**0.5
print('Cross-Validation RMSE Scores:', cv_rmse_scores)
print('Average Cross-Validation RMSE:', cv_rmse_scores.mean())

# Plot the actual vs predicted values
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')
plt.title('Actual vs. Predicted Sales (Random Forest)')
plt.show()
