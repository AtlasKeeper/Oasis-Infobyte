import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from joblib import dump, load

# Load the cleaned dataset
cleaned_file_path = "cleaned_data/cleaned_spam_data.csv"
spam_data = pd.read_csv(cleaned_file_path)

# Split the data into training and testing sets
train_data, test_data, train_labels, test_labels = train_test_split(
    spam_data['v2'], spam_data['v1'], test_size=0.2, random_state=42
)

# Convert text data to numerical features using TF-IDF
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_train = tfidf_vectorizer.fit_transform(train_data)
tfidf_test = tfidf_vectorizer.transform(test_data)

# Save the TF-IDF vectorizer
dump(tfidf_vectorizer, 'tfidf_vectorizer.joblib')

# Define SVM pipeline with feature scaling
svm_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english')),
    ('scaler', StandardScaler(with_mean=False)),  # Feature scaling for SVM
    ('svm', SVC())
])

# Define hyperparameter grids
svm_parameters = {
    'svm__C': [0.1, 1, 10],
    'svm__kernel': ['linear', 'rbf', 'poly'],
    'svm__gamma': ['scale', 'auto']
}

rf_parameters = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Train SVM with feature scaling using GridSearchCV
svm_grid_search = GridSearchCV(svm_pipeline, svm_parameters, cv=5)
svm_grid_search.fit(train_data, train_labels)

# Save SVM model
dump(svm_grid_search, 'svm_model.joblib')

# Train Random Forest using GridSearchCV
rf_classifier = RandomForestClassifier(random_state=42)
rf_grid_search = GridSearchCV(rf_classifier, rf_parameters, cv=5)
rf_grid_search.fit(tfidf_train, train_labels)

# Save Random Forest model
dump(rf_grid_search, 'random_forest_model.joblib')

# Load models
loaded_svm_model = load('svm_model.joblib')
loaded_rf_model = load('random_forest_model.joblib')

# Make predictions using loaded models
svm_loaded_predictions = loaded_svm_model.predict(test_data)
rf_loaded_predictions = loaded_rf_model.predict(tfidf_test)

# Evaluate loaded models
print("\nLoaded Model Performance Summary:")
print("=========================================")

print("\nLoaded Support Vector Machine (SVM):")
print(f"  Accuracy: {accuracy_score(test_labels, svm_loaded_predictions)}")
print("  Classification Report:")
print(classification_report(test_labels, svm_loaded_predictions))

print("\nLoaded Random Forest:")
print(f"  Accuracy: {accuracy_score(test_labels, rf_loaded_predictions)}")
print("  Classification Report:")
print(classification_report(test_labels, rf_loaded_predictions))
print("=========================================")
