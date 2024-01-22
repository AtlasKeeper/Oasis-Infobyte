import pandas as pd
from joblib import load

# Load the TF-IDF vectorizer
tfidf_vectorizer = load('tfidf_vectorizer.joblib')

# Load the SVM and Random Forest models
loaded_svm_model = load('svm_model.joblib')
loaded_rf_model = load('random_forest_model.joblib')

def predict_spam(message):
    # Vectorize the input message using the TF-IDF vectorizer
    tfidf_message = tfidf_vectorizer.transform([message])

    # Predict using the loaded SVM model
    svm_prediction = loaded_svm_model.predict([message])[0]

    # Predict using the loaded Random Forest model
    rf_prediction = loaded_rf_model.predict(tfidf_message)[0]

    return svm_prediction, rf_prediction

if __name__ == "__main__":
    # Example test messages
    test_messages = [
        "Hey there! Your package has been successfully delivered. Thank you for choosing our service.",
        "URGENT: You have won a cash prize! Claim now by clicking the link below.",
        "Meeting at 3 PM in the conference room. Please be on time.",
        "Special discount on our latest products. Limited time offer! Click here to shop now.",
        "Reminder: Pay your bills before the due date to avoid late fees.",
        "You've been selected for a survey. Share your feedback and get a chance to win a gift card."
    ]

    print("Test Messages and Predictions:")
    print("===============================")

    for i, message in enumerate(test_messages, start=1):
        svm_result, rf_result = predict_spam(message)
        print(f"Test Message {i}: {message}")
        print(f"- SVM Prediction: {svm_result}")
        print(f"- Random Forest Prediction: {rf_result}")
        print("===============================")
