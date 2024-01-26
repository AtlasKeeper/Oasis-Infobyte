<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body>

<h1>Spam Classification Project</h1>

<p>Welcome to the Spam Classification Project! This project aims to classify text messages as spam or non-spam (ham) using machine learning techniques. It includes scripts for training models, making predictions, and evaluating model performance.</p>

<h2>Table of Contents</h2>

<ul>
<li><a href="#overview">Overview</a></li>
<li><a href="#installation">Installation</a></li>
<li><a href="#usage">Usage</a></li>
<li><a href="#test">Test</a></li>
<li><a href="#models">Models</a></li>
<li><a href="#contributing">Contributing</a></li>
<li><a href="#license">License</a></li>
</ul>

<h2 id="overview">Overview</h2>

<p>Spam emails and messages are a significant issue in today's digital communication. This project addresses this problem by implementing machine learning algorithms to identify and filter out spam messages. The project uses a dataset of labeled text messages, where each message is categorized as spam or ham (non-spam). Two popular machine learning algorithms, Support Vector Machine (SVM) and Random Forest, are utilized for classification tasks.</p>

<h2 id="installation">Installation</h2>

<ol>
<li>Clone the repository:</li>
</ol>

<pre><code>git clone https://github.com/your_username/spam-classification.git
</code></pre>

<ol start="2">
<li>Install the required dependencies:</li>
</ol>

<pre><code>pip install pandas scikit-learn joblib
</code></pre>

<ol start="3">
<li>Ensure Python 3.x is installed on your system.</li>
</ol>

<h2 id="usage">Usage</h2>

<p>The project provides two main scripts for training models and making predictions:</p>

<ul>
<li><strong>Training Models:</strong> Use the <code>train.py</code> script to train SVM and Random Forest models. The script preprocesses the data, performs feature extraction using TF-IDF, and trains the models using GridSearchCV for hyperparameter tuning.</li>
</ul>

<pre><code>python train.py
</code></pre>

<ul>
<li><strong>Making Predictions:</strong> Use the <code>test.py</code> script to make predictions on test messages. The script loads the trained models and TF-IDF vectorizer, allowing you to input messages for classification.</li>
</ul>

<pre><code>python test.py
</code></pre>

<h2 id="test">Test</h2>

<p>The <code>test.py</code> script includes a set of test messages to evaluate model performance. You can modify the test messages or add new ones to assess the models' accuracy.</p>

<h2 id="models">Models</h2>

<p>The trained SVM and Random Forest models, along with the TF-IDF vectorizer, are saved as joblib files (<code>svm_model.joblib</code>, <code>random_forest_model.joblib</code>, <code>tfidf_vectorizer.joblib</code>). These files are loaded during prediction to classify input messages.</p>

<h2 id="contributing">Contributing</h2>

<p>Contributions are welcome! If you find any issues or have suggestions for improvement, please feel free to open an issue or submit a pull request on GitHub.</p>

<h2 id="license">License</h2>

<p>This project is licensed under the MIT License - see the <a href="LICENSE">LICENSE</a> file for details.</p>

</body>
</html>
