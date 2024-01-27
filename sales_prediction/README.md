<h1>Documentation: model.py</h1>

<h2>Overview</h2>
    <p>The <code>model.py</code> script implements a Random Forest Regression model for predicting sales based on advertising channels.</p>

<h2>Dependencies</h2>
    <ul>
        <li>Pandas</li>
        <li>Seaborn</li>
        <li>Scikit-learn</li>
        <li>Matplotlib</li>
    </ul>

<h2>Usage</h2>
    <p>To use this script, ensure that the 'Advertising.csv' dataset is available in the same directory.</p>
    <pre><code>python model.py</code></pre>

<h2>Functionality</h2>
    <ol>
        <li>Loading the dataset</li>
        <li>Feature selection and target variable definition</li>
        <li>Adding polynomial features</li>
        <li>Splitting the dataset into training and testing sets</li>
        <li>Creating a Random Forest Regressor model with hyperparameter tuning</li>
        <li>Training the model with the best hyperparameters</li>
        <li>Evaluating the model using various metrics</li>
        <li>Visualizing the results (residual plot, actual vs. predicted values)</li>
    </ol>

<h2>Output</h2>
    <p>The script provides model evaluation metrics such as mean absolute error, mean squared error, root mean squared error, and R-squared. It also visualizes the results using matplotlib plots.</p>

<h2>Example</h2>
    <p>Example usage of the script:</p>
    <pre><code>python model.py</code></pre>

<h2>Author</h2>
    <p>Author: AtlasKeeper</p>

<h2>Version</h2>
    <p>Version: 1.0</p>
</body>
</html>
