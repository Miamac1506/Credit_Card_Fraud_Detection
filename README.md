# Credit Card Fraud Detection Project
## About
The principal aim of this project is to systematically assess and contrast the performance of three machine learning algorithms—Logistic Regression, Decision Trees, and XGBoost—in detecting fraudulent activities within credit card transactions. My workflow is structured as follows:

+ Credit Card Data Acquisition: Initially, I acquire a dataset comprising credit card transactions, which includes both legitimate and fraudulent examples.

+ Data Pre-processing: This step involves cleaning and preparing the data for analysis, which may include handling missing values, encoding categorical variables, and normalizing or scaling numerical fields.

+ Data Analysis: I conduct exploratory data analysis to understand the characteristics and distribution of the data, identify potential outliers, and uncover insightful patterns that could influence the performance of the predictive models.

+ Train-Test Split: The dataset is divided into training and testing subsets to ensure the models are trained and evaluated on distinct sets of data, promoting unbiased estimation of their real-world performance.

+ Machine Learning Model Implementation: I implement three different machine learning models:

1. Logistic Regression, for its simplicity and effectiveness in binary classification tasks.
2. Decision Trees, which provide intuitive decision rules and are capable of capturing non-linear patterns.
3. XGBoost, known for its superior performance due to its ensemble approach that combines multiple weak learners into a strong predictive model.

+ Model Evaluation: Each model is rigorously evaluated using metrics such as accuracy, precision, recall, and the F1-score to determine their effectiveness in correctly identifying fraudulent transactions.

Through this structured workflow, I aim to determine the most effective technique for fraud detection in credit card transactions, balancing accuracy with computational efficiency.

## Dataset
Source: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud?resource=download

## Results
### Logistic Regression Results
+ General Performance: Shows good generalization with a smaller gap between training and testing metrics compared to the other models. However, the accuracy and other metrics are pretty low.
  
+ Precision and Recall: High precision indicates fewer false positives, which is good for minimizing disruption to legitimate transactions. However, recall is comparatively lower, meaning it misses a higher proportion of fraudulent transactions.

### Decision Trees Results
+ General Performance: Exhibits overfitting with perfect training scores and noticeably lower testing scores. This suggests that while the Decision Tree model learns the training data very well, it doesn't generalize as effectively to new data.
  
+ Precision and Recall: While precision remains high, the recall and F1-scores on the testing data are lower than those of XGBoost, indicating it does not balance detection and precision as effectively in new environments.

### XGBoost Results
+ General Performance: Similar to Decision Trees, XGBoost shows perfect training scores, but unlike Decision Trees, it maintains higher performance on the testing data. This indicates strong learning and generalization capabilities.
+ Precision and Recall: XGBoost not only achieves high precision but also maintains higher recall and F1 scores on the testing data compared to both Logistic Regression and Decision Trees. This suggests it is better at both identifying fraud accurately and capturing a higher percentage of fraudulent transactions without as many false negatives.



