# Credit-Card-Fraud-Detection-on-AWS-Sagemaker
Project Details: Fraud detection - Identify fraudulent activities using Amazon SageMaker.
o   You work for a multinational bank.
o   Over the last few months, there has been a significant uptick in the number of customers experiencing credit card fraud. 
o   You need to use ML to identify fraudulent credit card transactions before they have a larger impact on your company. 

Step 1: Problem formulation and getting started
Project selection: 
Questions to consider:
1. What is the business problem?
Over the last few months, there has been a significant uptick in the number of customers experiencing credit card fraud. 
2. What is the business goal?
To identify fraudulent credit card transactions before they have a larger impact on your company
3. What is the business metric?
To identify and stop 90% of fraudulent credit card transactions before they have a larger impact on the company.
4. Why is the business problem appropriate for machine learning?
ML is appropriate due to its scale, variability and speed. Scale: Thousands of transaction per second. Variability: Different merchants/type of transactions/locations. Speed: We need to identify the fraudulent cases quickly before any further damage can be done.
5. What type of ML should be used?
Supervised learning – Binary classification
6. Reframe the business problem as a machine learning problem:
To obtain the labelled dataset from the past few months and train, test and deploy ML (via Sagemaker) to predict future fraudulent credit card transactions. Success metric for the ML model to have at least 90% recall..  

Iteration I

Step 2: Data exploration and data preprocessing
Data preprocessing and visualization
Questions to consider:
1.     Did you have to make any assumptions about the data?
·      Pre-processing was done prior to the PCA step. 
·      Dataset was labelled correctly
·      Transaction amount assumed to be in the same currency
2.     What does exploratory data analysis and visualization tell you about the data?
○	Proportion of fraudulent cases is very small (0.173%) -> might need to do data augmentation
 
○	The transactions follow a periodic pattern -  bulk of transactions at 40000-75000 and 125000-150000
 
○	All the PCA components have relatively low correlation with each other as well as the target variable ‘Class’.
 
3.     What techniques did you use to clean and preprocess your data?
·      We might need to remove the outliers in the amount column (min amount = $0)

Step 3: Model training 
Questions to consider for model training:
4.     What percentage of the data should be training, validation, and test?
	80/10/10
5.     Did you randomize the split? If not, how might that impact the model?
	We randomized the split using the train_test_split function, using a common random seed so as to keep the splits consistent and allows us to reliably compare model performance against the same sets of training/validation/test data across multiple runs. We might increase the model bias if the dataset is ordered in a particular way.
6.     What algorithm should be used?
	LinearLearner - binary_classifier

Step 4: Model evaluation
Questions to consider for model evaluation:
1.     What metric did you choose to evaluate your model?
	Recall score, because we want to make sure the most number of correct fraud detections are being flagged out, minimize the false negatives, and false positives don’t really matter as such. 
	A false positive may inconvenience a user, while a false negative may cause a great deal of financial damage to both the customer and the bank.
2.     How did your model do on your chosen metric? 
	Recall scores:
		Model 1 (dropped ‘Time’ feature): 0.8043478260869565 
3.     What did you learn from your evaluation metric? 
      Our selected model (Model 1)  is able to predict 80.4% of the fraudulent transactions correctly. 
 
Iteration II

Step 5: Adding Feature engineering and hyperparameter optimization
Questions to consider:
1.     Which feature engineering features were used and why?
●	Resampled the dataset to create a balanced dataset to try and improve the recall score of our model. Two methods were used - Undersampling and Oversampling
●	Used t-SNE for dimensionality reduction

2.     What AWS SageMaker hyperparameter tuning did you use and what was the impact on the model?
●	Continuous Parameters were used:
○	Weight decay, 'wd': ContinuousParameter(1e-5, 1),
○	L1 regularization parameter, 'l1': ContinuousParameter(1e-5, 0.01),
○	Learning rate, 'learning_rate': ContinuousParameter(1e-4, 0.1)
●	Recall scores did not increase after hyperparameter tuning:
○	Before tuning: 0.937366148229 
○	After tuning: 0.9368044137954712

3.     What is the correlation between metric and individual hyperparameters?
●	Weight decay: Negative Correlation
 
●	L1 regularization parameter: Positive Correlation
 
●	Learning rate: Positive Correlation
 
Final Thoughts
1.     What is the original business success metric and the current model performance?
●	Original business success metric: Recall of 90%
●	Current model performance: Recall of 93.7% 
		 
2.     If you had more time what would you do?
●	Experiment and evaluate other models such as XGBoost, etc
●	Experiment with more hyperparameters and increase the range of each hyperparameter. For example, we can increase the upper bounds of l1 and learning rate to improve recall scores as l1 and learning rate has a positive correlation to the recall score.
●	Get more data
●	Do more feature engineering

3.     What are the key lessons you learned during this project?
●	Understanding the business problem and crafting a good ML problem statement is the most important step in the entire ML workflow
●	Domain knowledge will give us a good headstart in the data preprocessing phase
●	Data Visualization is very important when trying to explore the dataset
●	Data Visualization is also important when we are trying to present and convey project findings
●	Machine learning is a very iterative process (especially at the feature engineering and hyperparameter tuning steps)
●	Need to balance between time/effort and model performance. Model can always improve but at what cost?
●	Amazon Sagemaker makes it convenient for us to train, test, tune and deploy ML models on the cloud 
 

