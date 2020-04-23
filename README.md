# ML_Model_for_predict_clients_who_default_on_their_loans
Author: Parameswar rao

Date: 4 April 2020

I implimented a model using hmeq.csv data set for predicting clients who default on their loans through the current process of loan underwriting features to automate the decisionmaking process for approval of home equity lines of credit. This project is organised as follows:
(a) extract features from data set.
(b) data preprocessing and split into train and test data for a given model.
(c) Predict the output of the test cases using logistic regression algorithm.
(d) compare the output of the test cases using the decision tree algorithm.
(e) increase accuracy by using boosting technique.

hmeq.csv: data set for given model.

Result: 
1. log rig:Accuracy on test data for a given model is 0.7935943060498221
2. decision tree:Accuracy on test data for a given model is 0.8622267412303

3. Learning rate:  0.05
   Accuracy score (training): 0.809
   Accuracy score (validation): 0.798

   Learning rate:  0.1
   Accuracy score (training): 0.834
   Accuracy score (validation): 0.815

   Learning rate:  0.25
   Accuracy score (training): 0.887
   Accuracy score (validation): 0.861

   Learning rate:  0.5
   Accuracy score (training): 0.904
   Accuracy score (validation): 0.883

   Learning rate:  0.75
   Accuracy score (training): 0.906
   Accuracy score (validation): 0.886

   Learning rate:  1
   Accuracy score (training): 0.907
   Accuracy score (validation): 0.883
