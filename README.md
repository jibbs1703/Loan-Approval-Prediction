# LOAN APPROVAL PREDICTION MODEL

## Overview 
The use of machine learning models for loan approval predictions is a prime application of classification 
modelling in the loan application process. The classification model can learn patterns from ground truth
to predict new loan application outcomes. The models can provide quicker and more data-driven decisions 
by automating the loan application process by predicting if a credit/loan applicant should be approved or 
declined based on a set of features.

In this project, a loan approval classification model is developed to evaluate a credit/loan applicant’s 
ability and willingness to repay a loan. The model approves or declines an applicant's application once 
the applicant's information is entered, streamlining the loan approval process, making the loan underwriting
process more efficient for financial institutions while improving the user experience for loan applicants.

The model deployment is done using a FastAPI (backend) and Reflex(frontend), allowing for the application to 
be deployed end-to-end using python. The project also leverages cloud resources (AWS EC2 for computing and 
AWS S3 Bucket for data warehousing). To ensure wide use of this model, it is deployed in a docker container,
allowing the application to be hosted on any server, regardless of the inherent operating system.


## Dependencies

The [requirements.txt](requirements.txt) file contains the Python libraries needed to run the notebook 
and the model presented in this project.The dependencies can be installed using:
```
pip install -r requirements.txt
```


## Data Management

The data used to train the model was obtained from [Kaggle](https://www.kaggle.com/) and was collected from 
real-life loan applications with the personal details anonymized to protect the privacy of the individuals 
involved. To ensure availability of the data for training the model, AWS S3 is used to warehouse the data once
it is extracted from the source. The training data is then accessed from the S3 Bucket it is stored in using the
IAM user access obtained for the S3 resource. The dataset consists of 614 records and 13 features,described in 
the [data description](experiment/data-description.txt).

## Feature Engineering

As part of the model development, feature engineering is performed to select the best features for the model to
give the best prediction regarding an applicant's loan application. 

### Assigning Appropriate Datatypes and Dealing with Missing Values 

The first step after importing the dataset was to ensure that all variables, whether feature or target, conformed to
the right datatype and had no missing elements. only four features in the dataset had missing observation and this 
was dealt with by assigning modal values to fill in the missing values in the categorical features and assigning mean
values to fill in the missing values in the numeric features. To achieve correct datatypes, the variable 
"Credit History" was transformed into categorical variable according to the data description for the dataset.

### Encoding Categorical Variables and Scaling Numeric Variables

To ensure all features exist as numeric values for input into the model, the categorical variables in the dataset were
encoded into numbers using Target encoding. The numeric variables are scaled using the MinMax Scaler, putting all numeric
variables between zero and one. 

### Feature Creation and Dropped Columns

As part of the feature engineering process, three new features were created to boost model performance. Total Income was
created from the Applicant Income and Coapplicant Income, Payment Amount was created from the Total Loan Amount and the 
Loan Amount Term while Debt to Income Ratio was created from Loan Amount and the Applicant Income.

After several experiments on the data, the Applicant Income, Coapplicant Income and Loan Amount were dropped from the 
model features, leaving a total of 13 features for the model training. The features were dropped due to high correlation
with other features in the dataset. 


## Model Development

### Model Training

A test-train-split was used on the dataset to test the predictive ability of the model. The model is trained on 80% of the 
data and tested on the remaining 20%. This helps to check how the model performs on data not passed through it during 
training, similar to how it is expected to perform in the real world on unseen data. An 80:20 test-train split was used for
training and testing this model.

Several models were trained on the data. The baseline model (Logistic Regression Model) was used to compare the performance
of other models trained on the data. After the experiments were completed, the Random Forest Model was selected as the best
model as it scored the highest F1 and ROC-AUC Scores. 

### Model Evaluation

The loan classification model was evaluated to assess its performance on predicting the approval status of an applicant's
loan application. The goal was to ensure that the model not only performs well on the training data but also generalizes
effectively to unseen data. 

The precision score of the best model was 88%, while also maintaining a high recall of 82%. High precision means
the model predicts an applicant's approval status correctly with limited irrelevant predictions while a high recall, 
on the other hand, indicates the model is able to successfully predict an applicant's approval status correctly without
assigning the wrong approval status. 

|              Model               | Accuracy | F1-Score | Precision | Recall | AUC Score |
|:--------------------------------:|:--------:|:--------:|:---------:|:------:|:---------:|
| Base Model (Logistic Regression) |   0.75   |   0.83   |   0.71    |  0.98  |   0.71    |
|      Logistic Regression CV      |   0.77   |   0.83   |   0.74    |  0.96  |   0.75    |
|      RandomForestClassifier      |   0.83   |   0.85   |   0.88    |  0.82  |   0.83    |
|          XGBClassifier           |   0.80   |   0.82   |   0.85    |  0.80  |   0.80    |


## Model Deployment

The application is containerized and deployed on an AWS EC2 instance, providing availability and reliability of the 
application for getting loan approval predictions. The prediction API endpoints are built in the backend using FastAPI
connected to the User Interface, which is built using Reflex. 

