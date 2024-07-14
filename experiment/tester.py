from train.src.helper.aws_services import S3Buckets
from train.src.helper.data_reader import csv_loader
from train.src.data_engineering.transformation import Transformation
from train.src.model.model import Model

s3 = S3Buckets.credentials('us-east-2')
file_from_s3 = s3.download_file('jibbs-raw-datasets', 'Loan Prediction Data.csv', 'uncleaned_loan_prediction_data.csv')
data = csv_loader(file_from_s3)
transform = Transformation(data)
data = transform.run_pipeline()
data.to_csv('cleaned_loan_prediction_data.csv')
s3.upload_file('cleaned_loan_prediction_data.csv', 'jibbs-cleaned-datasets', 'cleaned_loan_prediction_data.csv')
model = Model(data)
predicted, actual, accuracy, f1, roc_score, conf_matrix, class_report= model.run_pipeline()

print(predicted[:5])
print(actual.head())
print(f"Accuracy: {accuracy}")
print(f"F-1 Score: {f1}")
print(f"AUC-ROC Score: {roc_score}")
print(conf_matrix)
print(class_report)