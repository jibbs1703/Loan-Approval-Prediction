from train.src.model.model import  ModelInputs, ModelTrain, ModelPredict
from train.src.model.metrics import ModelMetrics
from train.src.helper.aws_services import S3Buckets
from train.src.helper.data_reader import csv_loader
from train.src.data_engineering.transformation import Transformation
import warnings
import yaml
warnings.filterwarnings('ignore')

with open('predict/src/config.yaml', 'r') as file:
    yaml_file = yaml.safe_load(file)

s3 = S3Buckets.credentials('us-east-2')
uncleaned_file = s3.read_file('jibbs-raw-datasets', 'uncleaned_loan_prediction_data.csv')
data = csv_loader(uncleaned_file)

# Transformation and Feature Engineering
transform = Transformation(data)
data = transform.run_pipeline()

# Upload Cleaned Dataframe to S3 Bucket
s3.upload_dataframe_to_s3(data, 'jibbs-cleaned-datasets', 'cleaned_loan_prediction_data.csv')

# Load Cleaned File From S3 Bucket
cleaned_file = s3.read_file('jibbs-cleaned-datasets', 'cleaned_loan_prediction_data.csv')
data = csv_loader(cleaned_file)

# Split Data into Test/Training Features and Target
model_input = ModelInputs(data)
train_features, test_features, train_target, test_target = model_input.run_pipeline()

TRAIN = input('Do You want to Train the Model (T Trains Only and F Trains and Predicts on Test Data)?: ')

if TRAIN == 'T':
    model_train = ModelTrain()
    model = model_train.model_training(train_features, train_target)

else :
    model_train = ModelTrain()
    model = model_train.model_training(train_features, train_target)
    model_pred = ModelPredict()
    test_prediction = model_pred.model_prediction(model, test_features, test_target)

    metrics = ModelMetrics(test_target, test_prediction)
    print(metrics.accuracy())
