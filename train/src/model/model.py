import pandas as pd
from imblearn.over_sampling import SMOTENC
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import yaml
import pickle
from train.src.helper.aws_services import S3Buckets

# Access Config.yaml File For Training Package to Access Saved Parameters For Model Training
with open('train/src/config.yaml', 'r') as file:
    yaml_file = yaml.safe_load(file)


class ModelInputs:
    def __init__(self, df):
        self.df = df

    def target_feature_split(self):
        X = self.df.drop(yaml_file['TARGET'], axis=1)
        y = self.df[yaml_file['TARGET']]
        return X, y

    def target_balancer(self, X, y):
        cat_cols = yaml_file['CATEGORICAL_COLUMNS']
        smote_nc = SMOTENC(categorical_features = cat_cols, random_state=420)
        X_resampled, y_resampled = smote_nc.fit_resample(X, y)
        return X_resampled, y_resampled

    def train_test_split(self, X_resampled, y_resampled, test_size=0.):
        print(X_resampled.shape)
        X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=test_size, random_state=420)
        return X_train, X_test, y_train, y_test

    def run_pipeline(self):
        X, y = self.target_feature_split()
        X, y = self.target_balancer(X, y)
        X_train, X_test, y_train, y_test = self.train_test_split(X, y, yaml_file['TEST_SIZE'])
        return X_train, X_test, y_train, y_test


class ModelTrain:
    def model_training(self, X_train, y_train):

        # Instantiate Model and Fit to Training Data
        algorithm = RandomForestClassifier(random_state = 420)
        algorithm.fit(X_train, y_train)

        # Save the Fitted Model To S3 Bucket/ Application Directory
        s3 = S3Buckets.credentials('us-east-2')
        s3.save_model_to_s3(algorithm, yaml_file['MODEL_BUCKET'], yaml_file['MODEL_NAME'])

        # To Application Directory
        filename = f"objects/{yaml_file['MODEL_NAME']}"
        pickle.dump(algorithm, open(filename, 'wb'))

        # Return the Fitted Model
        return algorithm


class ModelPredict():
    def model_prediction(self, model, X_test, y_test):

        # Get Prediction on the Test Data
        y_test_pred = model.predict(X_test)

        # Transform Predicted Values to Same Format as Actual Values
        y_test_pred = pd.Series(y_test_pred)
        y_test_pred.index = y_test.index
        return y_test_pred
