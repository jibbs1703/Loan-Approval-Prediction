from sklearn.preprocessing import MinMaxScaler
from train.src.helper.aws_services import S3Buckets
from category_encoders import TargetEncoder
import yaml

# Access Config.yaml File to Access Saved Parameters
with open('train/src/config.yaml', 'r') as file:
    yaml_file = yaml.safe_load(file)


class Transformation:
    def __init__(self, df):
        self.df = df

    def fill_missing(self):
        # Fill Missing Observations in Numerical Columns with the Column Mean
        num_cols = yaml_file['NON_OBJECTS']
        for col in num_cols:
            self.df[col] = self.df[col].fillna(self.df[col].mean())

        # Fill Missing Observations in Categorical Columns with the Column Mode
        cat_cols = yaml_file['OBJECTS']
        for col in cat_cols:
            self.df[col] = self.df[col].fillna(self.df[col].mode()[0])

    def create_features(self):
        self.df['total_income'] = (self.df.ApplicantIncome + self.df.CoapplicantIncome)
        self.df['payment'] = (self.df.LoanAmount / self.df.Loan_Amount_Term)
        self.df['debt_income'] = (self.df.LoanAmount/ self.df.ApplicantIncome)

    def drop_columns(self):
        # Make the Loan_ID column the dataframe index and Drop the Loan_ID column
        self.df.index = self.df['Loan_ID']
        self.df = self.df.drop(columns=['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_ID'])

    def encode_target(self):
        target_map = {'Y': 1, 'N': 0}
        self.df[yaml_file['TARGET']] = self.df[yaml_file['TARGET']].map(target_map)

    def scale_numeric(self):
        num_cols = yaml_file['NUMERICAL_COLUMNS']
        mms = MinMaxScaler()
        mms.fit(self.df[num_cols])
        self.df[num_cols] = mms.transform(self.df[num_cols])

        # Save the Fitted Encoder To S3 Bucket
        s3 = S3Buckets.credentials('us-east-2')
        s3.save_model_to_s3(mms, yaml_file['MODEL_BUCKET'], yaml_file['NUM_SCALER'])

    def encode_categorical(self):
        # Instantiate Target Encoder and Target-Encode the Categorical Features
        cat_cols = yaml_file['CATEGORICAL_COLUMNS']
        enc = TargetEncoder().fit(self.df[cat_cols], self.df[yaml_file['TARGET']])
        self.df[cat_cols] = enc.transform(self.df[cat_cols], self.df[yaml_file['TARGET']])

        # Save the Fitted Encoder To S3 Bucket
        s3 = S3Buckets.credentials('us-east-2')
        s3.save_model_to_s3(enc, yaml_file['MODEL_BUCKET'], yaml_file['CAT_ENCODER'])

    def run_pipeline(self):
        self.fill_missing()
        self.create_features()
        self.drop_columns()
        self.encode_target()
        self.scale_numeric()
        self.encode_categorical()
        return self.df
