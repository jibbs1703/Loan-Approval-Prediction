from sklearn.preprocessing import MinMaxScaler
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
        self.df['Total_Income'] = (self.df.ApplicantIncome + self.df.CoapplicantIncome)
        self.df['Payment'] = (self.df.LoanAmount / self.df.Loan_Amount_Term)
        self.df['Debt_Income'] = (self.df.LoanAmount/ self.df.ApplicantIncome)

    def drop_columns(self):
        # Make the Loan_ID column the dataframe index and Drop the Loan_ID column
        self.df.index = self.df['Loan_ID']
        self.df = self.df.drop(columns=['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_ID'])

    def encode_target(self):
        target_map = {'Y': 1, 'N': 0}
        self.df[yaml_file['TARGET']] = self.df[yaml_file['TARGET']].map(target_map)

    def scale_numeric(self):
        num_cols = yaml_file['NUMERICAL_COLUMNS']
        self.mms = MinMaxScaler().fit(self.df[num_cols])
        self.df[num_cols] = self.mms.transform(self.df[num_cols])

    def encode_categorical(self):
        # Instantiate Target Encoder and Target-Encode the Categorical Features
        cat_cols = yaml_file['CATEGORICAL_COLUMNS']
        self.enc = TargetEncoder().fit(self.df[cat_cols], self.df[yaml_file['TARGET']])
        self.df[cat_cols] = self.enc.transform(self.df[cat_cols], self.df[yaml_file['TARGET']])

    def scaler_encoder(self):
        scaler = self.mms
        encoder = self.enc
        return scaler, encoder


    def run_pipeline(self):
        self.fill_missing()
        self.create_features()
        self.drop_columns()
        self.encode_target()
        self.scale_numeric()
        self.encode_categorical()
        return self.df
