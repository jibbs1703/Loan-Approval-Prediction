from typing import Literal
from fastapi import FastAPI, Depends
from pydantic import BaseModel, computed_field, conint
from aws_services import S3Buckets
import yaml
import pandas as pd
import numpy as np

# Access Config.yaml File For Training Package to Access Saved Parameters For Model Training
with open('config.yaml', 'r') as file:
    yaml_file = yaml.safe_load(file)

# Instantiate FastAPI
app = FastAPI(title="Loan Approval Prediction", version='2.0.0')


# Description of Model Features
class Features(BaseModel):
    Gender: Literal['Male', 'Female']
    Married: Literal['Yes', 'No']
    Dependents: Literal['0', '1', '2', '3+']
    Education: Literal['Graduate', 'Not Graduate']
    Self_Employed: Literal['Yes', 'No']
    Loan_Amount_Term: int
    Credit_History: int
    Property_Area: Literal['Semiurban', 'Rural', 'Urban']
    ApplicantIncome: int
    CoapplicantIncome: int
    LoanAmount: int

    @computed_field
    @property
    def total_income(self) -> int:
        return self.ApplicantIncome + self.CoapplicantIncome

    @computed_field
    @property
    def payment(self) -> float:
        return self.LoanAmount / self.Loan_Amount_Term

    @computed_field
    @property
    def debt_income(self) -> float:
        return self.LoanAmount / self.ApplicantIncome


# Load Trained Model Only When App Starts-Up
@app.on_event("startup")
def model_encoder_scaler():
    # Load Trained Model, Encoder and Scaler from S3 Bucket
    s3 = S3Buckets.credentials('us-east-2')
    trained_model = s3.load_model_from_s3(yaml_file['MODEL_BUCKET'], yaml_file['MODEL_NAME'])
    trained_encoder = s3.load_model_from_s3(yaml_file['MODEL_BUCKET'], yaml_file['CAT_ENCODER'])
    trained_scaler = s3.load_model_from_s3(yaml_file['MODEL_BUCKET'], yaml_file['NUM_SCALER'])

    return trained_model, trained_encoder, trained_scaler


# Assign Trained Model, Encoder and Scaler to Objects
model, encoder, scaler = model_encoder_scaler()


# Create Landing Page for Application
@app.get("/")
def home():
    return "Loan Approval Prediction Application"


# Create Prediction Path
@app.post("/predict")
def predict(features: Features = Depends()):
    data = features.dict()
    data.pop('LoanAmount')
    data.pop('ApplicantIncome')
    data.pop('CoapplicantIncome')

    feature_dict = {
        'Gender': data['Gender'],
        'Married': data['Married'],
        'Dependents': data["Dependents"],
        'Education': data["Education"],
        'Self_Employed': data["Self_Employed"],
        'Loan_Amount_Term': data["Loan_Amount_Term"],
        'Credit_History': data["Credit_History"],
        'Property_Area': data["Property_Area"],
        'total_income': data["total_income"],
        'payment': data["payment"],
        'debt_income': data["debt_income"]}

    feature_df = pd.DataFrame(feature_dict, index = [0])
    feature_df.Credit_History = feature_df.Credit_History.astype('object')
    feature_df[yaml_file['NUMERICAL_COLUMNS']] = scaler.transform(feature_df[yaml_file['NUMERICAL_COLUMNS']])
    feature_df[yaml_file['CATEGORICAL_COLUMNS']] = encoder.transform(feature_df[yaml_file['CATEGORICAL_COLUMNS']])

    prediction = model.pre

    return feature_df