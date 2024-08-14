from typing import Literal
from fastapi import FastAPI, Depends
from pydantic import BaseModel, computed_field
import yaml
import pandas as pd
import pickle

# Access Config.yaml File For Training Package to Access Saved Parameters For Model Training
with open('config.yaml', 'r') as file:
    yaml_file = yaml.safe_load(file)

# Instantiate FastAPI
app = FastAPI(title="Loan Approval Prediction Application", version='1.0.0')


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
    # Load the Model, Encoder and Scaler
    loaded_scaler = pickle.load(open(f"objects/{yaml_file['NUM_SCALER']}", 'rb'))
    loaded_encoder = pickle.load(open(f"objects/{yaml_file['CAT_ENCODER']}", 'rb'))
    loaded_model = pickle.load(open(f"objects/{yaml_file['MODEL_NAME']}", 'rb'))

    return loaded_model, loaded_encoder, loaded_scaler


# Assign Loaded Model, Encoder and Scaler to Objects
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

    data = pd.DataFrame(data, index=[0])
    data[yaml_file['NUMERICAL_COLUMNS']] = scaler.transform(data[yaml_file['NUMERICAL_COLUMNS']])
    data[yaml_file['CATEGORICAL_COLUMNS']] = encoder.transform(data[yaml_file['CATEGORICAL_COLUMNS']])

    # Get Prediction From transformed Features
    prediction = model.predict(data)

    # Create Condition of Output Returned
    if prediction == 0:
        output = 'We are sorry to decline your application'
    elif prediction == 1:
        output = 'Congratulations, you have been approved for your loan'
    else:
        output = 'Loan approval result could not be generated'

    return output