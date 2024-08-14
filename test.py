test = {
  "Gender": {
    "0": 0.6912350597609562
  },
  "Married": {
    "0": 0.6291079814621773
  },
  "Dependents": {
    "0": 0.6861111111111111
  },
  "Education": {
    "0": 0.7083333333333334
  },
  "Self_Employed": {
    "0": 0.6829356790832921
  },
  "Loan_Amount_Term": {
    "0": 0.7435897435897436
  },
  "Credit_History": {
    "0": "1"
  },
  "Property_Area": {
    "0": 0.6584158419442776
  },
  "total_income": {
    "0": 0.05460167425023253
  }}

import numpy as np
print([test[key]['0'] for key in test])

user = [0.6912350597609562, 0.6291079814621773, 0.6861111111111111, 0.7083333333333334, 0.6829356790832921, 0.7435897435897436, 1, 0.6584158419442776, 0.05460167425023253]
user = np.array(user, dtype='float64')
from app.aws_services import S3Buckets

s3 = S3Buckets.credentials('us-east-2')
loaded = s3.load_model_from_s3('jibbs-models', 'loan-prediction-model.pkl')
print(loaded.predict(user.reshape(-1,1)))