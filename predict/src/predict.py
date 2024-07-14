from train.src.helper.aws_services import S3Buckets
import numpy as np
import yaml

with open('predict/src/config.yaml', 'r') as file:
    yaml_file = yaml.safe_load(file)

# Load Model from S3 Bucket
s3 = S3Buckets.credentials('us-east-2')
model = s3.load_model_from_s3(yaml_file['MODEL_BUCKET'], yaml_file['MODEL_NAME'])

applicant_input = np.array([['Self-emp-not-inc','Under-Graduate', 50, 83311.0, 13.0, 0.0, 0.0,13.0]])
print(model.predict(applicant_input))


