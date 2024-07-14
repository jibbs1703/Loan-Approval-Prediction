from imblearn.over_sampling import SMOTENC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, f1_score, confusion_matrix
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

class Model:
    def __init__(self, df):
        self.df = df
    def target_feature_split(self):
        self.X = self.df.drop('Loan_Status', axis=1)
        self.y = self.df['Loan_Status']

    def target_balancer(self):
        cat_cols = ['Gender_Female', 'Gender_Male', 'Married_No',
       'Married_Yes', 'Dependents_0', 'Dependents_1', 'Dependents_2',
       'Dependents_3+', 'Education_Graduate', 'Education_Not Graduate',
       'Self_Employed_No', 'Self_Employed_Yes', 'Property_Area_Rural',
       'Property_Area_Semiurban', 'Property_Area_Urban', 'Credit_History_0.0',
       'Credit_History_1.0']
        smote_nc = SMOTENC(categorical_features = cat_cols, random_state=420)
        self.X_resampled, self.y_resampled = smote_nc.fit_resample(self.X, self.y)

    def train_test_split(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X_resampled, self.y_resampled, test_size=0.20, random_state=420)

    def model_training(self):
        self.algorithm = XGBClassifier(random_state = 420, n_estimators = 100,
                                       colsample_bytree = 0.7, gamma = 0.0,
                                       learning_rate = 0.2, max_depth = 12,
                                       min_child_weight = 1)
        self.algorithm.fit(self.X_train, self.y_train)
        y_train_pred = self.algorithm.predict(self.X_train)
        acc = accuracy_score(self.y_train, y_train_pred)
        roc = roc_auc_score(self.y_train, y_train_pred)
        f1 = f1_score(self.y_train, y_train_pred)

        print(f"The model scored a {acc} accuracy on the training dataset")
        print(f"The model had a {f1} F-1 Score on the training dataset")
        print(f"The model had an {roc} roc_auc_score on the training dataset")
        print(classification_report(self.y_train, y_train_pred))
        print(confusion_matrix(self.y_train, y_train_pred))

    def test_prediction(self):
        # Get the Prediction Accuracy on the Test Data
        self.y_test_pred = self.algorithm.predict(self.X_test)
        self.test_acc = accuracy_score(self.y_test, self.y_test_pred)
        self.test_roc = roc_auc_score (self.y_test, self.y_test_pred)
        self.test_f1 = f1_score(self.y_test, self.y_test_pred)
        self.conf_matrix = confusion_matrix(self.y_test, self.y_test_pred)
        self.class_report = classification_report(self.y_test, self.y_test_pred)
        print(f"The model scored a {self.test_acc} accuracy on the test dataset")
        print(f"The model had a {self.test_f1} F-1 Score on the test dataset")
        print(f"The model had an {self.test_roc} roc_auc_score on the test dataset")

    def run_pipeline(self):
        self.target_feature_split()
        self.target_balancer()
        self.train_test_split()
        self.model_training()
        self.test_prediction()
        return self.y_test_pred, self.y_test, self.test_acc, self.test_f1, self.test_roc, self.conf_matrix, self.class_report