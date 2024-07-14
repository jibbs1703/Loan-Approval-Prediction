from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, f1_score, \
                            confusion_matrix, recall_score, precision_score


class ModelMetrics:
    def __init__(self, actual, predicted):
        """
        The ModelMetrics class takes in the actual values and predicted results of the model when it is initialized.
        The actual and predicted values are made available globally to other methods in the class.

        :param actual: The actual values from the real-world data.
        :param predicted: The values predicted by the model.
        """
        self.actual = actual
        self.predicted = predicted

    def accuracy(self):
        """
        This method of the ModelMetrics class calculates the accuracy score of the predictions made by the
         model by comparing the actual values and predicted values.

        :return: The accuracy score of the predicted values vs the actual values
        """
        acc = accuracy_score(self.actual, self.predicted)
        return f" Model Accuracy : {acc}"

    def f1_score(self):
        """
        This method of the ModelMetrics class calculates the F1 score of the predictions made by the
         model by comparing the actual values and predicted values.

        :return: The F1 score of the predicted values vs the actual values
        """
        f1 = f1_score(self.actual, self.predicted)
        return f" Model f1-score : {f1}"

    def recall(self):
        """
        This method of the ModelMetrics class calculates the recall score of the predictions made by the
         model by comparing the actual values and predicted values.

        :return: The recall score of the predicted values vs the actual values
        """
        rec = recall_score(self.actual, self.predicted)
        return f" Model Recall Score : {rec}"

    def precision(self):
        """
        This method of the ModelMetrics class calculates the precision score of the predictions made by the
         model by comparing the actual values and predicted values.

        :return: The precision score of the predicted values vs the actual values
        """
        pre = precision_score(self.actual, self.predicted)
        return f" Model Precision Score : {pre}"

    def conf_matrix(self):
        """
        This method of the ModelMetrics class presents the confusion matrix of the model's predicted
        values compared to the actual values.

        :return: A confusion matrix of the model predictions.
        """
        matrix = confusion_matrix(self.actual, self.predicted)
        return matrix

    def class_report(self):
        """
        This method of the ModelMetrics class presents the classification matrix of the model's predicted
        values compared to the actual values. The classification report details the model's prediction
        accuracy, recall, precision, and F1-scores.

        :return: A classification report of the model predictions.
        """
        report = classification_report(self.actual, self.predicted)
        return report

    def roc_auc(self):
        """
        This method of the ModelMetrics class calculates the roc_auc_score of the predictions made by the
         model by comparing the actual values and predicted values.

        :return: The roc_auc_score of the predicted values vs the actual values
        """
        auc = roc_auc_score(self.actual, self.predicted)
        return f" Model ROC_AOC Score : {auc}"