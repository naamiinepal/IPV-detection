from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC


class InstantiateModel:
    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.model_name = args.model

    def __get__(self):
        # Pick your model.
        if self.model_name == "adaboost":
            self.model = AdaBoostClassifier()
        elif self.model_name == "logistic_regression":
            self.model = LogisticRegression()
        elif self.model_name == "svm":
            self.model = SVC()
        elif self.model_name == "nb":
            self.model = MultinomialNB()
        elif self.model_name == "random_forest":
            self.model = RandomForestClassifier()
        else:
            raise AssertionError

        # Update the parameters with ours.
        update_dict = self.args[self.model_name]
        self.model.__dict__.update(update_dict)
        self.logger.info(
            f"{self.model_name} updated with paramereters : \n {self.args[self.model_name]}!\n"
        )

        return self.model
