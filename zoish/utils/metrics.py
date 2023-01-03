import inspect
import os

import numpy as np
from sklearn.metrics import *

# for load Environment Variables
# True there will not be default args for metric


def f1_plus_tp(y_true, y_pred):
    """Return f1_score+True Possitive
    Parameters
    ----------
    y_true : Pandas DataFrame or Pandas Series
        True values
    y_pred : Pandas DataFrame Pandas Series
        predicted values.
    """
    f1 = f1_score(y_true, y_pred)
    _, _, _, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return f1 + tp


def f1_plus_tn(y_true, y_pred):
    """Return f1_score+True Negative
    Parameters
    ----------
    y_true : Pandas DataFrame or Pandas Series
        True values
    y_pred : Pandas DataFrame Pandas Series
        predicted values.
    """

    f1 = f1_score(y_true, y_pred)
    tn, _, _, _ = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return f1 + tn


def specificity(y_true, y_pred):
    """Return Specificity
    Parameters
    ----------
    y_true : Pandas DataFrame or Pandas Series
        True values
    y_pred : Pandas DataFrame Pandas Series
        predicted values.
    """

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return tn / (tn + fp)


def tn_score(y_true, y_pred):
    """Return Specificity
    Parameters
    ----------
    y_true : Pandas DataFrame or Pandas Series
        True values
    y_pred : Pandas DataFrame Pandas Series
        predicted values.
    """

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return tn


def tn(y_true, y_pred):
    """Return Specificity
    Parameters
    ----------
    y_true : Pandas DataFrame or Pandas Series
        True values
    y_pred : Pandas DataFrame Pandas Series
        predicted values.
    """

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return tn


def tp_score(y_true, y_pred):
    """Return Specificity
    Parameters
    ----------
    y_true : Pandas DataFrame or Pandas Series
        True values
    y_pred : Pandas DataFrame Pandas Series
        predicted values.
    """

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return tp


def tp(y_true, y_pred):
    """Return Specificity
    Parameters
    ----------
    y_true : Pandas DataFrame or Pandas Series
        True values
    y_pred : Pandas DataFrame Pandas Series
        predicted values.
    """

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return tp


def roc_plus_f1(y_true, y_pred):
    """Return ROC + f1_score
    Parameters
    ----------
    y_true : Pandas DataFrame or Pandas Series
        True values
    y_pred : Pandas DataFrame Pandas Series
        predicted values.
    """
    f1 = f1_score(y_true, y_pred)
    roc = roc_curve(y_true, y_pred)
    return np.sum(f1 + roc)


def auc_plus_f1(y_true, y_pred):
    """Return AUC + f1_score
    Parameters
    ----------
    y_true : Pandas DataFrame or Pandas Series
        True values
    y_pred : Pandas DataFrame Pandas Series
        predicted values.
    """
    f1 = f1_score(y_true, y_pred)
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    auc_result = auc(fpr, tpr)
    return f1 + auc_result


def det_curve_ret(y_true, y_pred):
    """Return det_curve
    Parameters
    ----------
    y_true : Pandas DataFrame or Pandas Series
        True values
    y_pred : Pandas DataFrame Pandas Series
        predicted values.
    """
    fpr, fnr, thresholds = det_curve(y_true, y_pred)
    return np.sum(fpr) + np.sum(fnr)


def precision_recall_curve_ret(y_true, y_pred):
    """Return precision_recall_curve
    Parameters
    ----------
    y_true : Pandas DataFrame or Pandas Series
        True values
    y_pred : Pandas DataFrame Pandas Series
        predicted values.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    return np.sum(precision) + np.sum(recall)


def precision_recall_fscore_support_ret(y_true, y_pred):
    """Return precision_recall_fscore_support
    Parameters
    ----------
    y_true : Pandas DataFrame or Pandas Series
        True values
    y_pred : Pandas DataFrame Pandas Series
        predicted values.
    """
    output = precision_recall_fscore_support(y_true, y_pred)
    return np.sum(output)


def roc_curve_ret(y_true, y_pred):
    """Return roc_curve
    Parameters
    ----------
    y_true : Pandas DataFrame or Pandas Series
        True values
    y_pred : Pandas DataFrame Pandas Series
        predicted values.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    return np.sum(tpr) - np.sum(fpr)


class CalcMetrics:
    """Return roc_curve
    Parameters
    ----------
    y_true : Pandas DataFrame or Pandas Series
        True values
    y_pred : Pandas DataFrame Pandas Series
        predicted values.
    metric : Str
        Name of a metric function, e.g., f1_score.
    d : dict
        A mapping dictionary that maps a metric function's string name
         to its representative function, e.g., "accuracy_score": accuracy_score.
    """

    def __init__(
        self,
        y_true,
        y_pred,
        metric,
        d={
            "accuracy_score": accuracy_score,  # normal
            "auc": auc,  # normal
            "precision_recall_curve": precision_recall_curve,  # normal
            "balanced_accuracy_score": balanced_accuracy_score,  # normal
            "cohen_kappa_score": cohen_kappa_score,  # normal
            "dcg_score": dcg_score,  # normal
            "det_curve": det_curve_ret,  # normal minimize
            "f1_score": f1_score,  # normal
            "fbeta_score": fbeta_score,  # normal
            "hamming_loss": hamming_loss,  # normal minimize
            "fbeta_score": fbeta_score,  # normal
            "jaccard_score": jaccard_score,  # normal
            "matthews_corrcoef": matthews_corrcoef,  # normal
            "ndcg_score": ndcg_score,  # normal
            "precision_score": precision_score,  # normal
            "recall_score": recall_score,  # normal
            "recall": recall_score,  # normal
            "roc_auc_score": roc_auc_score,  # normal
            "roc_curve": roc_curve_ret,  # normal
            "top_k_accuracy_score": top_k_accuracy_score,  # normal
            "zero_one_loss": zero_one_loss,  # normal minimize
            # customs
            "tn": tn,  # custom
            "tp": tp,  # custom
            "tn_score": tn_score,  # custom
            "tp_score": tp_score,  # custom
            "f1_plus_tp": f1_plus_tp,  # custom
            "f1_plus_tn": f1_plus_tn,  # custom
            "specificity": specificity,  # custom
            "roc_plus_f1": roc_plus_f1,  # custom
            "auc_plus_f1": auc_plus_f1,  # custom
            "precision_recall_curve_ret": precision_recall_curve_ret,  # custom
            "precision_recall_fscore_support": precision_recall_fscore_support_ret,  # custom
            # regression
            "explained_variance_score": explained_variance_score,
            "max_error": max_error,
            "mean_absolute_error": mean_absolute_error,
            "mean_squared_log_error": mean_squared_log_error,
            "mean_absolute_percentage_error": mean_absolute_percentage_error,
            "mean_squared_log_error": mean_squared_log_error,
            "median_absolute_error": median_absolute_error,
            "mean_absolute_percentage_error": mean_absolute_percentage_error,
            "r2_score": r2_score,
            "mean_poisson_deviance": mean_poisson_deviance,
            "mean_gamma_deviance": mean_gamma_deviance,
            "mean_tweedie_deviance": mean_tweedie_deviance,
            "d2_tweedie_score": d2_tweedie_score,
            "mean_pinball_loss": mean_pinball_loss,
            "d2_pinball_score": d2_pinball_score,
            "d2_absolute_error_score": d2_absolute_error_score,
        },
        change_default_args_of_metric=False,
        *args,
        **kwargs,
    ):
        self.y_true = y_true
        self.y_pred = y_pred
        self.metric = metric
        self.d = d
        # set change_default_args_of_metric based on debug mode
        self.change_default_args_of_metric = change_default_args_of_metric
        self.args = args
        self.kwargs = kwargs

    def resolve_name(self):
        """
        Return a metric function.
        """
        if self.d[self.metric]:
            return self.d[self.metric]

    def get_func_args(self):
        """
        Return a dictionary of arguments of a metric function.
        """
        args_of_func = inspect.signature(self.resolve_name()).parameters
        return args_of_func

    def get_func_default_args(self):
        """
        Return a dictionary of all default arguments of a metric function.
        """
        args_of_func = self.get_func_args()
        d = {}
        for name, value in args_of_func.items():
            value_str = str(value)
            if (
                "=" in value_str
                and str(value) != "y_true"
                and str(value) != "y_score"
                and str(value) != "y_pred"
                and str(value) != "y_prob"
            ):
                d[name] = value
        func_default_args = d
        return func_default_args

    def get_transformed_default_args(self):
        """
        Return a fixed dictionary of all default arguments of a metric function.
        """
        func_default_args = self.get_func_default_args()
        d = {}
        for name, value in func_default_args.items():
            value_str = str(value)
            if "None" in value_str:
                d[name] = None
            elif "True" in value_str:
                d[name] = True
            elif "False" in value_str:
                d[name] = False
            elif "'" in value_str and "=" in value_str:
                start = end = "'"
                d[name] = value_str.split(start)[1].split(end)[0]
            elif "'" not in value_str and "=" in value_str:
                start = end = "="
                d[name] = value_str.split(start)[1].split(end)[0]
            for k, v in func_default_args.items():
                try:
                    start = end = "="
                    str_var = str(v).split(start)[1].split(end)[0]
                    float_str = float(str_var)
                    d[k] = float_str
                except ValueError:
                    print(f"{v} Not a float")
                try:
                    start = end = "="
                    str_var = str(v).split(start)[1].split(end)[0]
                    int_str = int(str_var)
                    d[k] = int_str
                except ValueError:
                    print(f"{v} Not a int")

        transformed_defualt_args = d
        return transformed_defualt_args

    def asign_default(self):
        """
        Provide a way for users to change default values of default
        arguments of a metric function.
        """
        metric = self.resolve_name()
        transformed_defualt_args = self.get_transformed_default_args()
        if self.change_default_args_of_metric:
            if len(transformed_defualt_args) > 0:
                ans = input(
                    f"Do you want to change defualt arguments for {str(metric.__name__)} Y/N? : "
                )
                if ans == "Y" or ans == "y" or ans == "yes":
                    self.change_default_args_of_metric = True
                else:
                    self.change_default_args_of_metric = False
                if self.change_default_args_of_metric:
                    for t, v in transformed_defualt_args.items():
                        value_input = input(
                            f"Set value for {t} of {str(metric.__name__)} : "
                        )
                        if value_input == "":
                            transformed_defualt_args[t] = v
                        elif "True" == value_input or "true" == value_input:
                            transformed_defualt_args[t] = True
                        elif "False" == value_input or "false" == value_input:
                            transformed_defualt_args[t] = False
                        elif "None" == value_input or "none" == value_input:
                            transformed_defualt_args[t] = None
                        else:
                            transformed_defualt_args[t] = str(value_input)

                    for t, v in transformed_defualt_args.items():
                        if not isinstance(v, bool):
                            try:
                                float_v = float(v)
                                transformed_defualt_args[t] = float_v
                            except Exception as e:
                                print(f"The type of {v} is not float ! {e}")
                            try:
                                int_v = int(v)
                                transformed_defualt_args[t] = int_v
                            except Exception as e:
                                print(f"The type of {v} is not integer ! {e}!")

        return transformed_defualt_args

    def get_default_params_if_any(self):
        """
        Return a default params of  metric function.
        """
        asign_default = self.asign_default()
        return asign_default

    def get_func_string_if_any(self):
        """
        Return a metrics function if any.
        """
        metric = self.resolve_name()
        func = str(metric.__name__)
        return func

    def get_metric_func(self):
        """
        Return a metric function and its default assigned arguments.
        """
        asign_default = self.asign_default()
        metric = self.resolve_name()
        func = str(metric.__name__)
        f_str = func + "(" + "self.y_true,self.y_pred, **asign_default" + ")"
        print(f"default assigned arguments are :{asign_default}")
        print(f_str)
        print(eval(f_str))
        return eval(f_str)

    def calc_make_scorer(self, metric):
        """
        Calculate function body of a function using make_scorer
        see:
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.make_scorer.html
        Parameters:
        -----------
        metric: str
            String representation of a metric function.
        """
        metrics_list_for_maximizing = [
            "accuracy_score",
            "auc",
            "precision_recall_curve",
            "balanced_accuracy_score",
            "cohen_kappa_score",
            "dcg_score",
            "f1_score",
            "fbeta_score",
            "fbeta_score",
            "jaccard_score",
            "matthews_corrcoef",
            "ndcg_score",
            "precision_score",
            "recall_score",
            "roc_auc_score",
            "roc_curve",
            "top_k_accuracy_score",
            # customs
            "tn",
            "tn_score",
            "tp",
            "tp_score",
            "f1_plus_tp",
            "f1_plus_tn",
            "specificity",
            "roc_plus_f1",
            "auc_plus_f1",
            "precision_recall_curve",
            "precision_recall_fscore_support",
            "max_error",
            "r2_score",
        ]
        metrics_list_for_minimizing = [
            "det_curve",
            "hamming_loss" "zero_one_loss" "explained_variance_score",
            "mean_absolute_error" "mean_squared_log_error",
            "mean_absolute_percentage_error",
            "mean_squared_log_error",
            "median_absolute_error",
            "mean_absolute_percentage_error",
            "mean_poisson_deviance",
            "mean_gamma_deviance",
            "mean_tweedie_deviance",
            "d2_tweedie_score",
            "mean_pinball_loss",
            "d2_pinball_score",
            "d2_absolute_error_score",
        ]

        if metric in metrics_list_for_maximizing:
            return make_scorer(self.d[metric], greater_is_better=True)
        if metric in metrics_list_for_minimizing:
            return make_scorer(self.d[metric], greater_is_better=True)

    def get_simple_metric(self, metric, y_true, y_pred, params=None):
        """
        Return a simple metric of y_true, y_pred
        Parameters:
        -----------
        metric: str
            String representation of a metric function.
        y_true : Pandas DataFrame or Pandas Series
            True values
        y_pred : Pandas DataFrame Pandas Series
            predicted values.
        """
        if params is None:
            f_str = metric + "(" + "y_true,y_pred" + ")"
            return eval(f_str)
        if isinstance(params, dict) and len(params) == 0:
            f_str = metric + "(" + "y_true,y_pred" + ")"
            return eval(f_str)
        if isinstance(params, dict) and len(params) > 0:
            f_str = metric + "(" + "y_true,y_pred,**params" + ")"
            return eval(f_str)
        return None


if __name__ == "__main__":
    # test metrics
    calcmetric = CalcMetrics(
        y_true=np.array([0, 0, 1, 0]),
        y_pred=np.array([0, 1, 1, 0]),
        metric="d2_absolute_error_score",
    )
    calcmetric.get_metric_func()
    func = calcmetric.calc_make_scorer("d2_absolute_error_score")
    print(func)
