def _trail_param_retrive(trial, dict, keyword):
    """An internal function. Return a trial suggest using dict params of estimator and
    one keyword of it. Based on the keyword, it will return an
    Optuna.trial.suggest. The return will be trial.suggest_int(keyword, min(dict[keyword]), max(dict[keyword]))
    Example : _trail_param_retrive(trial, {
            "max_depth": [2, 3],
            "min_child_weight": [0.1, 0.9],
            "gamma": [1, 9],
             }, "gamma") --> will be trail.suggest_int for gamma using [1,9]
    Parameters
    ----------
    trial: Optuna trial
        A trial is a process of evaluating an objective function.
        For more info, visit
        https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html
    dict: dict
        A dictionary of estimator params.
        e.g., {
            "max_depth": [2, 3],
            "min_child_weight": [0.1, 0.9],
            "gamma": [1, 9],
             }
    Keyword: str
        A keyword of estimator key params. e.g., "gamma"
    """
    if isinstance(dict[keyword][0], str) or dict[keyword][0] is None:
        return trial.suggest_categorical(keyword, dict[keyword])
    if isinstance(dict[keyword][0], int):
        if len(dict[keyword]) >= 2:
            if isinstance(dict[keyword][1], int):
                return trial.suggest_int(
                    keyword, min(dict[keyword]), max(dict[keyword])
                )
        else:
            return trial.suggest_float(keyword, min(dict[keyword]), max(dict[keyword]))
    if isinstance(dict[keyword][0], float):
        return trial.suggest_float(keyword, min(dict[keyword]), max(dict[keyword]))


def _trail_params_retrive(trial, dict):
    """An internal function. Return a trial suggests using dict params of estimator.
    Example : _trail_param_retrive(trial, {
            "eval_metric": ["auc"],
            "max_depth": [2, 3],
            "min_child_weight": [0.1, 0.9],
            "gamma": [1, 9],
            "booster": ["gbtree", "gblinear", "dart"],
             }, "gamma") --> will return params where
             parmas = {
                "eval_metric": trial.suggest_categorical("eval_metric", ["auc"]),
                "max_depth": trial.suggest_int("max_depth", 2,3),
                "min_child_weight": trial.suggest_float("min_child_weight", 0.1, 0.9),
                "booster": trial.suggest_categorical("booster", ["gbtree", "gblinear", "dart"]),
             }
    Parameters
    ----------
    trial: Optuna trial
        A trial is a process of evaluating an objective function.
        For more info, visit
        https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html
    dict: dict
        A dictionary of estimator params.
        e.g., {
            "eval_metric": ["auc"],
            "max_depth": [2, 3],
            "min_child_weight": [0.1, 0.9],
            "gamma": [1, 9],
            "booster": ["gbtree", "gblinear", "dart"],
             }
    """
    params = {}
    for keyword in dict.keys():
        if keyword not in params.keys():
            if isinstance(dict[keyword][0], str) or dict[keyword][0] is None:
                params[keyword] = trial.suggest_categorical(keyword, dict[keyword])
            if isinstance(dict[keyword][0], int):
                if len(dict[keyword]) >= 2:
                    if isinstance(dict[keyword][1], int):
                        params[keyword] = trial.suggest_int(
                            keyword, min(dict[keyword]), max(dict[keyword])
                        )
                else:
                    params[keyword] = trial.suggest_float(
                        keyword, min(dict[keyword]), max(dict[keyword])
                    )
            if isinstance(dict[keyword][0], float):
                params[keyword] = trial.suggest_float(
                    keyword, min(dict[keyword]), max(dict[keyword])
                )
    return params
