# list of supported models

SUPPORTED_MODELS = [
    "XGBRegressor",
    "XGBClassifier",
    "RandomForestClassifier",
    "RandomForestRegressor",
    "CatBoostClassifier",
    "CatBoostRegressor",
    "BalancedRandomForestClassifier",
    "LGBMClassifier",
    "LGBMRegressor",
    "XGBSEKaplanNeighbors",
    "XGBSEDebiasedBCE",
    "XGBSEBootstrapEstimator",
]

# list of Integer parameters for the estimator. This is needed for breaking down parameters for trial
# suggestion

Integer_list = [
    "iterations",
    "penalties_coefficient",
    "l2_leaf_reg",
    "random_strength",
    "rsm",
    "depth",
    "border_count",
    "classes_count",
    "sparse_features_conflict_fraction",
    "best_model_min_trees",
    "model_shrink_rate",
    "min_data_in_leaf",
    "leaf_estimation_iterations",
    "max_leaves",
    "n_estimators",
    "min_samples_split",
    "min_samples_leaf",
    "num_leaves",
    "max_depth",
    "subsample_for_bin",
    "min_child_samples",
    "n_jobs",
    # TODO double check all these bellow
    "random_state",
    "max_leaf_nodes",
    "verbosity",
    "num_parallel_tree",
    "min_child_weight",
    "max_leaves",
    "max_bin",
]

# list of Categorical parameters for the estimator. This is needed for breaking down parameters for trial
# suggestion

Categorical_list = [
    "nan_mode",
    "eval_metric",
    "sampling_frequency",
    "leaf_estimation_method",
    "grow_policy",
    "boosting_type",
    "model_shrink_mode",
    "feature_border_type",
    "auto_class_weights",
    "leaf_estimation_backtracking",
    "loss_function",
    "score_function",
    "task_type",
    "bootstrap_type",
    "objective",
    "criterion",
    "max_features",
    "sampling_strategy",
    "silent",
    "importance_type",
    # TODO check all bellow
    "class_weight",
    "tree_method",
    "sampling_method",
    "predictor",
    "grow_policy",
    "eval_metric",
    "booster",
    # insert boolean hereafter
    "force_unit_auto_pair_weights",
    "boost_from_average",
    "use_best_model",
    "force_unit_auto_pair_weights",
    "boost_from_average",
    "use_best_model",
    "posterior_sampling",
    "use_label_encoder",
    "enable_categorical",
    "oob_score",
    "warm_start",
    "bootstrap",
    "oob_score",
    "warm_start",
    "validate_parameters",
    "class_weights",
]


# Catboost

CATBOOST_CLASSIFICATION_PARAMS_DEFAULT = {
    "nan_mode": "Min",
    "eval_metric": "Logloss",
    "iterations": 1000,
    "sampling_frequency": "PerTree",
    "leaf_estimation_method": "Newton",
    "grow_policy": "SymmetricTree",
    "penalties_coefficient": 1,
    "boosting_type": "Plain",
    "model_shrink_mode": "Constant",
    "feature_border_type": "GreedyLogSum",
    "bayesian_matrix_reg": 0.10000000149011612,
    "force_unit_auto_pair_weights": False,
    "l2_leaf_reg": 3,
    "random_strength": 1,
    "rsm": 1,
    "boost_from_average": False,
    "model_size_reg": 0.5,
    "pool_metainfo_options": {"tags": {}},
    "subsample": 1,
    "use_best_model": False,
    "class_names": [0, 1],
    "random_seed": 0,
    "depth": 6,
    "posterior_sampling": False,
    "border_count": 254,
    "classes_count": 0,
    "auto_class_weights": "None",
    "sparse_features_conflict_fraction": 0,
    "leaf_estimation_backtracking": "AnyImprovement",
    "best_model_min_trees": 1,
    "model_shrink_rate": 0,
    "min_data_in_leaf": 1,
    "loss_function": "Logloss",
    "learning_rate": 0.0010720000136643648,
    "score_function": "Cosine",
    "task_type": "CPU",
    "leaf_estimation_iterations": 10,
    "bootstrap_type": "MVS",
    "max_leaves": 64,
    "scale_pos_weight": None,
    "class_weights": None,
    "eta": None,
    "n_estimators": None,
}


CATBOOST_REGRESSION_PARAMS_DEFAULT = {
    "nan_mode": "Min",
    "eval_metric": "RMSE",
    "iterations": 1000,
    "sampling_frequency": "PerTree",
    "leaf_estimation_method": "Newton",
    "grow_policy": "SymmetricTree",
    "penalties_coefficient": 1,
    "boosting_type": "Plain",
    "model_shrink_mode": "Constant",
    "feature_border_type": "GreedyLogSum",
    "bayesian_matrix_reg": 0.10000000149011612,
    "force_unit_auto_pair_weights": False,
    "l2_leaf_reg": 3,
    "random_strength": 1,
    "rsm": 1,
    "boost_from_average": True,
    "model_size_reg": 0.5,
    "pool_metainfo_options": {"tags": {}},
    "subsample": 1,
    "use_best_model": False,
    "random_seed": 0,
    "depth": 6,
    "posterior_sampling": False,
    "border_count": 254,
    "classes_count": 0,
    "auto_class_weights": "None",
    "sparse_features_conflict_fraction": 0,
    "leaf_estimation_backtracking": "AnyImprovement",
    "best_model_min_trees": 1,
    "model_shrink_rate": 0,
    "min_data_in_leaf": 1,
    "loss_function": "RMSE",
    "learning_rate": 0.01635199971497059,
    "score_function": "Cosine",
    "task_type": "CPU",
    "leaf_estimation_iterations": 1,
    "bootstrap_type": "MVS",
    "max_leaves": 64,
    "eta": None,
    "n_estimators": None,
}


XGBOOST_CLASSIFICATION_PARAMS_DEFAULT = {
    "objective": "binary:logistic",
    "base_score": None,
    "booster": None,
    "colsample_bylevel": None,
    "colsample_bynode": None,
    "colsample_bytree": None,
    "enable_categorical": False,
    "eval_metric": None,
    "gamma": None,
    "grow_policy": None,
    "learning_rate": None,
    "max_bin": None,
    "max_delta_step": None,
    "max_depth": None,
    "max_leaves": None,
    "min_child_weight": None,
    "n_estimators": 100,
    "n_jobs": None,
    "num_parallel_tree": None,
    "predictor": None,
    "random_state": 0,
    "reg_alpha": None,
    "reg_lambda": None,
    "sampling_method": None,
    "scale_pos_weight": None,
    "subsample": None,
    "tree_method": None,
    "validate_parameters": None,
    "verbosity": None,
}


XGBOOST_REGRESSION_PARAMS_DEFAULT = {
    "objective": "reg:squarederror",
    "base_score": None,
    "booster": None,
    "colsample_bylevel": None,
    "colsample_bynode": None,
    "colsample_bytree": None,
    "enable_categorical": False,
    "eval_metric": None,
    "gamma": None,
    "grow_policy": None,
    "learning_rate": None,
    "max_bin": None,
    "max_delta_step": None,
    "max_depth": None,
    "max_leaves": None,
    "min_child_weight": None,
    "n_estimators": 100,
    "n_jobs": None,
    "num_parallel_tree": None,
    "predictor": None,
    "random_state": 0,
    "reg_alpha": None,
    "reg_lambda": None,
    "sampling_method": None,
    "scale_pos_weight": None,
    "subsample": None,
    "tree_method": None,
    "validate_parameters": None,
    "verbosity": None,
}


RANDOMFOREST_CLASSIFICATION_PARAMS_DEFAULT = {
    "n_estimators": 100,
    "criterion": "gini",
    "max_depth": None,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "min_weight_fraction_leaf": 0.0,
    "max_features": "sqrt",
    "max_leaf_nodes": None,
    "min_impurity_decrease": 0.0,
    "bootstrap": True,
    "oob_score": False,
    "n_jobs": None,
    "random_state": 0,
    "verbose": 0,
    "warm_start": False,
    "class_weight": None,
    "ccp_alpha": 0.0,
    "max_samples": None,
}


RANDOMFOREST_REGRESSION_PARAMS_DEFAULT = {
    "n_estimators": 100,
    "criterion": "gini",
    "max_depth": None,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "min_weight_fraction_leaf": 0.0,
    "max_features": "sqrt",
    "max_leaf_nodes": None,
    "min_impurity_decrease": 0.0,
    "bootstrap": True,
    "oob_score": False,
    "n_jobs": None,
    "random_state": 0,
    "verbose": 0,
    "warm_start": False,
    "ccp_alpha": 0.0,
    "max_samples": None,
}

BLF_CLASSIFICATION_PARAMS_DEFAULT = {
    "n_estimators": 100,
    "criterion": "gini",
    "max_depth": None,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "min_weight_fraction_leaf": 0.0,
    "max_features": "sqrt",
    "max_leaf_nodes": None,
    "min_impurity_decrease": 0.0,
    "bootstrap": True,
    "oob_score": False,
    "sampling_strategy": "auto",
    "replacement": False,
    "n_jobs": None,
    "random_state": 0,
    "verbose": 0,
    "warm_start": False,
    "class_weight": None,
    "ccp_alpha": 0.0,
    "max_samples": None,
}

LGB_CLASSIFICATION_PARAMS_DEFAULT = {
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "max_depth": -1,
    "learning_rate": 0.1,
    "n_estimators": 100,
    "subsample_for_bin": 200000,
    "objective": None,
    "class_weight": None,
    "min_split_gain": 0.0,
    "min_child_weight": 0.001,
    "min_child_samples": 20,
    "subsample": 1.0,
    "subsample_freq": 0,
    "colsample_bytree": 1.0,
    "reg_alpha": 0.0,
    "reg_lambda": 0.0,
    "random_state": 0,
    "n_jobs": -1,
    "silent": "warn",
    "importance_type": "split",
}

LGB_REGRESSION_PARAMS_DEFAULT = {
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "max_depth": -1,
    "learning_rate": 0.1,
    "n_estimators": 100,
    "subsample_for_bin": 200000,
    "objective": None,
    "min_split_gain": 0.0,
    "min_child_weight": 0.001,
    "min_child_samples": 20,
    "subsample": 1.0,
    "subsample_freq": 0,
    "colsample_bytree": 1.0,
    "reg_alpha": 0.0,
    "reg_lambda": 0.0,
    "random_state": 0,
    "n_jobs": -1,
    "silent": "warn",
    "importance_type": "split",
}
