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
]

# list of Integer parameters for estimator. This is need for beaking down parameters for trial
# suggestion

Integer_list = ["max_depth", "depth", "n_jobs", "verbose"]
Categorical_list = ["booster", "objective", "boosting_type", "bootstrap_type"]


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
}


XGBOOST_CLASSIFICATION_PARAMS_DEFAULT = {
    "objective": "binary:logistic",
    "use_label_encoder": False,
    "base_score": None,
    "booster": None,
    "callbacks": None,
    "colsample_bylevel": None,
    "colsample_bynode": None,
    "colsample_bytree": None,
    "early_stopping_rounds": None,
    "enable_categorical": False,
    "eval_metric": None,
    "gamma": None,
    "gpu_id": None,
    "grow_policy": None,
    "importance_type": None,
    "interaction_constraints": None,
    "learning_rate": None,
    "max_bin": None,
    "max_cat_to_onehot": None,
    "max_delta_step": None,
    "max_depth": None,
    "max_leaves": None,
    "min_child_weight": None,
    "missing": None,
    "monotone_constraints": None,
    "n_estimators": 100,
    "n_jobs": None,
    "num_parallel_tree": None,
    "predictor": None,
    "random_state": None,
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
    "callbacks": None,
    "colsample_bylevel": None,
    "colsample_bynode": None,
    "colsample_bytree": None,
    "early_stopping_rounds": None,
    "enable_categorical": False,
    "eval_metric": None,
    "gamma": None,
    "gpu_id": None,
    "grow_policy": None,
    "importance_type": None,
    "interaction_constraints": None,
    "learning_rate": None,
    "max_bin": None,
    "max_cat_to_onehot": None,
    "max_delta_step": None,
    "max_depth": None,
    "max_leaves": None,
    "min_child_weight": None,
    "missing": None,
    "monotone_constraints": None,
    "n_estimators": 100,
    "n_jobs": None,
    "num_parallel_tree": None,
    "predictor": None,
    "random_state": None,
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
    "random_state": None,
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
    "random_state": None,
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
    "random_state": None,
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
    "random_state": None,
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
    "class_weight": None,
    "min_split_gain": 0.0,
    "min_child_weight": 0.001,
    "min_child_samples": 20,
    "subsample": 1.0,
    "subsample_freq": 0,
    "colsample_bytree": 1.0,
    "reg_alpha": 0.0,
    "reg_lambda": 0.0,
    "random_state": None,
    "n_jobs": -1,
    "silent": "warn",
    "importance_type": "split",
}
