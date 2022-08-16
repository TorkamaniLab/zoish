import logging

import fasttreeshap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedKFold

from zoish.model_conf import (
    BLF_CLASSIFICATION_PARAMS_DEFAULT,
    CATBOOST_CLASSIFICATION_PARAMS_DEFAULT,
    CATBOOST_REGRESSION_PARAMS_DEFAULT,
    LGB_CLASSIFICATION_PARAMS_DEFAULT,
    LGB_REGRESSION_PARAMS_DEFAULT,
    RANDOMFOREST_CLASSIFICATION_PARAMS_DEFAULT,
    RANDOMFOREST_REGRESSION_PARAMS_DEFAULT,
    SUPPORTED_MODELS,
    XGBOOST_CLASSIFICATION_PARAMS_DEFAULT,
    XGBOOST_REGRESSION_PARAMS_DEFAULT,
)
from zoish.utils.helper_funcs import _calc_best_estimator_random_search


class RandomizedSearchCVShapFeatureSelector(BaseEstimator, TransformerMixin):
    """
        Feature Selector class using shap values and Randomized Search. It is extended from scikit-learn
        BaseEstimator and TransformerMixin.
    ...

    Attributes
    ----------
    verbose : int
        Controls the verbosity across all objects: the higher, the more messages.
    random_state : int
        Random number seed.
    logging_basicConfig : object
        Setting Logging process. Visit https://docs.python.org/3/library/logging.html
    n_features : int
        The number of features seen during:term:`fit`. Only defined if the
        underlying estimator exposes such an attribute when fitted.
    list_of_obligatory_features_that_must_be_in_model : [str]
        A list of strings (columns names of feature set pandas data frame)
        that should be among selected features. No matter if they have high or
        low shap values, they will be selected at the end of the feature selection
        step.
    list_of_features_to_drop_before_any_selection :  [str]
        A list of strings (columns names of feature set pandas data frame)
        you want to exclude should be dropped before the selection process starts features.
        For example, it is a good idea to exclude ``id`` and ``targets`` or ``class labels. ``
        from feature space before selection starts.
    estimator: object
        An unfitted estimator. For now, only tree-based estimators. Supported
        methods are, "XGBRegressor",
        ``XGBClassifier``, ``RandomForestClassifier``,``RandomForestRegressor``,
        ``CatBoostClassifier``,``CatBoostRegressor``,
        ``BalancedRandomForestClassifier``,
        ``LGBMClassifier``, and ``LGBMRegressor``.
    estimator_params: dict
        Parameters passed to find the best estimator using optimization
        method. For CATBOOST_CLASSIFICATION_PARAMS_DEFAULT are :     "nan_mode": "Min",
        "eval_metric","iterations","sampling_frequency","leaf_estimation_method",
        "grow_policy","penalties_coefficient","boosting_type","model_shrink_mode",
        "feature_border_type","bayesian_matrix_reg","force_unit_auto_pair_weights",
        "l2_leaf_reg","random_strength","rsm","boost_from_average","model_size_reg",
        "pool_metainfo_options","subsample","use_best_model","class_names",
        "random_seed","depth","posterior_sampling","border_count",
        "classes_count","auto_class_weights","sparse_features_conflict_fraction",
        "leaf_estimation_backtracking","best_model_min_trees","model_shrink_rate",
        "min_data_in_leaf","loss_function","learning_rate","score_function",
        "task_type","leaf_estimation_iterations","bootstrap_type","max_leaves"

        For CATBOOST_REGRESSION_PARAMS_DEFAULT are :
        "nan_mode","eval_metric","iterations","sampling_frequency","leaf_estimation_method",
        "grow_policy","penalties_coefficient","boosting_type","model_shrink_mode",
        "feature_border_type","bayesian_matrix_reg","force_unit_auto_pair_weights",
        "l2_leaf_reg","random_strength","rsm","boost_from_average","model_size_reg",
        "pool_metainfo_options","subsample","use_best_model","random_seed","depth",
        "posterior_sampling","border_count","classes_count","auto_class_weights",
        "sparse_features_conflict_fraction","leaf_estimation_backtracking",
        "best_model_min_trees","model_shrink_rate","min_data_in_leaf",
        "loss_function","learning_rate","score_function","task_type",
        "leaf_estimation_iterations","bootstrap_type","max_leaves"

        For XGBOOST_CLASSIFICATION_PARAMS_DEFAULT are :
        "objective","use_label_encoder","base_score","booster",
        "callbacks","colsample_bylevel","colsample_bynode","colsample_bytree",
        "early_stopping_rounds","enable_categorical","eval_metric","gamma",
        "gpu_id","grow_policy","importance_type","interaction_constraints",
        "learning_rate","max_bin","max_cat_to_onehot","max_delta_step",
        "max_depth","max_leaves","min_child_weight","missing","monotone_constraints",
        "n_estimators","n_jobs","num_parallel_tree","predictor","random_state",
        "reg_alpha","reg_lambda","sampling_method","scale_pos_weight","subsample",
        "tree_method","validate_parameters","verbosity"

        For XGBOOST_REGRESSION_PARAMS_DEFAULT are :
        "objective","base_score","booster","callbacks","colsample_bylevel","colsample_bynode",
        "colsample_bytree","early_stopping_rounds","enable_categorical","eval_metric",
        "gamma","gpu_id","grow_policy","importance_type","interaction_constraints",
        "learning_rate","max_bin","max_cat_to_onehot","max_delta_step","max_depth",
        "max_leaves","min_child_weight","missing","monotone_constraints","n_estimators",
        "n_jobs","num_parallel_tree","predictor","random_state","reg_alpha","reg_lambda",
        "sampling_method","scale_pos_weight","subsample","tree_method","validate_parameters",
        "verbosity"

        For RANDOMFOREST_CLASSIFICATION_PARAMS_DEFAULT are :
        "n_estimators","criterion","max_depth","min_samples_split",
        "min_samples_leaf","min_weight_fraction_leaf","max_features",
        "max_leaf_nodes","min_impurity_decrease","bootstrap","oob_score",
        "n_jobs","random_state","verbose","warm_start","class_weight",
        "ccp_alpha","max_samples"

        For RANDOMFOREST_REGRESSION_PARAMS_DEFAULT are :
        "n_estimators","criterion","max_depth","min_samples_split",
        "min_samples_leaf","min_weight_fraction_leaf","max_features",
        "max_leaf_nodes","min_impurity_decrease","bootstrap","oob_score",
        "n_jobs","random_state","verbose","warm_start","ccp_alpha","max_samples"

        For BLF_CLASSIFICATION_PARAMS_DEFAULT are :
        "n_estimators","criterion"","max_depth","min_samples_split","min_samples_leaf",
        "min_weight_fraction_leaf","max_features","max_leaf_nodes","min_impurity_decrease",
        "bootstrap","oob_score","sampling_strategy","replacement","n_jobs","random_state",
        "verbose","warm_start","class_weight","ccp_alpha","max_samples"

        For LGB_CLASSIFICATION_PARAMS_DEFAULT are:
        "boosting_type","num_leaves","max_depth","learning_rate","n_estimators",
        "subsample_for_bin","objective","class_weight","min_split_gain","min_child_weight",
        "min_child_samples","subsample","subsample_freq","colsample_bytree","reg_alpha",
        "reg_lambda","random_state","n_jobs","silent","importance_type"

        For LGB_REGRESSION_PARAMS_DEFAULT are:
        "boosting_type","num_leaves","max_depth","learning_rate",
        "n_estimators","subsample_for_bin","objective","class_weight",
        "min_split_gain","min_child_weight","min_child_samples","subsample",
        "subsample_freq","colsample_bytree","reg_alpha","reg_lambda","random_state",
        "n_jobs","silent","importance_type"

        https://shap-lrjball.readthedocs.io/en/docs_update/generated/shap.TreeExplainer.html

    model_output : str
            "raw", "probability", "log_loss", or model method name
            What output of the model should be explained. If "raw" then we explain the raw output of the
            trees, which varies by model. For regression models "raw" is the standard output, for binary
            classification in XGBoost this is the log odds ratio. If model_output is the name of a supported
            prediction method on the model object then we explain the output of that model method name.
            For example model_output="predict_proba" explains the result of calling model.predict_proba.
            If "probability" then we explain the output of the model transformed into probability space
            (note that this means the SHAP values now sum to the probability output of the model). If "logloss"
            then we explain the log base e of the model loss function, so that the SHAP values sum up to the
            log loss of the model for each sample. This is helpful for breaking down model performance by feature.
            Currently the probability and logloss options are only supported when feature_dependence="independent".
            For more info visit : https://github.com/linkedin/FastTreeSHAP/blob/master/fasttreeshap/explainers/_tree.py

    feature_perturbation : str
            "interventional" (default) or "tree_path_dependent" (default when data=None)
            Since SHAP values rely on conditional expectations we need to decide how to handle correlated
            (or otherwise dependent) input features. The "interventional" approach breaks the dependencies between
            features according to the rules dictated by causal inference (Janzing et al. 2019). Note that the
            "interventional" option requires a background dataset and its runtime scales linearly with the size
            of the background dataset you use. Anywhere from 100 to 1000 random background samples are good
            sizes to use. The "tree_path_dependent" approach is to just follow the trees and use the number
            of training examples that went down each leaf to represent the background distribution. This approach
            does not require a background dataset and so is used by default when no background dataset is provided.
            For more info visit : https://github.com/linkedin/FastTreeSHAP/blob/master/fasttreeshap/explainers/_tree.py

    algorithm : str
            "auto" (default), "v0", "v1" or "v2"
            The "v0" algorithm refers to TreeSHAP algorithm in SHAP package (https://github.com/slundberg/shap).
            The "v1" and "v2" algorithms refer to Fast TreeSHAP v1 algorithm and Fast TreeSHAP v2 algorithm
            proposed in paper https://arxiv.org/abs/2109.09847 (Jilei 2021). In practice, Fast TreeSHAP v1 is 1.5x
            faster than TreeSHAP while keeping the memory cost unchanged, and Fast TreeSHAP v2 is 2.5x faster than
            TreeSHAP at the cost of a slightly higher memory usage. The default value of algorithm is "auto",
            which automatically chooses the most appropriate algorithm to use. Specifically, we always prefer
            "v1" over "v0", and we prefer "v2" over "v1" when the number of samples to be explained is sufficiently
            large, and the memory constraint is also satisfied.
            For more info visit : https://github.com/linkedin/FastTreeSHAP/blob/master/fasttreeshap/explainers/_tree.py

    shap_n_jobs : int
            (default), or a positive integer
            Number of parallel threads used to run Fast TreeSHAP. The default value of n_jobs is -1, which utilizes
            all available cores in parallel computing (Setting OMP_NUM_THREADS is unnecessary since n_jobs will
            overwrite this parameter).
            For more info visit : https://github.com/linkedin/FastTreeSHAP/blob/master/fasttreeshap/explainers/_tree.py


    random_search_n_jobs : int
            (default), or a positive integer

    memory_tolerance : int
            (default), or a positive number
            Upper limit of memory allocation (in GB) to run Fast TreeSHAP v2. The default value of memory_tolerance is -1,
            which allocates a maximum of 0.25 * total memory of the machine to run Fast TreeSHAP v2.
            For more info visit : https://github.com/linkedin/FastTreeSHAP/blob/master/fasttreeshap/explainers/_tree.py
    feature_names : [str]
            Feature names.
    approximate : bool
            Run fast, but only roughly approximate the Tree SHAP values. This runs a method
            previously proposed by Saabas which only considers a single feature ordering. Take care
            since this does not have the consistency guarantees of Shapley values and places too
            much weight on lower splits in the tree.
            For more info visit : https://github.com/linkedin/FastTreeSHAP/blob/master/fasttreeshap/explainers/_tree.py
    shortcut: False (default) or True
            Whether to use the C++ version of TreeSHAP embedded in XGBoost, LightGBM and CatBoost packages directly
            when computing SHAP values for XGBoost, LightGBM and CatBoost models, and when computing SHAP interaction
            values for XGBoost models. Current version of FastTreeSHAP package supports XGBoost and LightGBM models,
            and its support to CatBoost model is working in progress (shortcut is automatically set to be True for
            CatBoost model).
            For more info visit : https://github.com/linkedin/FastTreeSHAP/blob/master/fasttreeshap/explainers/_tree.py
    plot_shap_summary : bool
        Set True if you want to see shap summary plot of
        selected features. (default ``False``)
    save_shap_summary_plot : bool
        Set True if you want to save a shap summary plot with data_time
        label root directory of your project. (default ``False``)
    path_to_save_plot = str
        Path for saving summary plot.
    shap_fig = object
        matplotlib pyplot figure object.
        For more info visit : https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.figure.html
    test_size : float or int
        If float, it should be between 0.0 and 1.0 and represent the proportion
        of the dataset to include in the train split during estimating the best estimator
        by optimization method. If int represents the
        absolute number of train samples. If None, the value is automatically
        set to the complement of the test size.
    performance_metric : str
        Measurement of performance for classification and
        regression estimator during hyperparameter optimization while
        estimating best estimator. Classification-supported measurments are
        f1, f1_score, acc, accuracy_score, pr, precision_score,
        recall, recall_score, roc, roc_auc_score, roc_auc,
        tp, true positive, tn, true negative. Regression supported
        measurements are r2, r2_score, explained_variance_score,
        max_error, mean_absolute_error, mean_squared_error,
        median_absolute_error, and mean_absolute_percentage_error.
    cv : int, cross-validation generator or an iterable, default=None
        For more information visit
        https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html

    Methods
    -------
    fit(X, y)
        Fit the feature selection estimator by best parameters extracted
        from optimization methods.
    transform(X)
        Transform the data, and apply the transform to data to be ready for feature selection
        estimator.
    get_feature_importance_df()
        Return a Pandas data frame with two columns. One is the name of features and another
        one is the mean of shap values sorted increasingly.
    get_fig_object()
        Return a figure object related to shap summary plot.
    """

    def __init__(
        self,
        # general argument setting
        verbose=1,
        random_state=0,
        logging_basicConfig=logging.basicConfig(
            level=logging.INFO,
            filemode="w",
            format="%(name)s - %(levelname)s - %(message)s",
        ),
        # general argument setting
        n_features=5,
        list_of_obligatory_features_that_must_be_in_model=[],
        list_of_features_to_drop_before_any_selection=[],
        # shap argument setting
        estimator=None,
        estimator_params=None,
        # shap arguments
        model_output="raw",
        feature_perturbation="interventional",
        algorithm="auto",
        shap_n_jobs=-1,
        random_search_n_jobs=1,
        memory_tolerance=-1,
        feature_names=None,
        approximate=False,
        shortcut=False,
        plot_shap_summary=False,
        save_shap_summary_plot=True,
        # for save shap plot
        path_to_save_plot="./summary.png",
        shap_fig=plt.figure(),
        # grid params
        cv=StratifiedKFold(n_splits=3, shuffle=True),
        performance_metric="f1",
        n_iter=10,
    ):
        """
            Parameters
            ----------
        verbose : int
            Controls the verbosity across all objects: the higher, the more messages.
        random_state : int
            Random number seed.
        logging_basicConfig : object
            Setting Logging process. Visit https://docs.python.org/3/library/logging.html
        n_features : int
            The number of features seen during:term:`fit`. Only defined if the
            underlying estimator exposes such an attribute when fitted.
        list_of_obligatory_features_that_must_be_in_model : [str]
            A list of strings (columns names of feature set pandas data frame)
            that should be among selected features. No matter if they have high or
            low shap values, they will be selected at the end of the feature selection
            step.
        list_of_features_to_drop_before_any_selection :  [str]
            A list of strings (columns names of feature set pandas data frame)
            you want to exclude should be dropped before the selection process starts features.
            For example, it is a good idea to exclude ``id`` and ``targets`` or ``class labels. ``
            from feature space before selection starts.
        estimator: object
            An unfitted estimator. For now, only tree-based estimators. Supported
            methods are, "XGBRegressor",
            ``XGBClassifier``, ``RandomForestClassifier``,``RandomForestRegressor``,
            ``CatBoostClassifier``,``CatBoostRegressor``,
            ``BalancedRandomForestClassifier``,
            ``LGBMClassifier``, and ``LGBMRegressor``.
        estimator_params: dict
            Parameters passed to find the best estimator using optimization
            method. For CATBOOST_CLASSIFICATION_PARAMS_DEFAULT are :     "nan_mode": "Min",
            "eval_metric","iterations","sampling_frequency","leaf_estimation_method",
            "grow_policy","penalties_coefficient","boosting_type","model_shrink_mode",
            "feature_border_type","bayesian_matrix_reg","force_unit_auto_pair_weights",
            "l2_leaf_reg","random_strength","rsm","boost_from_average","model_size_reg",
            "pool_metainfo_options","subsample","use_best_model","class_names",
            "random_seed","depth","posterior_sampling","border_count",
            "classes_count","auto_class_weights","sparse_features_conflict_fraction",
            "leaf_estimation_backtracking","best_model_min_trees","model_shrink_rate",
            "min_data_in_leaf","loss_function","learning_rate","score_function",
            "task_type","leaf_estimation_iterations","bootstrap_type","max_leaves"

            For CATBOOST_REGRESSION_PARAMS_DEFAULT are :
            "nan_mode","eval_metric","iterations","sampling_frequency","leaf_estimation_method",
            "grow_policy","penalties_coefficient","boosting_type","model_shrink_mode",
            "feature_border_type","bayesian_matrix_reg","force_unit_auto_pair_weights",
            "l2_leaf_reg","random_strength","rsm","boost_from_average","model_size_reg",
            "pool_metainfo_options","subsample","use_best_model","random_seed","depth",
            "posterior_sampling","border_count","classes_count","auto_class_weights",
            "sparse_features_conflict_fraction","leaf_estimation_backtracking",
            "best_model_min_trees","model_shrink_rate","min_data_in_leaf",
            "loss_function","learning_rate","score_function","task_type",
            "leaf_estimation_iterations","bootstrap_type","max_leaves"

            For XGBOOST_CLASSIFICATION_PARAMS_DEFAULT are :
            "objective","use_label_encoder","base_score","booster",
            "callbacks","colsample_bylevel","colsample_bynode","colsample_bytree",
            "early_stopping_rounds","enable_categorical","eval_metric","gamma",
            "gpu_id","grow_policy","importance_type","interaction_constraints",
            "learning_rate","max_bin","max_cat_to_onehot","max_delta_step",
            "max_depth","max_leaves","min_child_weight","missing","monotone_constraints",
            "n_estimators","n_jobs","num_parallel_tree","predictor","random_state",
            "reg_alpha","reg_lambda","sampling_method","scale_pos_weight","subsample",
            "tree_method","validate_parameters","verbosity"

            For XGBOOST_REGRESSION_PARAMS_DEFAULT are :
            "objective","base_score","booster","callbacks","colsample_bylevel","colsample_bynode",
            "colsample_bytree","early_stopping_rounds","enable_categorical","eval_metric",
            "gamma","gpu_id","grow_policy","importance_type","interaction_constraints",
            "learning_rate","max_bin","max_cat_to_onehot","max_delta_step","max_depth",
            "max_leaves","min_child_weight","missing","monotone_constraints","n_estimators",
            "n_jobs","num_parallel_tree","predictor","random_state","reg_alpha","reg_lambda",
            "sampling_method","scale_pos_weight","subsample","tree_method","validate_parameters",
            "verbosity"

            For RANDOMFOREST_CLASSIFICATION_PARAMS_DEFAULT are :
            "n_estimators","criterion","max_depth","min_samples_split",
            "min_samples_leaf","min_weight_fraction_leaf","max_features",
            "max_leaf_nodes","min_impurity_decrease","bootstrap","oob_score",
            "n_jobs","random_state","verbose","warm_start","class_weight",
            "ccp_alpha","max_samples"

            For RANDOMFOREST_REGRESSION_PARAMS_DEFAULT are :
            "n_estimators","criterion","max_depth","min_samples_split",
            "min_samples_leaf","min_weight_fraction_leaf","max_features",
            "max_leaf_nodes","min_impurity_decrease","bootstrap","oob_score",
            "n_jobs","random_state","verbose","warm_start","ccp_alpha","max_samples"

            For BLF_CLASSIFICATION_PARAMS_DEFAULT are :
            "n_estimators","criterion"","max_depth","min_samples_split","min_samples_leaf",
            "min_weight_fraction_leaf","max_features","max_leaf_nodes","min_impurity_decrease",
            "bootstrap","oob_score","sampling_strategy","replacement","n_jobs","random_state",
            "verbose","warm_start","class_weight","ccp_alpha","max_samples"

            For LGB_CLASSIFICATION_PARAMS_DEFAULT are:
            "boosting_type","num_leaves","max_depth","learning_rate","n_estimators",
            "subsample_for_bin","objective","class_weight","min_split_gain","min_child_weight",
            "min_child_samples","subsample","subsample_freq","colsample_bytree","reg_alpha",
            "reg_lambda","random_state","n_jobs","silent","importance_type"

            For LGB_REGRESSION_PARAMS_DEFAULT are:
            "boosting_type","num_leaves","max_depth","learning_rate",
            "n_estimators","subsample_for_bin","objective","class_weight",
            "min_split_gain","min_child_weight","min_child_samples","subsample",
            "subsample_freq","colsample_bytree","reg_alpha","reg_lambda","random_state",
            "n_jobs","silent","importance_type"

            https://shap-lrjball.readthedocs.io/en/docs_update/generated/shap.TreeExplainer.html

        model_output : str
                "raw", "probability", "log_loss", or model method name
                What output of the model should be explained. If "raw" then we explain the raw output of the
                trees, which varies by model. For regression models "raw" is the standard output, for binary
                classification in XGBoost this is the log odds ratio. If model_output is the name of a supported
                prediction method on the model object then we explain the output of that model method name.
                For example model_output="predict_proba" explains the result of calling model.predict_proba.
                If "probability" then we explain the output of the model transformed into probability space
                (note that this means the SHAP values now sum to the probability output of the model). If "logloss"
                then we explain the log base e of the model loss function, so that the SHAP values sum up to the
                log loss of the model for each sample. This is helpful for breaking down model performance by feature.
                Currently the probability and logloss options are only supported when feature_dependence="independent".
                For more info visit : https://github.com/linkedin/FastTreeSHAP/blob/master/fasttreeshap/explainers/_tree.py

        feature_perturbation : str
                "interventional" (default) or "tree_path_dependent" (default when data=None)
                Since SHAP values rely on conditional expectations we need to decide how to handle correlated
                (or otherwise dependent) input features. The "interventional" approach breaks the dependencies between
                features according to the rules dictated by causal inference (Janzing et al. 2019). Note that the
                "interventional" option requires a background dataset and its runtime scales linearly with the size
                of the background dataset you use. Anywhere from 100 to 1000 random background samples are good
                sizes to use. The "tree_path_dependent" approach is to just follow the trees and use the number
                of training examples that went down each leaf to represent the background distribution. This approach
                does not require a background dataset and so is used by default when no background dataset is provided.
                For more info visit : https://github.com/linkedin/FastTreeSHAP/blob/master/fasttreeshap/explainers/_tree.py

        algorithm : str
                "auto" (default), "v0", "v1" or "v2"
                The "v0" algorithm refers to TreeSHAP algorithm in SHAP package (https://github.com/slundberg/shap).
                The "v1" and "v2" algorithms refer to Fast TreeSHAP v1 algorithm and Fast TreeSHAP v2 algorithm
                proposed in paper https://arxiv.org/abs/2109.09847 (Jilei 2021). In practice, Fast TreeSHAP v1 is 1.5x
                faster than TreeSHAP while keeping the memory cost unchanged, and Fast TreeSHAP v2 is 2.5x faster than
                TreeSHAP at the cost of a slightly higher memory usage. The default value of algorithm is "auto",
                which automatically chooses the most appropriate algorithm to use. Specifically, we always prefer
                "v1" over "v0", and we prefer "v2" over "v1" when the number of samples to be explained is sufficiently
                large, and the memory constraint is also satisfied.
                For more info visit : https://github.com/linkedin/FastTreeSHAP/blob/master/fasttreeshap/explainers/_tree.py

        shap_n_jobs : int
                (default), or a positive integer
                Number of parallel threads used to run Fast TreeSHAP. The default value of n_jobs is -1, which utilizes
                all available cores in parallel computing (Setting OMP_NUM_THREADS is unnecessary since n_jobs will
                overwrite this parameter).
                For more info visit : https://github.com/linkedin/FastTreeSHAP/blob/master/fasttreeshap/explainers/_tree.py

        random_search_n_jobs : int
                (default), or a positive integer

        memory_tolerance : int
                (default), or a positive number
                Upper limit of memory allocation (in GB) to run Fast TreeSHAP v2. The default value of memory_tolerance is -1,
                which allocates a maximum of 0.25 * total memory of the machine to run Fast TreeSHAP v2.
                For more info visit : https://github.com/linkedin/FastTreeSHAP/blob/master/fasttreeshap/explainers/_tree.py
        feature_names : [str]
                Feature names.
        approximate : bool
                Run fast, but only roughly approximate the Tree SHAP values. This runs a method
                previously proposed by Saabas which only considers a single feature ordering. Take care
                since this does not have the consistency guarantees of Shapley values and places too
                much weight on lower splits in the tree.
                For more info visit : https://github.com/linkedin/FastTreeSHAP/blob/master/fasttreeshap/explainers/_tree.py
        shortcut: False (default) or True
                Whether to use the C++ version of TreeSHAP embedded in XGBoost, LightGBM and CatBoost packages directly
                when computing SHAP values for XGBoost, LightGBM and CatBoost models, and when computing SHAP interaction
                values for XGBoost models. Current version of FastTreeSHAP package supports XGBoost and LightGBM models,
                and its support to CatBoost model is working in progress (shortcut is automatically set to be True for
                CatBoost model).
                For more info visit : https://github.com/linkedin/FastTreeSHAP/blob/master/fasttreeshap/explainers/_tree.py
        plot_shap_summary : bool
            Set True if you want to see shap summary plot of
            selected features. (default ``False``)
        save_shap_summary_plot : bool
            Set True if you want to save a shap summary plot with data_time
            label root directory of your project. (default ``False``)
        path_to_save_plot = str
            Path for saving summary plot.
        shap_fig = object
            matplotlib pyplot figure object.
            For more info visit : https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.figure.html
        test_size : float or int
            If float, it should be between 0.0 and 1.0 and represent the proportion
            of the dataset to include in the train split during estimating the best estimator
            by optimization method. If int represents the
            absolute number of train samples. If None, the value is automatically
            set to the complement of the test size.
        performance_metric : str
            Measurement of performance for classification and
            regression estimator during hyperparameter optimization while
            estimating best estimator. Classification-supported measurments are
            f1, f1_score, acc, accuracy_score, pr, precision_score,
            recall, recall_score, roc, roc_auc_score, roc_auc,
            tp, true positive, tn, true negative. Regression supported
            measurements are r2, r2_score, explained_variance_score,
            max_error, mean_absolute_error, mean_squared_error,
            median_absolute_error, and mean_absolute_percentage_error.
        cv : int, cross-validation generator or an iterable, default=None
            For more information visit
            https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html
        n_iter : int
            Number of parameter settings that are sampled. n_iter trades off runtime vs quality of the solution.
            For more information visit
            https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html
        """
        self.logging_basicConfig = logging_basicConfig
        self.verbose = verbose
        self.random_state = random_state
        self.n_features = n_features
        self.list_of_obligatory_features_that_must_be_in_model = (
            list_of_obligatory_features_that_must_be_in_model
        )

        self.list_of_features_to_drop_before_any_selection = (
            list_of_features_to_drop_before_any_selection
        )

        self.estimator = estimator
        self.estimator_params = estimator_params
        self.model_output = model_output
        self.feature_perturbation = feature_perturbation
        self.algorithm = algorithm
        self.shap_n_jobs = shap_n_jobs
        self.random_search_n_jobs = random_search_n_jobs
        self.memory_tolerance = memory_tolerance
        self.feature_names = feature_names
        self.approximate = approximate
        self.shortcut = shortcut
        self.plot_shap_summary = plot_shap_summary
        self.save_shap_summary_plot = save_shap_summary_plot
        self.path_to_save_plot = path_to_save_plot
        self.shap_fig = shap_fig
        self.cv = cv
        self.performance_metric = performance_metric
        self.n_iter = n_iter

        # Set logging config
        logging.basicConfig = self.logging_basicConfig
        self.best_estimator = None
        self.importance_df = None

    @property
    def logging_basicConfig(self):
        logging.info("Getting value for logging_basicConfig")
        return self._logging_basicConfig

    @logging_basicConfig.setter
    def logging_basicConfig(self, value):
        logging.info("Setting value for logging_basicConfig")
        self._logging_basicConfig = value

    @property
    def verbose(self):
        logging.info("Getting value for verbose")
        return self._verbose

    @verbose.setter
    def verbose(self, value):
        logging.info("Setting value for verbose")
        self._verbose = value

    @property
    def random_state(self):
        logging.info("Getting value for random_state")
        return self._random_state

    @random_state.setter
    def random_state(self, value):
        logging.info("Setting value for random_state")
        self._random_state = value

    @property
    def n_features(self):
        logging.info("Getting value for n_features")
        return self._n_features

    @n_features.setter
    def n_features(self, value):
        logging.info("Setting value for n_features")
        if value < 0:
            raise ValueError("n_features below 0 is not possible")
        self._n_features = value

    @property
    def list_of_obligatory_features_that_must_be_in_model(self):
        logging.info(
            "Getting value for list_of_obligatory_features_that_must_be_in_model"
        )
        return self._list_of_obligatory_features_that_must_be_in_model

    @list_of_obligatory_features_that_must_be_in_model.setter
    def list_of_obligatory_features_that_must_be_in_model(self, value):
        logging.info(
            "Setting value for list_of_obligatory_features_that_must_be_in_model"
        )
        self._list_of_obligatory_features_that_must_be_in_model = value

    @property
    def list_of_features_to_drop_before_any_selection(self):
        logging.info("Getting value for list of features to drop before any selection")
        return self._list_of_features_to_drop_before_any_selection

    @list_of_features_to_drop_before_any_selection.setter
    def list_of_features_to_drop_before_any_selection(self, value):
        logging.info("Setting value for list of features to drop before any selection")
        self._list_of_features_to_drop_before_any_selection = value

    @property
    def estimator(self):
        logging.info("Getting value for estimator")
        return self._estimator

    @estimator.setter
    def estimator(self, value):
        logging.info("Setting value for estimator")
        if value.__class__.__name__ not in SUPPORTED_MODELS:

            raise TypeError(
                f"{value.__class__.__name__} \
                 model is not supported yet"
            )
        self._estimator = value

    @property
    def estimator_params(self):
        logging.info("Getting value for estimator_params")
        return self._estimator_params

    @estimator_params.setter
    def estimator_params(self, value):
        logging.info(self.estimator)
        # get parameters for lightgbm.LGBMRegressor and check if
        # the selected parameters in the list or not
        if self.estimator.__class__.__name__ == "LGBMRegressor":
            if value.keys() <= LGB_REGRESSION_PARAMS_DEFAULT.keys():
                logging.info("Setting value for estimator_params")
                self._estimator_params = value
            else:
                raise TypeError(
                    f"error occures during parameter checking for \
                        {value.__class__.__name__}"
                )
        # get parameters for lightgbm.LGBMClassifier and check if
        # the selected parameters in the list or not
        if self.estimator.__class__.__name__ == "LGBMClassifier":
            if value.keys() <= LGB_CLASSIFICATION_PARAMS_DEFAULT.keys():
                logging.info("Setting value for estimator_params")
                self._estimator_params = value
            else:
                raise TypeError(
                    f"error occures during parameter checking for \
                        {value.__class__.__name__}"
                )
        # get parameters for XGBRegressor and check if
        # the selected parameters in the list or not
        if self.estimator.__class__.__name__ == "BalancedRandomForestClassifier":
            if value.keys() <= BLF_CLASSIFICATION_PARAMS_DEFAULT.keys():
                logging.info("Setting value for estimator_params")
                self._estimator_params = value
            else:
                raise TypeError(
                    f"error occures during parameter checking for \
                        {value.__class__.__name__}"
                )
        # get parameters for XGBRegressor and check if
        # the selected parameters in the list or not
        if self.estimator.__class__.__name__ == "RandomForestRegressor":
            if value.keys() <= RANDOMFOREST_REGRESSION_PARAMS_DEFAULT.keys():
                logging.info("Setting value for estimator_params")
                self._estimator_params = value
            else:
                raise TypeError(
                    f"error occures during parameter checking for \
                        {value.__class__.__name__}"
                )
        # get parameters for XGBRegressor and check if
        # the selected parameters in the list or not
        if self.estimator.__class__.__name__ == "RandomForestClassifier":
            if value.keys() <= RANDOMFOREST_CLASSIFICATION_PARAMS_DEFAULT.keys():
                logging.info("Setting value for estimator_params")
                self._estimator_params = value
            else:
                raise TypeError(
                    f"error occures during parameter checking for \
                        {value.__class__.__name__}"
                )

        # get parameters for XGBRegressor and check if
        # the selected parameters in the list or not
        if self.estimator.__class__.__name__ == "XGBRegressor":
            if value.keys() <= XGBOOST_REGRESSION_PARAMS_DEFAULT.keys():
                logging.info("Setting value for estimator_params")
                self._estimator_params = value
            else:
                raise TypeError(
                    f"error occures during parameter checking for \
                        {value.__class__.__name__}"
                )
        # get parameters for XGBClassifier and check if
        # the selected parameters in the list or not
        if self.estimator.__class__.__name__ == "XGBClassifier":
            if value.keys() <= XGBOOST_CLASSIFICATION_PARAMS_DEFAULT.keys():
                logging.info("Setting value for estimator_params")
                self._estimator_params = value
            else:
                raise TypeError(
                    f"error occures during parameter checking for \
                        {value.__class__.__name__}"
                )

        # get parameters for CatBoostClassifier and check if
        # the selected parameters in the list or not
        if self.estimator.__class__.__name__ == "CatBoostClassifier":
            if value.keys() <= CATBOOST_CLASSIFICATION_PARAMS_DEFAULT.keys():
                logging.info("Setting value for estimator_params")
                self._estimator_params = value
            else:
                raise TypeError(
                    f"error occures during parameter checking for \
                        {value.__class__.__name__}"
                )
        # get parameters for CatBoostRegressor and check if
        # the selected parameters in the list or not
        if self.estimator.__class__.__name__ == "CatBoostRegressor":
            if value.keys() <= CATBOOST_REGRESSION_PARAMS_DEFAULT.keys():
                logging.info("Setting value for estimator_params")
                self._estimator_params = value
            else:
                raise TypeError(
                    f"error occures during parameter checking for \
                        {value.__class__.__name__}"
                )

    @property
    def model_output(self):
        logging.info("Getting value for model_output")
        return self._model_output

    @model_output.setter
    def model_output(self, value):
        logging.info("Setting value for model_output")
        self._model_output = value

    @property
    def feature_perturbation(self):
        logging.info("Getting value for feature perturbation")
        return self._feature_perturbation

    @feature_perturbation.setter
    def feature_perturbation(self, value):
        logging.info("Setting value for feature perturbation")
        self._feature_perturbation = value

    @property
    def algorithm(self):
        logging.info("Getting value for algorithm")
        return self._algorithm

    @algorithm.setter
    def algorithm(self, value):
        logging.info("Setting value for algorithm")
        self._algorithm = value

    @property
    def shap_n_jobs(self):
        logging.info("Getting value for shap_n_jobs")
        return self._shap_n_jobs

    @shap_n_jobs.setter
    def shap_n_jobs(self, value):
        logging.info("Setting value for shap_n_jobs")
        self._shap_n_jobs = value

    @property
    def random_search_n_jobs(self):
        logging.info("Getting value for random_search_n_jobs")
        return self._random_search_n_jobs

    @random_search_n_jobs.setter
    def random_search_n_jobs(self, value):
        logging.info("Setting value for random_search_n_jobs")
        self._random_search_n_jobs = value

    @property
    def memory_tolerance(self):
        logging.info("Getting value for memory_tolerance")
        return self._memory_tolerance

    @memory_tolerance.setter
    def memory_tolerance(self, value):
        logging.info("Setting value for memory_tolerance")
        self._memory_tolerance = value

    @property
    def feature_names(self):
        logging.info("Getting value for feature_names")
        return self._feature_names

    @feature_names.setter
    def feature_names(self, value):
        logging.info("Setting value for feature_names")
        self._feature_names = value

    @property
    def approximate(self):
        logging.info("Getting value for approximate")
        return self._approximate

    @approximate.setter
    def approximate(self, value):
        logging.info("Setting value for approximate")
        self._approximate = value

    @property
    def shortcut(self):
        logging.info("Getting value for shortcut")
        return self._shortcut

    @shortcut.setter
    def shortcut(self, value):
        logging.info("Setting value for shortcut")
        self._shortcut = value

    @property
    def plot_shap_summary(self):
        logging.info("Getting value for plot shap summary")
        return self._plot_shap_summary

    @plot_shap_summary.setter
    def plot_shap_summary(self, value):
        logging.info("Setting value for plot shap summary")
        self._plot_shap_summary = value

    @property
    def save_shap_summary_plot(self):
        logging.info("Getting value for save_shap_summary_plot")
        return self._save_shap_summary_plot

    @save_shap_summary_plot.setter
    def save_shap_summary_plot(self, value):
        logging.info("Setting value for save_shap_summary_plot")
        self._save_shap_summary_plot = value

    @property
    def path_to_save_plot(self):
        logging.info("Getting value for path_to_save_plot")
        return self._path_to_save_plot

    @path_to_save_plot.setter
    def path_to_save_plot(self, value):
        logging.info("Setting value for path_to_save_plot")
        self._path_to_save_plot = value

    @property
    def shap_fig(self):
        logging.info("Getting value for shap_fig")
        return self._shap_fig

    @shap_fig.setter
    def shap_fig(self, value):
        logging.info("Setting value for shap_fig")
        self._shap_fig = value

    @property
    def cv(self):
        logging.info("Getting value for cv")
        return self._cv

    @cv.setter
    def cv(self, value):
        logging.info("Setting value for cv")
        self._cv = value

    @property
    def performance_metric(self):
        logging.info("Getting value for performance metric")
        return self._performance_metric

    @performance_metric.setter
    def performance_metric(self, value):
        logging.info("Setting value for performance metric")
        self._performance_metric = value

    @property
    def n_iter(self):
        logging.info("Getting value for n_iter")
        return self._n_iter

    @n_iter.setter
    def n_iter(self, value):
        logging.info("Setting value for n_iter")
        self._n_iter = value

    @property
    def best_estimator(self):
        logging.info("Getting value for best estimator")
        return self._best_estimator

    @best_estimator.setter
    def best_estimator(self, value):
        logging.info("Setting value for best estimator")
        self._best_estimator = value

    @property
    def importance_df(self):
        logging.info("Getting value for importance_df")
        return self._importance_df

    @importance_df.setter
    def importance_df(self, value):
        logging.info("Setting value for importance_df")
        self._importance_df = value

    def fit(self, X, y):
        """Fit the feature selection estimator by best params extracted
        from optimization methods.
        Parameters
        ----------
        X : Pandas DataFrame
            Training data. Must fulfill input requirements of the feature selection
            step of the pipeline.
        y : Pandas DataFrame or Pandas series
            Training targets. Must fulfill label requirements of feature selection
            step of the pipeline.
        """
        self.cols = X.columns
        self.cols = X.columns
        self.best_estimator = _calc_best_estimator_random_search(
            X,
            y,
            self.estimator,
            self.estimator_params,
            self.performance_metric,
            self.verbose,
            self.random_search_n_jobs,
            self.n_iter,
            self.cv,
        )
        print(self.estimator.__class__.__name__)

        if self.estimator.__class__.__name__ is None:
            # for unknown reason fasttreeshap does not work with RandomForestClassifier
            print(self.algorithm)
            exp = fasttreeshap.TreeExplainer(
                model=self.best_estimator,
                data=X,
                model_output=self.model_output,
                feature_perturbation=self.feature_perturbation,
                algorithm=self.algorithm,
                shap_n_jobs=self.shap_n_jobs,
                memory_tolerance=self.memory_tolerance,
                feature_names=self.feature_names,
                approximate=self.approximate,
                shortcut=self.shortcut,
            )
            shap_values_for_none_class = exp.shap_values(X)
            shapObj = exp(X)
            if self.plot_shap_summary:
                shap.summary_plot(
                    shap_values=np.take(shapObj.values, 0, axis=-1), features=X
                )
            shap_sum = np.abs(shap_values_for_none_class).mean(axis=0)
            shap_sum = shap_sum.tolist()
            print(shap_sum)

        else:
            shap_explainer = fasttreeshap.TreeExplainer(
                model=self.best_estimator,
                model_output=self.model_output,
                feature_perturbation=self.feature_perturbation,
                algorithm=self.algorithm,
                shap_n_jobs=self.shap_n_jobs,
                memory_tolerance=self.memory_tolerance,
                feature_names=self.feature_names,
                approximate=self.approximate,
                shortcut=self.shortcut,
            )
            shap_values = shap_explainer(X)
            print(shap_values)
            if self.plot_shap_summary:
                shap.summary_plot(shap_values.values, X, max_display=self.n_features)

            if self.save_shap_summary_plot:
                # Plot simple sinus function
                shap.summary_plot(
                    shap_values.values, X, max_display=self.n_features, show=False
                )
                plt.tight_layout()
                plt.savefig(self.path_to_save_plot)
                self.shap_fig.show()

            shap_sum = np.abs(shap_values.values).mean(axis=0)
            shap_sum = shap_sum.tolist()

        self.importance_df = pd.DataFrame([X.columns.tolist(), shap_sum]).T
        print(self.importance_df)
        self.importance_df.columns = ["column_name", "shap_importance"]

        print(self.importance_df)
        self.importance_df = self.importance_df.sort_values(
            "shap_importance", ascending=False
        )
        print(self.importance_df)
        print(self.importance_df[0 : self.n_features])
        num_feat = min([self.n_features, self.importance_df.shape[0]])
        self.selected_cols = self.importance_df["column_name"][0:num_feat].to_list()

        return self

    def transform(self, X):
        """Transform the data, and apply the transform to data to be ready for feature selection
        estimator.
        Parameters
        ----------
        X : Pandas DataFrame
            Training data. Must fulfill input requirements of feature selection
            step of the pipeline.
        """

        return X[self.selected_cols]

    def get_feature_importance_df(self):
        """Return a Pandas data frame with two columns. One is the name of features and another
                one is the mean of shap values sorted increasingly.
        .
                Parameters
                ----------
                None

                Raises
                ------
                ValueError
                    Before feature selection, this data frame is None.
        """
        if self.importance_df is None:
            raise ValueError(
                "The data frame is empty. First, use fit or transform method to populate it :( "
            )
        return self.importance_df

    def get_fig_object(self):
        """Return a figure object related to shap summary plot.

        Parameters
        ----------
        None

        Raises
        ------
        ValueError
            If the saved plot is false, the return value is not meaningful.
        """
        if self.shap_fig is None:
            raise ValueError("The fig object is empty :( ")
        return self.shap_fig
