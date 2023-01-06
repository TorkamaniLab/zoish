# https://www.youtube.com/watch?v=KN4FgzRj4d4

import pytest
from zoish.feature_selectors.shap_selectors import ShapFeatureSelector
from feature_engine.imputation import CategoricalImputer, MeanMedianImputer
from category_encoders import OrdinalEncoder
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score

from zoish.feature_selectors.recursive_feature_addition import (
    RecursiveFeatureAdditionFeatureSelector,
)
from zoish.feature_selectors.recursive_feature_elimination import (
    RecursiveFeatureEliminationFeatureSelector,
)
from zoish.feature_selectors.select_by_shuffling import SelectByShufflingFeatureSelector
from zoish.feature_selectors.single_feature_selectors import (
    SingleFeaturePerformanceFeatureSelector,
)
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner
import xgboost
import pandas as pd
from sklearn.model_selection import train_test_split
import optuna 

 
@pytest.fixture()
def datasets():
    class DataHandling:
        def __init__(self, url,col_names, problem_name, random_state, test_size):
            self.url=url
            self.col_names=col_names
            self.problem_name=problem_name
            self.random_state=random_state
            self.test_size=test_size
            self.data=None
            self.X=None
            self.y=None
            self.X_train=None
            self.y_train=None
            self.X_test=None
            self.y_test=None
            self.int_cols=None
            self.float_cols=None
            self.cat_cols=None
        
        def read_data(self):
            self.data = pd.read_csv(self.url, header=None, names=self.col_names, sep=",")
            return self.data

        def x_y_split(self):
            self.read_data()
            if self.problem_name=="hardware":
                    self.X = self.data.loc[:, self.data.columns != "PRP"]
                    self.y = self.data.loc[:, self.data.columns == "PRP"]
                    self.X_train,self.X_test,self.y_train,self.y_test= train_test_split(self.X, self.y, test_size=self.test_size, random_state=self.random_state)
                    self.int_cols =  self.X_train.select_dtypes(include=['int']).columns.tolist()
                    self.float_cols =  self.X_train.select_dtypes(include=['float']).columns.tolist()
                    self.cat_cols =  self.X_train.select_dtypes(include=['object']).columns.tolist()

            if self.problem_name=="audiology":
                    self.data.loc[
                        (self.data["class"] == 1) | (self.data["class"] == 2), "class"
                    ] = 0
                    self.data.loc[self.data["class"] == 3, "class"] = 1
                    self.data.loc[self.data["class"] == 4, "class"] = 2
                    self.data["class"] = self.data["class"].astype(int)
                    self.X = self.data.loc[:, self.data.columns != "class"]
                    self.y = self.data.loc[:, self.data.columns == "class"]
                    self.X_train,self.X_test,self.y_train,self.y_test= train_test_split(self.X, self.y, test_size=self.test_size, random_state=self.random_state)
                    self.y_test = self.y_test .values.ravel()
                    self.y_train = self.y_train.values.ravel()
                    self.int_cols =  self.X_train.select_dtypes(include=['int']).columns.tolist()

            if self.problem_name=="adult":
                    self.data.loc[self.data["label"] == "<=50K", "label"] = 0
                    self.data.loc[self.data["label"] == " <=50K", "label"] = 0

                    self.data.loc[self.data["label"] == ">50K", "label"] = 1
                    self.data.loc[self.data["label"] == " >50K", "label"] = 1

                    self.data["label"] = self.data["label"].astype(int)
                    self.X = self.data.loc[:, self.data.columns != "label"]
                    self.y = self.data.loc[:, self.data.columns == "label"]
                    self.X_train,self.X_test,self.y_train,self.y_test = train_test_split(self.X, self.y,test_size=self.test_size,stratify=self.y["label"],random_state=self.random_state)
                    self.int_cols =  self.X_train.select_dtypes(include=['int']).columns.tolist()
                    self.float_cols =  self.X_train.select_dtypes(include=['float']).columns.tolist()
                    self.cat_cols =  self.X_train.select_dtypes(include=['object']).columns.tolist()

            return self
    
    audiology = DataHandling(
        url="https://archive.ics.uci.edu/ml/machine-learning-databases/lymphography/lymphography.data",
        col_names=[
            "class",
            "lymphatics",
            "block of affere",
            "bl. of lymph. c",
            "bl. of lymph. s",
            "by pass",
            "extravasates",
            "regeneration of",
            "early uptake in",
            "lym.nodes dimin",
            "lym.nodes enlar",
            "changes in lym.",
            "defect in node",
            "changes in node",
            "special forms",
            "dislocation of",
            "exclusion of no",
            "no. of nodes in",

        ],
        problem_name="audiology",
        random_state=42,
        test_size=0.33,
        )
    adult = DataHandling(
        url="https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
        col_names=[
           "age",
            "workclass",
            "fnlwgt",
            "education",
            "education-num",
            "marital-status",
            "occupation",
            "relationship",
            "race",
            "sex",
            "capital-gain",
            "capital-loss",
            "hours-per-week",
            "native-country",
            "label",
        ],
        problem_name="adult",
        random_state=42,
        test_size=0.33,
        )
    hardware = DataHandling(
        url="https://archive.ics.uci.edu/ml/machine-learning-databases/cpu-performance/machine.data",
        col_names=[
           "vendor name",
            "Model Name",
            "MYCT",
            "MMIN",
            "MMAX",
            "CACH",
            "CHMIN",
            "CHMAX",
            "PRP"
        ],
        problem_name="hardware",
        random_state=42,
        test_size=0.33,
        )
    return [adult.x_y_split(),audiology.x_y_split(),hardware.x_y_split()]


@pytest.fixture()
def setup_factories(datasets):

    class FeatureSelectorFactories:
        def __init__(
            self,
            X=None,
            y=None,
            verbose=None,
            random_state=None,
            estimator=None,
            estimator_params=None,
            fit_params=None,
            method=None,
            n_features=None,
            threshold=None,
            list_of_obligatory_features_that_must_be_in_model=None,
            list_of_features_to_drop_before_any_selection=None,
            model_output=None,
            feature_perturbation=None,
            algorithm=None,
            shap_n_jobs=None,
            memory_tolerance=None,
            feature_names=None,
            approximate=None,
            shortcut=None,
            #
            measure_of_accuracy=None,
            # optuna params
            with_stratified=None,
            test_size=None,
            n_jobs=None,
            # optuna params
            # optuna study init params
            study=None,
            # optuna optimization params
            study_optimize_objective=None,
            study_optimize_objective_n_trials=None,
            study_optimize_objective_timeout=None,
            study_optimize_n_jobs=None,
            study_optimize_catch=None,
            study_optimize_callbacks=None,
            study_optimize_gc_after_trial=None,
            study_optimize_show_progress_bar=None,
            cv=None,
            variables=None,
            scoring=None,
            confirm_variables=None,
        ):
            self.X=X
            self.y=y
            self.verbose=verbose
            self.random_state=random_state
            self.estimator=estimator
            self.estimator_params=estimator_params
            self.fit_params=fit_params
            self.method=method
            self.n_features=n_features
            self.threshold=threshold
            self.list_of_obligatory_features_that_must_be_in_model=list_of_obligatory_features_that_must_be_in_model
            self.list_of_features_to_drop_before_any_selection=list_of_features_to_drop_before_any_selection
            self.model_output=model_output
            self.feature_perturbation=feature_perturbation
            self.algorithm=algorithm
            self.shap_n_jobs=shap_n_jobs
            self.memory_tolerance=memory_tolerance
            self.feature_names=feature_names
            self.approximate=approximate
            self.shortcut=shortcut
            #
            self.measure_of_accuracy=measure_of_accuracy
            # optuna params
            self.with_stratified=with_stratified
            self.test_size=test_size
            self.n_jobs=n_jobs
            # optuna params
            # optuna study init params
            self.study=study
            # optuna optimization params
            self.study_optimize_objective=study_optimize_objective
            self.study_optimize_objective_n_trials=study_optimize_objective_n_trials
            self.study_optimize_objective_timeout=study_optimize_objective_timeout
            self.study_optimize_n_jobs=study_optimize_n_jobs
            self.study_optimize_catch=study_optimize_catch
            self.study_optimize_callbacks=study_optimize_callbacks
            self.study_optimize_gc_after_trial=study_optimize_gc_after_trial
            self.study_optimize_show_progress_bar=study_optimize_show_progress_bar
            self.cv=cv
            self.variables=variables
            self.scoring=scoring
            self.confirm_variables=confirm_variables

        def get_shap_selector(self):
            shap = ShapFeatureSelector.shap_feature_selector_factory.set_model_params(
                X=self.X,
                y=self.y,
                verbose=self.verbose,
                random_state=self.random_state,
                estimator=self.estimator,
                estimator_params=self.estimator_params,
                fit_params=self.fit_params,
                method=self.method,
                n_features=self.n_features,
                threshold=self.threshold,
                list_of_obligatory_features_that_must_be_in_model=self.list_of_obligatory_features_that_must_be_in_model,
                list_of_features_to_drop_before_any_selection=self.list_of_features_to_drop_before_any_selection,
                ).set_shap_params(
                model_output=self.model_output,
                feature_perturbation=self.feature_perturbation,
                algorithm=self.algorithm,
                shap_n_jobs=self.shap_n_jobs,
                memory_tolerance=self.memory_tolerance,
                feature_names=self.feature_names,
                approximate=self.approximate,
                shortcut=self.shortcut,
                ).set_optuna_params(
                measure_of_accuracy=self.measure_of_accuracy,
                # optuna params
                with_stratified=self.with_stratified,
                test_size=self.test_size,
                n_jobs=self.n_jobs,
                # optuna params
                # optuna study init params
                study=self.study,
                # optuna optimization params
                study_optimize_objective=self.study_optimize_objective,
                study_optimize_objective_n_trials=self.study_optimize_objective_n_trials,
                study_optimize_objective_timeout=self.study_optimize_objective_timeout,
                study_optimize_n_jobs=self.study_optimize_n_jobs,
                study_optimize_catch=self.study_optimize_catch,
                study_optimize_callbacks=self.study_optimize_callbacks,
                study_optimize_gc_after_trial=self.study_optimize_gc_after_trial,
                study_optimize_show_progress_bar=self.study_optimize_show_progress_bar,
                )
            return shap

    shap_1_hardware = FeatureSelectorFactories(
        X=datasets[2].X_train,
        y=datasets[2].y_train,
        verbose=10,
        random_state=0,
        estimator=xgboost.XGBRegressor(),
        estimator_params={
            "max_depth": [4, 5],
        },
        fit_params={
            "callbacks": None,
        },
        method="optuna",
        # if n_features=None only the threshold will be considered as a cut-off of features grades.
        # if threshold=None only n_features will be considered to select the top n features.
        # if both of them are set to some values, the threshold has the priority for selecting features.
        n_features=5,
        threshold=None,
        list_of_obligatory_features_that_must_be_in_model=[],
        list_of_features_to_drop_before_any_selection=[],
        model_output="raw",
        feature_perturbation="interventional",
        algorithm="v2",
        shap_n_jobs=-1,
        memory_tolerance=-1,
        feature_names=None,
        approximate=False,
        shortcut=False,
        measure_of_accuracy="r2_score(y_true, y_pred)",
        # optuna params
        with_stratified=False,
        test_size=0.3,
        n_jobs=-1,
        # optuna params
        # optuna study init params
        study=optuna.create_study(
            storage=None,
            sampler=TPESampler(),
            pruner=HyperbandPruner(),
            study_name="example of optuna optimizer",
            direction="maximize",
            load_if_exists=False,
            directions=None,
        ),
        # optuna optimization params
        study_optimize_objective=None,
        study_optimize_objective_n_trials=20,
        study_optimize_objective_timeout=600,
        study_optimize_n_jobs=-1,
        study_optimize_catch=(),
        study_optimize_callbacks=None,
        study_optimize_gc_after_trial=False,
        study_optimize_show_progress_bar=False
    ).get_shap_selector()

    return shap_1_hardware
    
        

def test_second_func(datasets,setup_factories):

    print(datasets[2].X_test.head())

    pipeline =Pipeline([
            # int missing values imputers
            ('intimputer', MeanMedianImputer(
                imputation_method='median', variables=datasets[2].int_cols)),
            # category missing values imputers
            ('catimputer', CategoricalImputer(variables=datasets[2].cat_cols)),
            #
            ('catencoder', OrdinalEncoder()),
            # feature selection
            ("sfsf", setup_factories),
            # add any regression model from sklearn e.g., LinearRegression
            ('regression', LinearRegression())])

    pipeline.fit(datasets[2].X_train,datasets[2].y_train)
    y_pred = pipeline.predict(datasets[2].X_test)
    assert r2_score(datasets[2].y_test,y_pred) > 80

    
# https://docs.pytest.org/en/7.1.x/how-to/fixtures.html


# #test_second_func(read_datasets,setup_factories) """

# # contents of test_append.py
# import pytest


# # Arrange
# @pytest.fixture
# def first_entry():
#     return "a"


# # Arrange
# @pytest.fixture
# def order(first_entry):
#     b = 4
#     return [first_entry],b


# def test_string(order):
#     # Act
#     order[0].append("b")

#     # Assert
#     assert order[0] == ["a", "b"]


# def test_int(order):
#     # Act
#     order[0].append(2)

#     # Assert
#     assert order[0] == ["a", 2]
