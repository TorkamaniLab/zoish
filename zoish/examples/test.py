# %% [markdown]
# ### Imports

# %%
from zoish.feature_selectors.single_feature_selectors import SingleFeaturePerformanceFeatureSelector
import xgboost
from sklearn.model_selection import KFold, train_test_split
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, f1_score,make_scorer
from feature_engine.imputation import CategoricalImputer, MeanMedianImputer
from category_encoders import OrdinalEncoder


# %% [markdown]
# #### Example: Use Adult Data Set (a classification problem)
# ###### https://archive.ics.uci.edu/ml/datasets/Adult
# 

# %% [markdown]
# #### Read data

# %%
urldata = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
# column names
col_names = [
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
]
# read data
data = pd.read_csv(urldata, header=None, names=col_names, sep=",")
data.head()

# %% [markdown]
# #### Define labels and train-test split
# 

# %%
data.loc[data["label"] == "<=50K", "label"] = 0
data.loc[data["label"] == " <=50K", "label"] = 0

data.loc[data["label"] == ">50K", "label"] = 1
data.loc[data["label"] == " >50K", "label"] = 1

data["label"] = data["label"].astype(int)

# # Train test split

X = data.loc[:, data.columns != "label"]
y = data.loc[:, data.columns == "label"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, stratify=y["label"], random_state=42
)


# %% [markdown]
# #### Define feature selector step 

# %%

single_feature_performance_feature_selector_factory = (
    SingleFeaturePerformanceFeatureSelector.single_feature_performance_feature_selector_factory.set_model_params(
        X=X_train,
        y=y_train,
        verbose=0,
        random_state=0,
        estimator=xgboost.XGBClassifier(),
        estimator_params={
            "max_depth": [4, 5],
        },
        method="gridsearch",
        n_features=5,
        list_of_obligatory_features_that_must_be_in_model=[],
        list_of_features_to_drop_before_any_selection=[],
    )
    .set_single_feature_params(
        threshold=0.6,
        cv=3,
        variables=None,
        confirm_variables=None,
        scoring='roc_auc',
        
    )
    .set_gridsearchcv_params(
        measure_of_accuracy=make_scorer(f1_score, greater_is_better=True, average='macro'),
        verbose=0,
        n_jobs=-1,
        cv=KFold(5),
    )
)

# %% [markdown]
# #### Find feature type for later use
# 

# %%
int_cols = X_train.select_dtypes(include=["int"]).columns.tolist()
float_cols = X_train.select_dtypes(include=["float"]).columns.tolist()
cat_cols = X_train.select_dtypes(include=["object"]).columns.tolist()


# %% [markdown]
# #### Define pipeline

# %%
pipeline = Pipeline(
    [
        # int missing values imputers
        (
            "intimputer",
            MeanMedianImputer(imputation_method="median", variables=int_cols),
        ),
        # category missing values imputers
        ("catimputer", CategoricalImputer(variables=cat_cols)),
        #
        ("catencoder", OrdinalEncoder()),
        # feature selection
        ("sfpfsf", single_feature_performance_feature_selector_factory),
        # classification model
        ("SVC", SVC()),
    ]
)

# %% [markdown]
# #### Run Pipeline

# %%
pipeline.fit(X_train, y_train.values.ravel())
y_pred = pipeline.predict(X_test)

# %% [markdown]
# #### Check performance of the Pipeline

# %%

print("F1 score : ")
print(f1_score(y_test, y_pred))
print("Classification report : ")
print(classification_report(y_test, y_pred))
print("Confusion matrix : ")
print(confusion_matrix(y_test, y_pred))


# %% [markdown]
# #### Plot summary plot for selected features

# %%
SingleFeaturePerformanceFeatureSelector.single_feature_performance_feature_selector_factory.plot_features_all(
    path_to_save_plot="../plots/random_search_classification_3_summary_plot"
)


SingleFeaturePerformanceFeatureSelector.single_feature_performance_feature_selector_factory.get_list_of_features_and_grades()

