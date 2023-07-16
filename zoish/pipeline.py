import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from xgboost import XGBClassifier
from zoish.feature_selectors.shap_selectors import *

# Generate a classification dataset
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)

# Convert to pandas dataframes
X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
y_df = pd.Series(y, name='target')

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_df, y_df, test_size=0.2, random_state=42)

# Create a pipeline with the model and the selector
model = XGBClassifier(use_label_encoder=False, eval_metric="logloss").fit(X_train,y_train)
selector = ShapFeatureSelector(model=model, num_features=10)
pipeline = make_pipeline(selector, model)

# Fit and predict using the pipeline
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

# Plot the feature importance
factory = ShapPlotFeatures(selector , type_of_plot="summary_plot")
factory.summary_plot()
