import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from xgboost import XGBClassifier
from zoish.feature_selectors.shap_selectors import *

# Generate a classification dataset
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)

# Convert to pandas dataframes
X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
y_df = pd.Series(y, name='target')

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_df, y_df, test_size=0.2, random_state=42)

# Create and fit the model
model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
model.fit(X_train, y_train)

# Create and fit the selector
selector = ShapFeatureSelector(model=model, num_features=10)
selector.fit(X_train, y_train)

# Transform the training data
X_train_transformed = selector.transform(X_train)

# Re-train your model on the transformed data
model.fit(X_train_transformed, y_train)

# Now you can make predictions using transformed data
X_test_transformed = selector.transform(X_test)
y_pred = model.predict(X_test_transformed)

print(f"Model score: {selector.score(X_test, y_test)}")

# Plot the feature importance
factory = ShapPlotFeatures(selector , type_of_plot="decision_plot")
factory.plot_features()
