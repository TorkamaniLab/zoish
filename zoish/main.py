from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score
from xgboost import XGBClassifier
from zoish.feature_selectors.shap_selectors import *
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
model.fit(X_train, y_train)
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
factory = ShapFeatureSelectorFactory(model)
factory.set_plot_features_params(path_to_save_plot='./shap_plot.png')
factory.plot_features_all()
