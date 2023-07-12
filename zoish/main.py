from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score
from xgboost import XGBClassifier
from zoish.feature_selectors.shap_selectors import *

# Create a binary classification problem
X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, n_redundant=10, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Create the pipeline
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train,y_train)
selector = ShapFeatureSelector(model, num_features=10)

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('selector', selector),
    ('classifier', XGBClassifier(use_label_encoder=False, eval_metric='logloss'))
])

# Fit and predict
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)

# Evaluate the performance
score = f1_score(y_test, y_pred)
print(f'F1 score: {score}')

# Plot the feature importance
factory = ShapFeatureSelectorFactory()
factory.set_plot_features_params(path_to_save_plot='./shap_plot.png')
factory.plot_features_all()
