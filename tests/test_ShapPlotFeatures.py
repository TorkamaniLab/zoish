# Import necessary libraries
import pandas as pd
import numpy as np
import pytest
from unittest.mock import patch
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from zoish.feature_selectors.shap_selectors import ShapFeatureSelector, ShapPlotFeatures

# Use pytest's parameterize decorator to run the test for different data types
@pytest.mark.parametrize("data_type", ['DataFrame', 'ndarray'])
def test_initialization_and_plotting(data_type):
    # Generate synthetic data for testing
    X_numpy = np.random.rand(100, 10)
    y = np.random.randint(0, 2, size=100)
    
    # Convert to DataFrame if required
    if data_type == 'DataFrame':
        X = pd.DataFrame(X_numpy)
    else:
        X = X_numpy

    # Initialize and fit the RandomForest model
    model = RandomForestClassifier()
    model.fit(X, y)

    # Initialize and fit ShapFeatureSelector
    selector = ShapFeatureSelector(
        model,
        num_features=int(X.shape[1] * 0.5),
        n_iter=5,
        direction="maximum",
        scoring="f1",
        cv=KFold(n_splits=5, shuffle=True),
    )
    selector.fit(X, y)

    # Initialize ShapPlotFeatures for visualization
    shap_plot = ShapPlotFeatures(selector)

    # Use mock objects to test if the plotting methods are called correctly
    with patch.object(shap_plot, 'summary_plot') as mock_summary, \
         patch.object(shap_plot, 'bar_plot') as mock_bar, \
         patch.object(shap_plot, 'summary_plot_full') as mock_summary_full, \
         patch.object(shap_plot, 'bar_plot_full') as mock_bar_full, \
         patch.object(shap_plot, 'decision_plot') as mock_decision, \
         patch.object(shap_plot, 'decision_plot_full') as mock_decision_full, \
         patch.object(shap_plot, 'dependence_plot') as mock_dependence_plot:

        # Call various SHAP plotting methods
        shap_plot.summary_plot()
        shap_plot.bar_plot()
        shap_plot.summary_plot_full()
        shap_plot.bar_plot_full()
        shap_plot.decision_plot()
        shap_plot.decision_plot_full()

        # Specifically test the dependence plot for the first feature
        first_feature = 0 if data_type == 'ndarray' else X.columns[0]
        shap_plot.dependence_plot(first_feature)

        # Check if the mock methods were called
        mock_summary.assert_called_once()
        mock_bar.assert_called_once()
        mock_summary_full.assert_called_once()
        mock_bar_full.assert_called_once()
        mock_decision.assert_called_once()
        mock_decision_full.assert_called_once()
        mock_dependence_plot.assert_called_with(first_feature)
