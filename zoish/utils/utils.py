def is_lightgbm_binary_classification(model):
    """
    Checks if the given model is an instance of LightGBM's LGBMClassifier configured for binary classification.
    This function explicitly checks if the model is a LightGBM model and if it's set up for binary classification.

    Args:
    - model: The model instance to check.

    Returns:
    - (bool, bool): A tuple where the first boolean indicates if the model is a LightGBM model,
                    and the second boolean indicates if it is configured for binary classification.
    """
    try:
        # Dynamically import LGBMClassifier to check the model type
        from lightgbm import LGBMClassifier

        # Check if the model is an instance of LGBMClassifier
        if isinstance(model, LGBMClassifier):
            # If the model has the 'classes_' attribute and exactly two classes, it's binary classification
            if hasattr(model, "classes_") and len(model.classes_) == 2:
                return True, True  # It's a LightGBM binary classification model
            else:
                return (
                    True,
                    False,
                )  # It's a LightGBM model, but not binary classification
        else:
            return False, False  # Not a LightGBM model
    except ImportError:
        # Handle the case where LightGBM is not installed
        print("LightGBM is not installed.")
        return False, False
