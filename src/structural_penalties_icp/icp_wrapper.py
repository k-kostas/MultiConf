import torch
import numpy as np
import pandas as pd
import warnings

from typing import Union, List
from sklearn.utils.validation import check_array
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError

from .icp_predictor import InductiveConformalPredictor
from .prediction_regions import PredictionRegions
from .utils import _check_multihot_labels, _fingerprint_model, _normalize_device, _is_tensor

InputData = Union[torch.Tensor, np.ndarray, pd.DataFrame, pd.Series, List, float]

class ICPWrapper:
    """
    A wrapper for Inductive Conformal Prediction with Structural Penalties (Scikit-Learn compatible).

    This class manages the lifecycle of the underlying multi-label classifier and the
    conformal predictor. It handles model training, calibration, and the efficient
    update of structural penalties weights (Hamming/Cardinality) without unnecessary retraining.


    .. note::
        **Advanced Usage - Switching Strategies:**
        You can switch the classification strategy or update its parameters dynamically.
        If the wrapper detects a change (via fingerprinting) during `calibrate`, it will
        automatically retrain the new model on the cached training data.


    Parameters
    ----------
    classification_strategy : sklearn.base.BaseEstimator
        The underlying multi-label classification model (e.g., RandomForest, ClassifierChain).
        Must support `fit` and `predict_proba`.
    weight_hamming : float, optional, default=0.0
        Initial weight for the Hamming distance penalty.
    weight_cardinality : float, optional, default=0.0
        Initial weight for the Cardinality penalty.
    device : str or torch.device, optional, default='cpu'
        The device to use for tensor computations ('cpu' or 'cuda').
    """

    def __init__(self,
                 classification_strategy,
                 weight_hamming: float = 0.0,
                 weight_cardinality: float = 0.0,
                 device: Union[str, torch.device] = 'cpu'
                 ):
        self.strategy = classification_strategy
        self._weight_hamming = weight_hamming
        self._weight_cardinality = weight_cardinality
        self.device = _normalize_device(device)

        self.strategy_fingerprint = None
        self.kwargs = {}
        self.proper_train_features = None
        self.proper_train_labels = None
        self.icp = None


    def predict_proba_to_tensor(self, features:InputData) -> torch.Tensor:
        """
        Predicts probabilities and converts them to a unified Tensor format.

        This method handles different output formats from Scikit-Learn classifiers (e.g., standard arrays
        vs. list of arrays from `MultiOutputClassifier`) and ensures the output is a single
        tensor of shape `(n_samples, n_classes)`.


        Parameters
        ----------
        features : array-like
            The input features for prediction. Shape: (n_samples, n_features).


        Returns
        -------
        torch.Tensor
            A tensor containing the predicted probabilities for the positive class (1).
            Shape: (n_samples, n_classes).


        Example
        -------
        >>> # 1. Initialize and Fit Wrapper
        >>> # Load data (X_train, y_train) and model
        >>> wrapper = ICPWrapper(model)
        >>> wrapper.fit(X_train, y_train)
        >>>
        >>> # 2. Convert Probabilities to Tensor
        >>> # Internally, this handles the list conversion and any single-class edge cases.
        >>> probs = wrapper.predict_proba_to_tensor(X_train)
        """

        if torch.is_tensor(features):
            features = check_array(features.detach().cpu().numpy(), accept_sparse=True, dtype=None, ensure_2d=True)
        else:
            features = check_array(features, accept_sparse=True, dtype=None, ensure_2d=True)

        try:
            check_is_fitted(self.strategy)
        except NotFittedError:
            raise RuntimeError("Classifier has not been fitted yet. Please call fit() first.")

        probs = self.strategy.predict_proba(features)
        if isinstance(probs, list):
            extracted_probs = []
            for i, p in enumerate(probs):
                if p.shape[1] == 2:
                    extracted_probs.append(p[:, 1])
                elif p.shape[1] == 1:
                    print('\n', f'{i=}, {p=}, {p.shape= }')
                    warnings.warn("One of the labels has only 1 class. Getting the predicted class from the classifier.", RuntimeWarning)
                    present_class = self.strategy.classes_[i][0]
                    print(f'{self.strategy.classes_}')
                    print(f'{present_class=}')
                    if present_class == 0:
                        extracted_probs.append(np.zeros_like(p[:, 0]))
                    else:
                        extracted_probs.append(p[:, 0])
                else:
                    extracted_probs.append(p[:, 1])
            probs = np.array(extracted_probs).T

        return torch.tensor(probs, device=self.device, dtype=torch.float32)


    def fit(self, train_features: InputData, train_labels: InputData, **kwargs):
        """
        Fits the underlying multi-label classification model.

        This method trains the `classification_strategy` on the provided `features` and `labels`.
        It also initializes the `InductiveConformalPredictor` engine (calculating the Covariance Matrix
        and Structural Penalties).


        Parameters
        ----------
        train_features : array-like
            The training features. Shape: (n_samples, n_features).
        train_labels : array-like
            The training labels (binary multi-hot). Shape: (n_samples, n_classes).
        **kwargs : dict
            Optional arguments to pass to the classifier's `set_params` or `fit` method.


        Returns
        -------
        self : object
            The fitted wrapper instance.


        Examples
        --------
        >>> import numpy as np
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> from sklearn.multioutput import MultiOutputClassifier
        >>>
        >>> # 1. Generate Dummy Training Data
        >>> # 100 samples, 5 features, 3 target labels
        >>> X_train = np.random.rand(100, 5)
        >>> y_train = np.random.randint(0, 2, (100, 3))
        >>>
        >>> # 2. Initialize Wrapper
        >>> base_model = MultiOutputClassifier(RandomForestClassifier())
        >>> wrapper = ICPWrapper(base_model)
        >>>
        >>> # 3. Fit the Model (Standard)
        >>> wrapper.fit(X_train, y_train)
        >>>
        >>> # 4. Fit with Dynamic Parameters (Advanced)
        >>> # You can update the classifier's hyperparameters during the fit call.
        >>> # Note the 'estimator__' prefix for wrapped sklearn models.
        >>> args = {'estimator__n_neighbors': 5}
        >>> wrapper.fit(X_train, y_train, **args)
        """

        print("--- Starting Fit Procedure ---")
        train_labels = _check_multihot_labels(train_labels)
        if torch.is_tensor(train_labels):
            train_labels = check_array(train_labels.detach().cpu().numpy(), ensure_2d=False, allow_nd=True)
        else:
            train_labels = check_array(train_labels, ensure_2d=False, allow_nd=True)

        if torch.is_tensor(train_features):
            train_features = check_array(train_features.detach().cpu().numpy(), accept_sparse=True, dtype=None, ensure_2d=True)
        else:
            train_features = check_array(train_features, accept_sparse=True, dtype=None, ensure_2d=True)

        if kwargs:
            self.strategy.set_params(**kwargs)
            self.kwargs = kwargs

        print("Fitting Classifier...")
        self.strategy.fit(train_features, train_labels)
        self.strategy_fingerprint = _fingerprint_model(self.strategy, self.kwargs)
        self.proper_train_features = train_features
        self.proper_train_labels = train_labels

        print(f"Classifier trained with features shape: {train_features.shape}")

        # print("Initializing Predictor & Calculating Penalties...")
        train_probabilities = self.predict_proba_to_tensor(train_features)

        self.icp = InductiveConformalPredictor(
            predicted_probabilities=train_probabilities.to(self.device),
            true_labels=_is_tensor(self.proper_train_labels).to(self.device),
            weight_hamming=self._weight_hamming,
            weight_cardinality=self._weight_cardinality,
            device=self.device
        )

        print("--- Fit Complete ---\n")

        return self


    def calibrate(self, calib_features: InputData, calib_labels: InputData):
        """
        Calibrates the conformal predictor using a dedicated calibration set.

        This step calculates the nonconformity scores and determines the thresholds
        required to guarantee coverage. It also handles updates: if the
        underlying classifier configuration has changed since `fit` was called,
        it automatically retrains the classifier and updates the ICP covariance matrix.


        Parameters
        ----------
        calib_features : array-like
            Features of the calibration set.
        calib_labels : array-like
            Labels of the calibration set.


        Returns
        -------
        self : object
            The calibrated wrapper instance.


        Raises
        ------
        ValueError
            If the 'icp' attribute is None.
        ValueError
            If the classifier has not been fitted yet (`fit` must be called first).


        Example
        --------
        >>> # 1. Initialize & Fit (See fit() function documentation for details)
        >>>
        >>> # 2. Generate Dummy Training Data
        >>> # 100 samples, 5 features, 3 target labels
        >>> X_calib = np.random.rand(100, 5)
        >>> y_calib = np.random.randint(0, 2, (100, 3))
        >>>
        >>> # Calibrate
        >>> wrapper.calibrate(X_calib, y_calib)


        Example
        --------
        >>> # Automatic Retraining (Strategy Switching)
        >>> # Change the underlying model dynamically
        >>> from sklearn.neighbors import KNeighborsClassifier
        >>> from sklearn.multioutput import ClassifierChain
        >>>
        >>> wrapper.strategy = ClassifierChain(KNeighborsClassifier(n_neighbors=5))
        >>>
        >>> # Calling calibrate again detects the change and retrains automatically
        >>> wrapper.calibrate(X_calib, y_calib)
        """

        print("--- Starting Calibration ---")
        calib_labels = _check_multihot_labels(calib_labels)
        calib_labels = _is_tensor(calib_labels).to(self.device)

        if torch.is_tensor(calib_features):
            calib_features = check_array(calib_features.detach().cpu().numpy(), accept_sparse=True, dtype=None, ensure_2d=True)
        else:
            calib_features = check_array(calib_features, accept_sparse=True, dtype=None, ensure_2d=True)

        if self.icp is None:
            raise RuntimeError("Run the fit() procedure first.")

        try:
            check_is_fitted(self.strategy)
            is_fitted = True
        except NotFittedError:
            is_fitted = False

        if not is_fitted or self.strategy_fingerprint is None or self.strategy_fingerprint != _fingerprint_model(self.strategy, self.kwargs):
            print("Classifier model changed. Retraining the classifier, unless the arguments are not compatible with the new model. A type error will be raised.")

            if self.proper_train_features is None or self.proper_train_labels is None:
                raise RuntimeError("Cannot retrain the classifier. The training data is not available. Please call fit() first.")
            else:
                try:
                    self.strategy.fit(self.proper_train_features, self.proper_train_labels)
                    check_is_fitted(self.strategy)
                    self.strategy_fingerprint = _fingerprint_model(self.strategy, self.kwargs)

                    print("Updating ICP Covariance Matrix...")
                    new_train_probs = self.predict_proba_to_tensor(self.proper_train_features)
                    self.icp.covariance_matrix_preprocessing(new_train_probs.to(self.device), _is_tensor(self.proper_train_labels).to(self.device))
                except Exception as e:
                    raise RuntimeError(f"Failed to retrain classifier with new parameters: {e}")

        calib_probabilities = self.predict_proba_to_tensor(calib_features)

        self.icp.calibrate(calib_probabilities.to(self.device), calib_labels.to(self.device))
        print("--- Calibration Complete ---\n")

        return self


    def predict(self, test_features: InputData) -> PredictionRegions:
        """
        Generates conformal prediction regions for the input features.

        This method calculates p-values for all test samples based on the calibrated scores.

        Parameters
        ----------
        test_features : array-like
            The test features. Shape: (n_samples, n_features).

        Returns
        -------
        PredictionRegions
            A callable object containing p-values. You must call this object
            with a specific `significance_level` to retrieve the final prediction sets.

        Raises
        ------
        RuntimeError
            If the 'icp' attribute is None.
        RuntimeError
            If the classifier has not been fitted yet (`fit` must be called first).
        RuntimeError
            If the classifier model changed. Run the fit and calibration procedure.

        Example
        --------
        >>> # ... Assume wrapper is already fitted and calibrated (see fit() for details) ...
        >>> X_test = np.random.rand(10, 5)
        >>>
        >>> # 1. Get the Prediction Container
        >>> # This calculates p-values but doesn't apply a threshold yet.
        >>> prediction_region_obj = wrapper.predict(X_test)
        >>>
        >>> # 2. Extract Prediction Sets (e.g., at 10% significance / 90% confidence)
        >>> # Returns a list of Tensors, where each Tensor contains the indices of predicted labels.
        >>> prediction_sets = prediction_region_obj(significance_level=0.1)

        Example
        --------
        >>> # You can update penalties on the fly and predict again immediately.
        >>> # The default weight penalties values are 0.0.
        >>> wrapper.icp.weight_hamming = 2.0
        >>> wrapper.icp.weight_cardinality = 1.5
        >>> penalized_obj = wrapper.predict(X_test)
        >>> penalized_sets = penalized_obj(significance_level=0.1)
        """

        print("--- Starting Prediction ---")
        if self.icp is None:
            raise RuntimeError("Run fit procedure first.")

        try:
            check_is_fitted(self.strategy)
        except NotFittedError:
            raise RuntimeError("Classifier must be fitted.")

        if self.strategy_fingerprint != _fingerprint_model(self.strategy, self.kwargs):
            raise RuntimeError("Classifier model changed. Run the fit and calibration procedure.")

        if torch.is_tensor(test_features):
            test_features = check_array(test_features.detach().cpu().numpy(), accept_sparse=True, dtype=None, ensure_2d=True)
        else:
            test_features = check_array(test_features, accept_sparse=True, dtype=None, ensure_2d=True)

        test_probabilities = self.predict_proba_to_tensor(test_features)
        print("---The object of PredictionRegions class is returned.---\n")

        return self.icp.predict(test_probabilities.to(self.device))