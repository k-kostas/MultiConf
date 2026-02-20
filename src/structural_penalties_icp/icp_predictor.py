import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from typing import Union

from .prediction_regions import PredictionRegions
from .utils import _check_multihot_labels, _is_tensor, _normalize_device

_BATCH_SIZE = 2048
InputData = Union[torch.Tensor, np.ndarray, list, pd.DataFrame, pd.Series]

class InductiveConformalPredictor:
    """
    Inductive Conformal Predictor with Structural Penalties.

    This class implements Inductive Conformal Prediction (ICP) for multi-label classification,
    extended with structural penalties (Hamming and Cardinality). It uses the Mahalanobis
    distance in the error vectors space to account for label correlations. Also, it allows
    for dynamic re-weighting of penalties without retraining the underlying model or
    recalculating the covariance matrix.

    .. note::
       The predictor calculates the Covariance Matrix based on the training set's error vectors,
       and pre-calculates the structural penalty vectors for all possible label combinations.

    .. note::
       **Memory Management**:
       This class processes large datasets in batches to avoid GPU/CPU memory overflow.
       For systems with 16GB of memory, we recommend limiting the task to a maximum of 25 labels.
       The internal batch size is controlled by the module-level constant ``_BATCH_SIZE`` (default: 2048).

    Parameters
    ----------
    predicted_probabilities : Union[torch.Tensor, np.ndarray, list, pd.DataFrame, pd.Series]
        The predicted probabilities for the proper training set.
        Shape: (n_samples, n_classes).
    true_labels : Union[torch.Tensor, np.ndarray, list, pd.DataFrame, pd.Series]
        The ground truth binary labels for the proper training set.
        Shape: (n_samples, n_classes).
    weight_hamming : float, optional, default=0.0
        The weight for the Hamming distance penalty. Higher values penalize predictions
        that are structurally different from observed training labels.
    weight_cardinality : float, optional, default=0.0
        The weight for the Cardinality penalty. Higher values penalize prediction set sizes
        that are infrequent in the training data.
    device : str or torch.device, optional, default='cpu'
        The device to use for computations ('cpu' or 'cuda').


    Raises
    ------
    ValueError
        If weights are negative.
    RuntimeError
        If `predicted_probabilities` shape does not match the number of classes.


    Examples
    --------
    >>> import torch
    >>>
    >>> # 1. Generate dummy training data (100 samples, 5 classes)
    >>> train_probs = torch.rand(100, 5)
    >>> train_labels = torch.randint(0, 2, (100, 5)).float()
    >>>
    >>> # 2. Initialize the predictor
    >>> icp = InductiveConformalPredictor(
    ...     predicted_probabilities=train_probs,
    ...     true_labels=train_labels,
    ...     weight_hamming=0.5,
    ...     weight_cardinality=0.3,
    ... )
    """

    def __init__(self,
                 predicted_probabilities : InputData,
                 true_labels : InputData,
                 weight_hamming: float = 0.0,
                 weight_cardinality: float = 0.0,
                 device: Union[str, torch.device] = 'cpu',
                 ):

        print(f'\nInitializing Inductive Conformal Predictor')

        if weight_hamming < 0:
            raise ValueError("The weight of hamming penalty cannot be negative.")

        if weight_cardinality < 0:
            raise ValueError("The weight of cardinality penalty cannot be negative.")

        self.device = _normalize_device(device)

        true_labels = _check_multihot_labels(true_labels)
        true_labels = _is_tensor(true_labels).to(self.device)

        predicted_probabilities = _is_tensor(predicted_probabilities).to(self.device)
        if predicted_probabilities.shape[1] != true_labels.shape[1]:
            raise RuntimeError("Proper train labels and probabilities must have the same number of columns.")

        self.n_classes = true_labels.shape[1]
        self._weight_hamming = weight_hamming
        self._weight_cardinality = weight_cardinality

        self.combinations = torch.cartesian_prod(
            *[torch.tensor([False, True], device=self.device)] * self.n_classes
        )

        self.proper_train_labels = true_labels

        self._hamming_penalties = None
        self._cardinality_penalties = None
        self._inverse_covariance_matrix = None

        self._mahalanobis_max_score = None
        self._calib_normalized_scores = None
        self._calib_indices = None
        self.sorted_calibration_scores = None

        self._update_weight_hamming = False
        self._update_weight_cardinality = False

        self.hamming_penalties_preprocessing(self.proper_train_labels)
        self.cardinality_penalties_preprocessing(self.proper_train_labels)
        self.covariance_matrix_preprocessing(predicted_probabilities, self.proper_train_labels)


    @property
    def weight_hamming(self) -> float:
        """
        Getter for the current Hamming penalty weight.
        """

        return self._weight_hamming

    @weight_hamming.setter
    def weight_hamming(self, value: float):
        """
        Set the Hamming penalty weight.

        Setting this property triggers a flag to recalculate calibration scores
        during the next prediction call without re-running the
        full calibration procedure.

        .. note::
            If switching from 0.0 to a positive value, Hamming penalties are recalculated.


        Returns
        -------
        float
            The current Hamming penalty weight.
        """

        if value < 0:
            raise ValueError("Hamming penalty weight cannot be negative.")

        if self._weight_hamming != value:
            print(f'\n---Updating weight for Hamming penalties---')
            if self._weight_hamming == 0 and value > 0:
                self._weight_hamming = value
                self._update_weight_hamming = True
                self.hamming_penalties_preprocessing(self.proper_train_labels)
                print(f"Hamming penalty weight updated to {value}.")
                print(f'Hamming penalties recalculated.')
            else:
                self._weight_hamming = value
                self._update_weight_hamming = True
                print(f"Hamming penalty weight updated to {value}.")


    @property
    def weight_cardinality(self) -> float:
        """
        Getter for the current Cardinality penalty weight.
        """

        return self._weight_cardinality

    @weight_cardinality.setter
    def weight_cardinality(self, value: float):
        """
        Set the Cardinality penalty weight.

        Setting this property triggers a flag to recalculate calibration scores
        during the next prediction call without re-running the
        full calibration procedure.

        .. note::
            If switching from 0.0 to a positive value, Cardinality penalties are recalculated.


        Returns
        -------
        float
            The current Cardinality penalty weight.
        """

        if value < 0:
            raise ValueError("Cardinality penalty weight cannot be negative.")

        if self._weight_cardinality != value:
            print(f'\n---Updating weight for Cardinality penalties---')
            if self._weight_cardinality == 0 and value > 0:
                self._weight_cardinality = value
                self._update_weight_cardinality = True
                self.cardinality_penalties_preprocessing(self.proper_train_labels)
                print(f"Cardinality penalty weight updated to {value}.")
                print(f'Cardinality penalties recalculated.')
            else:
                self._weight_cardinality = value
                self._update_weight_cardinality = True
                print(f"Cardinality penalty weight updated to {value}.")


    @torch.no_grad()
    def hamming_penalties_preprocessing(self, labels: torch.Tensor):
        """
        Calculates Hamming penalties for all possible label combinations.

        The penalty is defined as the minimum Hamming distance from a combination
        to any observed label vector in the provided labels.

        Parameters
        ----------
        labels : torch.Tensor
            The set of ground truth labels of the proper training set.
            Shape: (n_samples, n_classes).

        Examples
        --------
        >>> # 1. Generate dummy data (100 samples, 5 classes)
        >>> labels = torch.randint(0, 2, (100, 5)).float()
        >>> icp.hamming_penalties_preprocessing(labels)
        """

        if self._weight_hamming == 0:
            self._hamming_penalties = torch.zeros(self.combinations.shape[0], device=self.device)
        else:
            batch_size = _BATCH_SIZE
            min_distances_list = []
            iterator = range(0, self.combinations.shape[0], batch_size)
            if self.combinations.shape[0] > batch_size:
                iterator = tqdm(iterator, desc="Calculating Hamming Penalties")
            for i in iterator:
                comb_batch = self.combinations[i: i + batch_size]
                diffs = (comb_batch.unsqueeze(1) != labels.unsqueeze(0))
                batch_loss = torch.sum(diffs.float() / self.n_classes, dim=-1)
                batch_min_dists = torch.min(batch_loss, dim=1).values
                min_distances_list.append(batch_min_dists)
            self._hamming_penalties = torch.cat(min_distances_list)
        print("Hamming penalties calculated with shape:", self._hamming_penalties.shape)


    @torch.no_grad()
    def cardinality_penalties_preprocessing(self, labels: torch.Tensor):
        """
        Calculates Cardinality penalties based on label set size frequencies.

        Combinations with a cardinality (number of active labels) that appears frequently
        in the training data receive lower penalties.


        Parameters
        ----------
        labels : torch.Tensor
            The set of ground truth labels used to calculate size frequencies.
            Shape: (n_samples, n_classes).


        Examples
        --------
        >>> # Generate dummy data (100 samples, 5 classes)
        >>> labels = torch.randint(0, 2, (100, 5)).float()
        >>> icp.cardinality_penalties_preprocessing(labels)
        """

        if self._weight_cardinality == 0:
            self._cardinality_penalties = torch.zeros(self.combinations.shape[0], device=self.device)
        else:
            card_counts = torch.bincount(torch.sum(labels, dim=1).long(), minlength=self.n_classes + 1)
            total_counts = card_counts.sum()
            comb_cards = torch.sum(self.combinations.long(), dim=1)

            if total_counts > 0:
                self._cardinality_penalties = 1 - (card_counts[comb_cards] / total_counts)
            else:
                self._cardinality_penalties = torch.ones_like(comb_cards, dtype=torch.float)
        print("Cardinality penalties calculated with shape:", self._cardinality_penalties.shape)


    @torch.no_grad()
    def covariance_matrix_preprocessing(self, probabilities: torch.Tensor, labels: torch.Tensor):
        """
        Computes the Inverse Covariance Matrix of the error vectors
        (|Predicted Probabilities - Labels|) on the Proper Training Set.


        Parameters
        ----------
        probabilities : torch.Tensor
            Predicted probabilities for the proper training set.
        labels : torch.Tensor
            True labels for the proper training set.


        Examples
        --------
        >>> # Generate dummy data (100 samples, 5 classes)
        >>> probabilities = torch.rand(100, 5)
        >>> labels = torch.randint(0, 2, (100, 5)).float()
        >>> icp.covariance_matrix_preprocessing(probabilities, labels)
        """

        if probabilities.ndim == 1:
            probabilities = probabilities.unsqueeze(0)

        errors = torch.abs(probabilities - labels)
        covariance_matrix = torch.cov(errors.T)
        eigvalues, eigvectors = torch.linalg.eig(covariance_matrix)
        eigvalues_real = eigvalues.real
        eigvalues_real[eigvalues_real <= 0] = 1e-32

        self._inverse_covariance_matrix = torch.linalg.inv(
            eigvectors.real @ torch.diag(eigvalues_real) @ eigvectors.real.T
        ).to(device=self.device)

        ones = torch.ones(self.n_classes, device=self.device)
        self._mahalanobis_max_score = torch.sqrt(ones @ self._inverse_covariance_matrix @ ones).to(device=self.device)
        print("Covariance matrix calculated with shape:", self._inverse_covariance_matrix.shape)


    def _update_calibration_scores(self):
        """
        Updates and sorts calibration scores based on current penalty weights.
        Internal method called automatically by predict() or calibrate() if weights change.

        Raises
        ------
        RuntimeError
            If calibration scores are not initialized. Call calibrate() with calibration features probabilities and labels first.
        """

        if self._calib_normalized_scores is not None and self._calib_indices is not None:
            calibration_scores = self._calib_normalized_scores + \
                                 (self.weight_hamming * self._hamming_penalties[self._calib_indices]) + \
                                 (self.weight_cardinality * self._cardinality_penalties[self._calib_indices])

            self.sorted_calibration_scores, _ = torch.sort(calibration_scores, descending=True)
            self._update_weight_hamming = False
            self._update_weight_cardinality = False
            print("Calibration scores calculated with shape:", self.sorted_calibration_scores.shape)
        else:
            raise RuntimeError("Calibration scores are not initialized. Call calibrate() with calibration features probabilities and labels first.")


    @torch.no_grad()
    def calibrate(self, probabilities: InputData=None, labels:InputData=None):
        """
        Calibrates the predictor using a dedicated calibration set.

        This method computes nonconformity scores for the calibration data and
        sorts them to determine thresholds for future predictions.

        .. note::
            If called without arguments, it recalculates the calibration scores
            using the currently set penalty weights on the existing calibration data.


        Parameters
        ----------
        probabilities : Union[torch.Tensor, np.ndarray, list, pd.DataFrame, pd.Series]
            Predicted probabilities for the calibration set.
            Shape: (n_samples, n_classes).
        labels : Union[torch.Tensor, np.ndarray, list, pd.DataFrame, pd.Series]
            Ground truth labels for the calibration set.
            Shape: (n_samples, n_classes).


        Returns
        -------
        self : object
            The initialized and calibrated predictor object.


        Raises
        ------
        RuntimeError
            If one of `probabilities` or `labels` is provided but not the other.
        RuntimeError
            If `labels` shape does not match the number of classes.
        RuntimeError
            If `probabilities` shape does not match the number of classes.


        Examples
        --------
        >>> # 1. Generate dummy calibration data (25 samples, 5 classes)
        >>> calib_probs = torch.rand(25, 5)
        >>> calib_labels = torch.randint(0, 2, (25, 3)).float()
        >>>
        >>> # 2. Calibrate
        >>> icp.calibrate(calib_probs, calib_labels)


        Examples
        --------
        >>> # Optional: The `calibrate` method recalculates calibration scores after weight update.
        >>> icp.weight_hamming = 1.0
        >>> icp.weight_cardinality = 0.5
        >>> icp.calibrate()
        """

        if probabilities is not None and labels is not None:
            labels = _check_multihot_labels(labels)
            labels = _is_tensor(labels).to(self.device)
            if labels.shape[1] != self.n_classes:
                raise RuntimeError("Labels must have the same number of columns as the number of classes.")

            probabilities = _is_tensor(probabilities).to(self.device)
            if probabilities.shape[1] != self.n_classes:
                raise RuntimeError("Calibration labels and probabilities must have the same number of columns.")

            if probabilities.ndim == 1:
                probabilities = probabilities.unsqueeze(0)

            errors = torch.abs(probabilities - labels)
            mahalanobis_scores = torch.sqrt(torch.sum((errors @ self._inverse_covariance_matrix) * errors, dim=1))
            self._calib_normalized_scores = mahalanobis_scores / self._mahalanobis_max_score

            powers_desc = 2 ** torch.arange(labels.shape[1] - 1, -1, -1, device=self.device)
            self._calib_indices = (labels * powers_desc).sum(dim=1).long()

        elif (probabilities is None) != (labels is None):
            raise RuntimeError("Both 'probabilities' and 'labels' must be provided for calibration, or neither.")

        self._update_calibration_scores()

        return self


    def all_combinations_scoring(self, probabilities: torch.Tensor):
        """
        Computes nonconformity scores for a test sample against all possible combinations.


        Parameters
        ----------
        probabilities : torch.Tensor
            Predicted probabilities for the input sample.
            Shape: (n_samples, n_classes).


        Returns
        -------
        torch.Tensor
            A 1D tensor containing the calculated nonconformity scores for every
            possible label combination (2^n_classes) for the given test sample.
            Shape: (2^n_classes,)
        """

        # probabilities = _is_tensor(probabilities).to(self.device)
        if probabilities.ndim == 1:
            probabilities = probabilities.unsqueeze(0)

        errors = torch.abs(probabilities - self.combinations.float())

        mahalanobis_scores = torch.sqrt(torch.sum((errors @ self._inverse_covariance_matrix) * errors, dim=1))
        normalized_scores = mahalanobis_scores / self._mahalanobis_max_score

        return normalized_scores + \
            (self.weight_hamming * self._hamming_penalties) + \
            (self.weight_cardinality * self._cardinality_penalties)


    @torch.no_grad()
    def predict(self, probabilities: InputData) -> PredictionRegions:
        """
        Computes p-values for the test samples.

        This method calculates the p-value for every possible label combination
        based on the calibrated scores.


        Parameters
        ----------
        probabilities : torch.Tensor
            Predicted probabilities for the test set.
            Shape: (n_samples, n_classes).

        Returns
        -------
        PredictionRegions
            A callable object that wraps the p-values and combinations.
            You must call this object with a significance level to get the actual prediction sets.


        Raises
        ------
        RuntimeError
            If `calibrate` has not been called before `predict`.
        RuntimeError
            If `probabilities` shape does not match the number of classes.


        Examples
        --------
        >>> # Generate dummy test probabilities
        >>> test_probs = torch.rand(10, 5)
        >>>
        >>> # Get prediction regions object
        >>> prediction_obj = icp.predict(test_probs)
        >>>
        >>> # Extract prediction sets for significance level 0.1 (90% confidence)
        >>> prediction_sets = prediction_obj(significance_level=0.1)
        """

        if self.sorted_calibration_scores is None:
            raise RuntimeError("Model is not calibrated.")

        probabilities = _is_tensor(probabilities).to(self.device)
        if probabilities.shape[1] != self.n_classes:
            raise RuntimeError("Test set probabilities must have the same number of columns as the number of classes.")

        if self._update_weight_hamming or self._update_weight_cardinality:
            self._update_calibration_scores()

        cal_scores_ascending = torch.flip(self.sorted_calibration_scores.to(self.device), dims=[0])
        n_cal = len(cal_scores_ascending)

        all_scores_list = []
        for i in range(len(probabilities)):
            scores = self.all_combinations_scoring(probabilities[i])
            all_scores_list.append(scores)
        all_combinations_scores = torch.stack(all_scores_list)

        flat_scores = all_combinations_scores.view(-1)
        indices = torch.searchsorted(cal_scores_ascending, flat_scores, side='left')
        n_greater_equal = n_cal - indices
        p_values_flat = (n_greater_equal + 1).float() / (n_cal + 1)
        p_values = p_values_flat.view(all_combinations_scores.shape)

        return PredictionRegions(p_values, self.combinations)

    __call__ = predict