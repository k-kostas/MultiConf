MultiConf Package
================================

A flexible Python package for **Conformal Prediction (CP)** in **Multi-Label** classification tasks.
It implements the **Powerset Scoring** approach [3]_ using the **Mahalanobis
nonconformity measure** [1]_, and applies **Structural Penalties** —based on
Hamming distance and label-set cardinality— to respect the label correlations of the proper training data,
producing valid and informative prediction sets [2]_. Designed for efficiency, it handles
model training, calibration, and the dynamic update of structural penalty weights without the need for
retraining. This package bridges **Scikit-Learn** (for the underlying classifiers) and **PyTorch**
(for efficient tensor computations and GPU acceleration).

Key Features
------------
* **Multi-label Conformal Prediction**: Provides sets of label-sets with guaranteed coverage under the assumption of data exchangeability.
* **Powerset Scoring**: Uses the powerset of the label space to compute the conformal prediction regions.
* **Mahalanobis Nonconformity Measure**: Utilizes the Mahalanobis distance in the error vectors space to account for label correlations
* **Structural Penalties**: Incorporates label correlations via Hamming and Cardinality penalties to produce more informative prediction sets.
* **Dynamic Updates**: Update penalty weights on the fly **without retraining** the model or recalculating the covariance matrix.
* **Smart Strategy Switching**: Switch the underlying classifier (e.g., from :class:`~sklearn.ensemble.RandomForestClassifier` to :class:`~sklearn.neighbors.KNeighborsClassifier`) dynamically; the wrapper handles retraining automatically.
* **Scikit-Learn Compatible**: Wraps any sklearn multi-label classifier (e.g., :class:`~sklearn.multioutput.MultiOutputClassifier`, :class:`~sklearn.multioutput.ClassifierChain`).
* **GPU Support**: Offloads heavy matrix computations to CUDA devices.

References
----------
.. [1] Katsios, K., & Papadopoulos, H. (2024). Multi-label conformal prediction with a Mahalanobis distance nonconformity measure.
    *Proceedings of Machine Learning Research*, 230, 1-14.
.. [2] Katsios, K., & Papadopoulos, H. (2025). Incorporating Structural Penalties in Multi-label Conformal Prediction.
    *Proceedings of Machine Learning Research*, 266, 1-20.
.. [3] Papadopoulos, H. (2014). A cross-conformal predictor for multi-label classification. In *Artificial Intelligence
        Applications and Innovations: AIAI 2014 Workshops: CoPA, MHDW, IIVC, and MT4BD, Rhodes, Greece, September 19-21, 2014. Proceedings 10* (pp. 241–250). Springer.

.. toctree::
   :maxdepth: 2
   :caption: User Guide:

   getting_started


.. toctree::
   :maxdepth: 2
   :caption: Developer Reference:

   documentation

.. toctree::
   :maxdepth: 2
   :caption: Citing Package:

   citing

.. toctree::
   :maxdepth: 2
   :caption: References:

   references