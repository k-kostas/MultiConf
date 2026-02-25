Table of Contents
- [Key Features](#key-features)
- [Installation](#installation)
- [Documentation](#documentation)
- [Quickstart](#quickstart)
- [Alternative Usage](#alternative-usage)
- [Examples](#examples)
- [Citing Structural Penalties ICP](#citing-structural-penalties-icp)
- [References](#references)

# Structural Penalties ICP

A flexible Python package for **Conformal Prediction (CP)** in multi-label classification tasks.
It implements the **Powerset Scoring** approach [[3]](#papadopoulos2014) using the **Mahalanobis 
nonconformity measure** [[1]](#katsios2024), and applies **Structural Penalties** —based on 
Hamming distance and label-set cardinality— to respect the label correlations of the proper training data, 
producing valid and informative prediction sets [[2]](#katsios2025). Designed for efficiency, it handles 
model training, calibration, and the dynamic update of structural penalty weights without the need for 
retraining. This package bridges **Scikit-Learn** (for the underlying classifiers) and **PyTorch** 
(for efficient tensor computations and GPU acceleration).

## Key Features

* **Multi-label Conformal Prediction**: Provides sets of label-sets with guaranteed coverage under the assumption of data exchangeability.
* **Powerset Scoring**: Uses the powerset of the label space to compute the conformal prediction regions.
* **Mahalanobis Nonconformity Measure**: Utilizes the Mahalanobis distance in the error vectors space to account for label correlations.
* **Structural Penalties**: Incorporates label correlations via Hamming and Cardinality penalties to produce more informative prediction sets.
* **Dynamic Updates**: Update penalty weights on the fly **without retraining** the model or recalculating the covariance matrix.
* **Smart Strategy Switching**: Switch the underlying classifier (e.g., from Random Forest to KNN) dynamically; the wrapper handles retraining automatically.
* **Scikit-Learn Compatible**: Wraps any sklearn multi-label classifier (e.g., `MultiOutputClassifier`, `ClassifierChain`).
* **GPU Support**: Offloads heavy matrix computations to CUDA devices.


[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: BSD 3-Clause](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/bsd-3-clause)

## Installation

```bash
pip install structural-penalties-icp
```

## Documentation
For the complete documentation see [structuralpenaltiesicp.readthedocs.io](https://structuralpenaltiesicp.readthedocs.io/en/latest/)


## Quickstart
This guide demonstrates the core usage of the Structural Penalties ICP package for a multi-label classification task 
to produce prediction sets for a new test sample in different significance levels. 

We will load the data,
split it into proper training, calibration and test sets, train the model and evaluate the conformal predictions.
For example, we will use the **Yeast** dataset after we have preprocessed the data into features and labels
in CSV format. The labels are represented as **multi-hot vectors**.

```python
import pandas as pd
from sklearn.model_selection import train_test_split

# 1. Define the path to your data
data_path = "/data/yeast"

# 2. Load the Yeast dataset (Features and Labels)
X = pd.read_csv(f"{data_path}/X_yeast.csv")
y = pd.read_csv(f"{data_path}/y_yeast.csv")

# 3. Split the data
# First, separate out the Test set (10%)
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Then, split the remaining data into Proper Train and Calibration (30%)
X_train, X_calib, y_train, y_calib = train_test_split(X_temp, y_temp, test_size=0.3, random_state=42)
```

```text
Loading Yeast dataset...
Data shapes: Train=(1522, 103), Calib=(653, 103), Test=(242, 103)
```

We initialize the underlying classifier from Scikit-Learn before fitting it on the proper training data. We have
chosen the Random Forest classifier here, wrapped by MultiOutputClassifier. Then, we initialize the ICPWrapper
setting the model and the weights of the structural penalties (default values are 0.0). Notice that there are two ways
to adjust the classifiers' arguments either by passing them directly

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier

from structural_penalties_icp.icp_wrapper import ICPWrapper

base_model = MultiOutputClassifier(RandomForestClassifier(n_estimators=10))
wrapper = ICPWrapper(base_model, weight_hamming=2.0, weight_cardinality=1.5, device='cpu')
wrapper.fit(X_train, y_train)
```

or as a dictionary.   

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier

from structural_penalties_icp.icp_wrapper import ICPWrapper

base_model = MultiOutputClassifier(RandomForestClassifier())
wrapper = ICPWrapper(base_model, weight_hamming=2.0, weight_cardinality=1.5, device='cpu')
args = {'estimator__n_estimators': 5}
wrapper.fit(X_train, y_train, **args)
```

Once the model is fitted, the next step is calibration. This process uses the calibration set to compute
nonconformity scores, which are essential for calculating the p-values required to produce valid prediction regions.

```python
wrapper.calibrate(X_calib, y_calib)
```

> [!NOTE]
> **Switching Underlying Scikit-Learn Strategies** :
> You can switch the classification strategy or update its parameters dynamically. If the wrapper detects a change (via fingerprinting) during calibration, it will automatically retrain the new model on the cached proper training data.
>
> ```python
> from sklearn.neighbors import KNeighborsClassifier
> from sklearn.multioutput import ClassifierChain
>
> # Switch strategy to Classifier Chains with KNN
> wrapper.strategy = ClassifierChain(KNeighborsClassifier())
> wrapper.kwargs = {'estimator__n_neighbors': 5}
>
> # Trigger automatic retraining and calibration
> wrapper.calibrate(X_calib, y_calib)
> ```

Finally, we generate prediction regions for the test set using the predict method.

```python
prediction_regions_obj = wrapper.predict(X_test)
```

The predict method returns a PredictionRegions container holding the conformal prediction regions for each sample.
You can query this object to extract valid label sets at a specific significance level
(e.g., $\alpha=0.1$ for 90% confidence) or multiple levels (e.g., $\alpha=[0.05, 0.1, 0.2]$).

The label-sets are returned as multi-hot vectors. In the example below, we retrieve the valid label combinations
for the first sample in the test set.

```python
prediction_sets = prediction_regions_obj(significance_level=0.1)
print(prediction_sets[0])
```

```text
tensor([[0, 0, 0,  ..., 1, 1, 0],
        [0, 0, 0,  ..., 1, 0, 0],
        [0, 0, 0,  ..., 1, 1, 0],
        ...,
        [1, 1, 1,  ..., 1, 1, 0],
        [1, 1, 1,  ..., 0, 0, 0],
        [1, 1, 1,  ..., 1, 1, 0]], dtype=torch.int32)
```

Equivalent one-liner:

```python
prediction_sets = wrapper.predict(X_test)(significance_level=0.1)
```

> [!NOTE]
> **Dynamic Penalties Weights Update**: We update the penalty weights dynamically without retraining the model.
>
> ```python
> wrapper.icp.weight_hamming = 1.5
> wrapper.icp.weight_cardinality = 0.5
>
> # Predict with new penalties
> updated_prediction_sets = wrapper.predict(X_test)(significance_level=0.1)
>```

> [!NOTE]
> **Accessing P-Values**: You also have direct access to the raw p-values for every possible label combination.
> Below, we print the p-values for the first test sample.
>
> ```python
> print(prediction_regions_obj.p_values[0])
>```
>
> ```text
> tensor([0.0627, 0.0015, 0.0719,  ..., 0.0015, 0.0015, 0.0015])
> ```


The `evaluate` method provides a convenient way to calculate performance metrics, including Coverage, 
N-Criterion, S-Criterion, and statistical validity via the KS-test. Additionally, it can return the p-values
corresponding to the true labels.

The method requires the **ground truth labels** (`true_labelsets`) and the desired **significance level**.
All other metric-specific arguments are optional boolean flags, which default to `True` if not specified.

```python
metrics = prediction_regions_obj.evaluate(
    return_true_label_p_value = False,
    return_coverage=True,
    return_n_criterion=True,
    return_s_criterion=True,
    return_ks_test=True,
    true_labelsets=y_test,
    significance_level=0.1,
)

print(metrics)
```

```text
{
'coverage': 0.9008264462809917,
 'n_criterion': 858.8636363636364,
 's_criterion': 412.99029541015625,
 'ks_test_metrics': {
                    'ks_statistic': np.float64(0.05622110017075027),
                    'ks_p_value': np.float64(0.4135919018220534),
                    'is_valid': np.True_
                    }
 }
```


## Alternative usage
You can also use the InductiveConformalPredictor class as a standalone engine if you prefer to manage the underlying
classifier yourself or not using Scikit-Learn. In this mode, you must provide the **predicted probabilities** for the
proper training, calibration, and test sets, as well as the **ground truth labels** for the training and calibration sets.

The package is flexible regarding input formats: it accepts PyTorch Tensors, NumPy arrays, Pandas DataFrames/Series,
or lists. All data is automatically converted to tensors and moved to the specified device (CPU or GPU) for 
efficient processing.

First, we need to initialize the InductiveConformalPredictor class to calculate the structural penalties and to form
the covariance matrix using the proper training data.

```python
from structural_penalties_icp.icp_predictor import InductiveConformalPredictor

icp = InductiveConformalPredictor(
    predicted_probabilities=train_probs,
    true_labels=train_labels,
    weight_hamming=1.5,  
    weight_cardinality=0.5,
    device='cpu'
)
```

Next, we call the `calibrate` method to calculate the calibration scores based on the calibration probabilities
and labels.

```python
icp.calibrate(probabilities=calib_probs,labels=calib_labels)
```
Then, we can generate predictions regions for the test set by calling the `predict` method and passing the test
probabilities.

```python
prediction_regions_obj = icp.predict(test_probs)
```

The predict method returns a PredictionRegions container holding the conformal prediction regions. You can extract
valid label sets at a specific significance level (e.g., $\alpha=0.1$ for 90% confidence) or multiple levels
(e.g., $\alpha=[0.05, 0.1, 0.2]$). In the example below, we print the prediction regions for the first sample
in the test set.

```python
prediction_sets = prediction_regions_obj(significance_level=0.1)
print(prediction_sets[0])
```

```text
tensor([[0, 0, 0,  ..., 1, 1, 0],
        [0, 0, 0,  ..., 1, 0, 0],
        [0, 0, 0,  ..., 1, 1, 0],
        ...,
        [1, 1, 1,  ..., 1, 1, 0],
        [1, 1, 1,  ..., 0, 0, 0],
        [1, 1, 1,  ..., 1, 1, 0]], dtype=torch.int32)
```

And of course, we have access to the p-values. In the example below, we get the p-values of the first sample in the
test set.

```python
print(prediction_regions_obj.p_values[0])
```

```text
tensor([0.0627, 0.0015, 0.0719,  ..., 0.0015, 0.0015, 0.0015])
```

Also, it allows us to get the p-values of test set's true labels and evaluate metrics like Coverage, N-Criterion,
S-Criterion and KS-test.

```python
metrics = prediction_regions_obj.evaluate(
    return_true_label_p_value = False,
    return_coverage=True,
    return_n_criterion=True,
    return_s_criterion=True,
    return_ks_test=True,
    true_labelsets=test_labels,
    significance_level=0.1,
)

print(metrics)
```

```text
{
'coverage': 0.9008264462809917,
 'n_criterion': 858.8636363636364,
 's_criterion': 412.99029541015625,
 'ks_test_metrics': {
                    'ks_statistic': np.float64(0.05622110017075027),
                    'ks_p_value': np.float64(0.4135919018220534),
                    'is_valid': np.True_
                    }
 }
```

> [!NOTE]
> **Dynamic Penalties Weights Update**: We update the penalty weights dynamically without retraining the model.
>
> ```python
> wrapper.icp.weight_hamming = 1.5
> wrapper.icp.weight_cardinality = 0.5
>
> # Predict with new penalties
> updated_prediction_sets = wrapper.predict(X_test)(significance_level=0.1)
> ```


## Examples

For additional examples of how to use the package, see the [documentation](https://structuralpenaltiesicp.readthedocs.io/en/latest/documentation.html#).


## Citing Structural Penalties ICP

If you use the package for a scientific publication, you are kindly requested to cite the following paper:

> <a id="katsios2025"></a>Katsios, K., & Papadopoulos, H. (2025). Incorporating Structural Penalties in Multi-label Conformal Prediction.
> *Proceedings of Machine Learning Research*, 266, 1-20.
[[Download PDF](https://raw.githubusercontent.com/mlresearch/v266/main/assets/katsios25a/katsios25a.pdf)]

**BibTeX:**

```bibtex
@article{katsios2025incorporating,
  title={Incorporating Structural Penalties in Multi-label Conformal Prediction},
  author={Katsios, Kostas and Papadopoulos, Harris},
  journal={Proceedings of Machine Learning Research},
  volume={266},
  pages={1--20},
  year={2025}
}
```


## References

1. <a id="katsios2025"></a>Katsios, K., & Papadopoulos, H. (2025). Incorporating Structural Penalties in Multi-label Conformal Prediction.
    *Proceedings of Machine Learning Research*, 266, 1-20.

2. <a id="katsios2024"></a>Katsios, K., & Papadopoulos, H. (2024). Multi-label conformal prediction with a Mahalanobis distance nonconformity measure.
    *Proceedings of Machine Learning Research*, 230, 1-14.

3. <a id="papadopoulos2014"></a>Papadopoulos, H. (2014). A cross-conformal predictor for multi-label classification. In *Artificial Intelligence Applications and Innovations: AIAI 2014 Workshops: CoPA, MHDW, IIVC, and MT4BD, Rhodes, Greece, September 19-21, 2014. Proceedings 10* (pp. 241–250). Springer.

4. <a id="lambrou2016"></a>Lambrou, A., & Papadopoulos, H. (2016). Binary relevance multi-label conformal predictor. In *Conformal and Probabilistic Prediction with Applications* (pp. 90–104). Springer.

5. <a id="maltou2022"></a>Maltoudoglou, L., Paisios, A., Lenc, L., Martı́nek, J., Král, P., & Papadopoulos, H. (2022). Well-calibrated confidence measures for multi-label text classification with a large number of labels. *Pattern Recognition*, 122, 108271.

6. <a id="papadopoulos2002a"></a>Papadopoulos, H., Proedrou, K., Vovk, V., & Gammerman, A. (2002a). Inductive confidence machines for regression. In *Machine learning: ECML 2002: 13th European conference on machine learning Helsinki, Finland, August 19–23, 2002 proceedings 13* (pp. 345–356). Springer.

7. <a id="papadopoulos2002b"></a>Papadopoulos, H., Vovk, V., & Gammerman, A. (2002b). Qualified prediction for large data sets in the case of pattern recognition. In *ICMLA* (pp. 159–163).

8. <a id="vovk2005"></a>Vovk, V., Gammerman, A., & Shafer, G. (2005). *Algorithmic Learning in a Random World* (Vol. 29). Springer.

9. <a id="vovk2016"></a>Vovk, V., Fedorova, V., Nouretdinov, I., & Gammerman, A. (2016). Criteria of efficiency for conformal prediction. In *Conformal and Probabilistic Prediction with Applications: 5th International Symposium, COPA 2016, Madrid, Spain, April 20-22, 2016, Proceedings 5* (pp. 23–39). Springer.