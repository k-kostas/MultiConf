import numpy as np
import pandas as pd
import torch
import warnings
import pickle
import hashlib

from typing import Union


# def _check_multihot_labels(y):
#     if hasattr(y, 'values'):
#         y_arr = y.values
#     else:
#         y_arr = np.array(y)
#     is_binary = np.all(np.isin(y_arr, [0, 1]))
#     if not is_binary:
#         raise ValueError("Labels must be binary (0 and 1).")
#
#     return y_arr

# def _check_multihot_labels(
#         labels: Union[torch.Tensor, np.ndarray, list, pd.DataFrame, pd.Series]
# ) -> torch.Tensor:
#     """
#     Checks if the labels are in a binary multi-hot format (containing only 0s and 1s).
#     Accepts integers (0, 1) and floats (0.0, 1.0).
#
#     Parameters
#     ----------
#     labels : Union[torch.Tensor, np.ndarray, list, pd.DataFrame, pd.Series]
#         The input labels to check and convert.
#
#     Returns
#     -------
#     torch.Tensor
#         The labels validated and converted to a float32 tensor.
#
#     Raises
#     ------
#     ValueError
#         If labels contain values other than 0 or 1.
#     """
#
#     if isinstance(labels, (pd.DataFrame, pd.Series)):
#         labels = torch.tensor(labels.values)
#     elif isinstance(labels, (np.ndarray, list)):
#         labels = torch.tensor(labels)
#     if not isinstance(labels, torch.Tensor):
#         labels = torch.tensor(np.array(labels))
#
#     unique_vals = torch.unique(labels)
#     for val in unique_vals:
#         if not (val == 0 or val == 1):
#             raise ValueError(f"Labels must be binary (0 and 1). Found value: {val}")
#
#     return labels.float()




def _check_multihot_labels(
        labels: Union[torch.Tensor, np.ndarray, list, pd.DataFrame, pd.Series]
) -> Union[torch.Tensor, np.ndarray, list, pd.DataFrame, pd.Series]:
    """
    Checks if the labels are in a binary multi-hot format (containing only 0s and 1s).

    This function validates the values WITHOUT forcing a conversion to PyTorch tensors.
    It returns the input object unmodified.

    Parameters
    ----------
    labels : Union[torch.Tensor, np.ndarray, list, pd.DataFrame, pd.Series]
        The input labels to check.

    Returns
    -------
    Union[torch.Tensor, np.ndarray, list, pd.DataFrame, pd.Series]
        The original labels object, validated.

    Raises
    ------
    ValueError
        If labels contain values other than 0 or 1.
    """

    if isinstance(labels, (pd.DataFrame, pd.Series)):
        vals = labels.values
    elif isinstance(labels, torch.Tensor):
        vals = labels
    else:
        vals = np.asarray(labels)

    if isinstance(vals, torch.Tensor):
        unique_vals = torch.unique(vals)
        unique_vals = unique_vals.detach().cpu().numpy()
    else:
        unique_vals = np.unique(vals)

    for val in unique_vals:
        if val != 0 and val != 1:
            raise ValueError(f"Labels must be binary (0 and 1). Found value: {val}")

    return labels


def _is_tensor(data: Union[torch.Tensor, np.ndarray, list, pd.DataFrame, pd.Series],
               dtype=torch.float32) -> torch.Tensor:
    """
    Converts input data to a PyTorch tensor.
    Handles PyTorch Tensors, NumPy arrays, Pandas DataFrame/Series,
    standard lists, and Scikit-Learn friendly array-likes.

    Parameters
    ----------
    data : array-like
        The input data to convert.
    dtype : torch.dtype, optional, default=torch.float32
        The desired data type of the output tensor.

    Returns
    -------
    torch.Tensor
        The converted PyTorch tensor.
    """

    if torch.is_tensor(data):
        return data.to(dtype=dtype)

    if isinstance(data, (pd.DataFrame, pd.Series)):
        data = data.values
    elif hasattr(data, "values"):
        data = data.values

    if not isinstance(data, np.ndarray):
        data = np.array(data)

    return torch.tensor(data, dtype=dtype)


def _normalize_device(dev: Union[torch.device, str, None]) -> torch.device:
    if dev is None:
        return torch.device("cpu")
    if isinstance(dev, str):
        dev = torch.device(dev)
    if dev.type == "cuda" and not torch.cuda.is_available():
        warnings.warn("CUDA requested but not available. Falling back to CPU.", RuntimeWarning)
        return torch.device("cpu")

    return dev


def _fingerprint_model(model, params) -> str:
    h = hashlib.sha1()

    if params:
        for k, v in sorted(params.items()):
            h.update(str(k).encode())
            h.update(repr(v).encode())
    try:
        if hasattr(model, "state_dict"):
            for name, t in model.state_dict().items():
                h.update(name.encode())
                if torch.is_tensor(t):
                    h.update(t.detach().cpu().numpy().tobytes())
                else:
                    h.update(str(t).encode())
        else:
            try:
                model_bytes = pickle.dumps(model)
                h.update(model_bytes)
            except (AttributeError, TypeError, pickle.PicklingError):
                h.update(str(model).encode())

    except Exception:
        return "unknown"

    return h.hexdigest()


# def _fingerprint_model(model) -> str:
#     h = hashlib.sha1()
#     try:
#         # PyTorch
#         if hasattr(model, "state_dict"):
#             for name, t in model.state_dict().items():
#                 h.update(name.encode())
#                 if torch.is_tensor(t):
#                     h.update(t.detach().cpu().numpy().tobytes())
#                 else:
#                     h.update(str(t).encode())
#         # Sklearn/Generic
#         else:
#             model_bytes = pickle.dumps(model)
#             h.update(model_bytes)
#     except Exception:
#         return "unknown"
#     return h.hexdigest()


# def _to_tensor(self, data) -> torch.Tensor:
#     if hasattr(data, 'values'):
#         data = data.values
#     if not torch.is_tensor(data):
#         data = torch.tensor(data, dtype=torch.float32)
#     return data.to(self._device, dtype=torch.float32)
