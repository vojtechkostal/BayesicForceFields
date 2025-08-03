import numpy as np


def mape(y_true: np.ndarray, y_trial: np.ndarray) -> np.ndarray:
    """Calculate the Mean Absolute Error."""
    y_true = np.asarray(y_true)
    y_trial = np.asarray(y_trial)
    return np.abs(y_true - y_trial) / y_true


def mape_fn(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    abs_diff = np.sum(np.abs(y_true - y_pred), axis=1)
    norm = np.sum(np.abs(y_true), axis=1) + np.sum(np.abs(y_pred), axis=1)

    return np.mean(abs_diff / norm)


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Calculate the Symmetric Mean Absolute Percentage Error."""
    abs_diff = np.abs(y_true - y_pred)
    norm = 0.5 * (np.abs(y_true) + np.abs(y_pred))
    return 100 * np.mean(abs_diff / norm)


def mape_function(
    x_true: np.ndarray,
    x_pred: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> float:
    """
    Computes the overlap error between two curves. If the x-domain
    of the true and predicted data differ, the predicted y-domain
    will be interpolated to match the true x-domain.

    Parameters
    ----------
    x_true : np.ndarray
        x-coordinates of the reference curve.
    x_pred : np.ndarray
        x-coordinates of the predicted curve.
    y_true : np.ndarray
        y-coordinates of the reference curve.
    y_pred : np.ndarray
        y-coordinates of the predicted curve.

    Returns
    -------
    error: float
        The overlap error between the two curves.
    """

    y_pred = np.interp(x_true, x_pred, y_pred)
    diff = np.abs(y_true - y_pred)
    norm = np.sum(y_true) + np.sum(y_pred)
    error = np.sum(diff) / norm
    return error


def score_fnc(fnc_true: dict, fnc_pred: dict) -> list:
    """Computes MAEs of pairs of functions."""
    errors = []
    for atomtype, (r_true, g_true) in fnc_true.items():
        r_pred = fnc_pred[atomtype][0]
        g_pred = fnc_pred[atomtype][1]
        mae = mape_function(r_true, r_pred, g_true, g_pred)
        errors.append(mae)
    return errors
