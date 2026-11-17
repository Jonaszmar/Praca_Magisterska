"""Metryki i zbieranie wyników modeli."""
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

results = {}


def wmae(y_true, y_pred, is_holiday):
    """Weighted MAE - metryka konkursu Kaggle: tygodnie świąteczne mają wagę 5."""
    weights = np.where(is_holiday == 1, 5, 1)
    return np.sum(weights * np.abs(y_true - y_pred)) / np.sum(weights)


def metrics(y_true, y_pred, is_holiday):
    """Słownik metryk dla jednej predykcji (bez efektów ubocznych)."""
    y_true, y_pred, is_holiday = np.asarray(y_true), np.asarray(y_pred), np.asarray(is_holiday)
    return {'WMAE': wmae(y_true, y_pred, is_holiday),
            'MAE': mean_absolute_error(y_true, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'R2': r2_score(y_true, y_pred)}


def evaluate(name, y_true, y_pred, is_holiday):
    """Liczy metryki, zapisuje je w `results[name]` i wypisuje wiersz podsumowania."""
    results[name] = metrics(y_true, y_pred, is_holiday)
    w, mae, rmse, r2 = results[name].values()
    print(f"{name:38s}  WMAE={w:10.1f}  MAE={mae:10.1f}  RMSE={rmse:10.1f}  R2={r2:6.3f}")
    return results[name]
