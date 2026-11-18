"""Uśrednianie predykcji kilku modeli (ensemble) jako dodatkowy „model” w tabeli wyników.

Dla każdego wariantu danych bierzemy XGBoost, LightGBM i sieć z embeddingami - wersję
strojoną, jeśli jest policzona na tym wariancie, w przeciwnym razie domyślną - i uśredniamy
ich predykcje na teście z równymi wagami. Wagi NIE są dobierane na teście (to byłby przeciek).

Liczone z zapisanych predykcji (results/predykcje/), bez ponownego treningu, więc musi się
wykonać po zakończeniu siatki i strojenia (run_all.py robi to na końcu).
Model do zapisu (AveragingRegressor) buduje src/best.py, gdy średnia trafi do najlepszych.
"""
import numpy as np
import pandas as pd

from .config import TARGET
from .evaluation import metrics
from .experiment import RESULTS_CSV, _flush_rows, load_preds, load_results, save_preds

ENSEMBLE_NAME = 'Średnia XGB+LGBM+Sieć'
MEMBERS = ['XGBoost', 'LightGBM', 'Sieć z embeddingami']
TUNED = ' (strojony)'


def members_for(dataset, results):
    """Nazwy składników średniej dla wariantu danych; None, jeśli któregoś brakuje."""
    done = set(map(tuple, results[['dane', 'model']].values))
    out = []
    for m in MEMBERS:
        if (dataset, m + TUNED) in done:
            out.append(m + TUNED)
        elif (dataset, m) in done:
            out.append(m)
        else:
            return None
    return out


def add_ensembles(path=RESULTS_CSV):
    """Dopisuje do tabeli wyników średnią dla każdego wariantu danych (nadpisuje poprzednią)."""
    results = load_results(path)
    rows = []
    for d in sorted(results['dane'].unique()):
        names = members_for(d, results)
        if names is None:
            print(f'[{d}] {ENSEMBLE_NAME}: brak któregoś ze składników {MEMBERS} - pomijam', flush=True)
            continue
        preds = [load_preds(d, m) for m in names]
        keys = preds[0][['Store', 'Dept', 'Date']].values
        if not all((p[['Store', 'Dept', 'Date']].values == keys).all() for p in preds[1:]):
            print(f'[{d}] {ENSEMBLE_NAME}: predykcje składników nie są wyrównane - pomijam', flush=True)
            continue
        test = preds[0].rename(columns={'y_true': TARGET})
        pred = np.mean([p['y_pred'].to_numpy() for p in preds], axis=0)
        save_preds(d, ENSEMBLE_NAME, test, pred)

        part = results[(results['dane'] == d) & results['model'].isin(names)]
        row = {'dane': d, 'model': ENSEMBLE_NAME, **metrics(test[TARGET], pred, test['IsHoliday']),
               'czas_s': round(part['czas_s'].sum(), 1), 'n_cech': part['n_cech'].iat[0],
               'n_train': part['n_train'].iat[0]}
        best_single = part['WMAE'].min()
        print(f'[{d}] {ENSEMBLE_NAME:22s} WMAE={row["WMAE"]:9.1f}  (najlepszy składnik {best_single:.1f}; '
              f'{" + ".join(names)})', flush=True)
        rows.append(row)
    if rows and _flush_rows(rows, path):
        raise RuntimeError(f'Nie udało się zapisać średnich do {path}')
    return pd.DataFrame(rows)


class AveragingRegressor:
    """Wytrenowane modele + średnia ich predykcji. `fill` - mediany do uzupełniania braków
    (None dla modeli, które przyjmują NaN). Do zapisu joblib przez src/best.py."""

    def __init__(self, estimators, names, fill=None):
        self.estimators = list(estimators)
        self.names = list(names)
        self.fill = list(fill) if fill is not None else [None] * len(self.estimators)

    def predict(self, X):
        return np.mean([e.predict(X if f is None else X.fillna(f))
                        for e, f in zip(self.estimators, self.fill)], axis=0)

    def __repr__(self):
        return f'AveragingRegressor({self.names})'
