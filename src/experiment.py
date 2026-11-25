"""Siatka eksperymentów: każdy wariant danych x każdy model -> jedna tabela wyników."""
import time

import numpy as np
import pandas as pd

from .config import RESULTS_DIR, TARGET, TEST_WEEKS
from .data import clean, load_raw, time_split
from .datasets import DATASETS
from .evaluation import metrics
from .models import GRID, MODELS, NEEDS_IMPUTE, save_model
from .storage import atomic_to_csv, locked

RESULTS_CSV = RESULTS_DIR / 'eksperymenty.csv'
PREDS_DIR = RESULTS_DIR / 'predykcje'
KEYS = ['dane', 'model']


def load_results(path=RESULTS_CSV):
    return pd.read_csv(path) if path.exists() else pd.DataFrame(columns=KEYS)


def _preds_path(dataset, model):
    return PREDS_DIR / f'{dataset}__{model}.csv'.replace(' ', '_')


def save_preds(dataset, model, test, pred):
    """Zapisuje predykcje na teście: Store, Dept, Date, IsHoliday, y_true, y_pred."""
    PREDS_DIR.mkdir(parents=True, exist_ok=True)
    out = test[['Store', 'Dept', 'Date', 'IsHoliday']].copy()
    out['y_true'] = test[TARGET].to_numpy()
    out['y_pred'] = pred
    atomic_to_csv(out, _preds_path(dataset, model))


def load_preds(dataset, model):
    """Predykcje zapisane przez run_experiments (wymaga wcześniejszego treningu tej pary)."""
    path = _preds_path(dataset, model)
    if not path.exists():
        raise FileNotFoundError(f'Brak predykcji {path.name} - uruchom run_experiments dla tej pary')
    return pd.read_csv(path, parse_dates=['Date'])


def prepare(variant, df_clean, test_weeks=TEST_WEEKS):
    """Zwraca (X_train, y_train, X_test, test) dla wariantu danych."""
    df = variant.build(df_clean)
    train, test = time_split(df, test_weeks, verbose=False)
    if variant.train_filter is not None:
        train = variant.train_filter(train)
    return train[variant.features], train[TARGET], test[variant.features], test


def run_experiments(datasets=None, models=None, df_clean=None, test_weeks=TEST_WEEKS,
                    force=False, save_models=False, path=RESULTS_CSV):
    """Trenuje wybrane modele na wybranych wariantach danych.

    datasets / models - listy nazw z rejestrów DATASETS / MODELS
    (None = wszystkie warianty danych / modele z listy GRID w models.py).
    Wyniki dopisywane są do CSV; pary (dane, model) już policzone są pomijane,
    chyba że force=True. Zwraca (tabela wyników, słownik predykcji na teście).
    """
    datasets = list(datasets or DATASETS)
    models = list(models or GRID)
    if df_clean is None:
        df_clean = clean(load_raw(), verbose=False)

    done = set() if force else set(map(tuple, load_results(path)[KEYS].values))
    preds, pending, failed = {}, [], []

    for d in datasets:
        todo = [m for m in models if (d, m) not in done]
        if not todo:
            print(f"[{d}] wszystko policzone - pomijam", flush=True)
            continue
        variant = DATASETS[d]
        X_train, y_train, X_test, test = prepare(variant, df_clean, test_weeks)
        med = X_train.median()

        for m in todo:
            Xtr, Xte = (X_train.fillna(med), X_test.fillna(med)) if m in NEEDS_IMPUTE else (X_train, X_test)
            try:
                model = MODELS[m]()
                t0 = time.time()
                model.fit(Xtr, y_train)
                fit_s = time.time() - t0
                pred = model.predict(Xte)
                if not np.all(np.isfinite(pred)):
                    raise ValueError('predykcje zawierają NaN/inf')
            except ImportError as e:
                print(f"[{d}] {m}: pominięto ({e})", flush=True)
                continue
            except Exception as e:   # jeden zepsuty model nie przerywa całej siatki
                print(f"[{d}] {m}: BŁĄD {type(e).__name__}: {e}", flush=True)
                failed.append((d, m, f'{type(e).__name__}: {e}'))
                continue

            preds[(d, m)] = pd.Series(pred, index=test.index)
            save_preds(d, m, test, pred)
            row = {'dane': d, 'model': m, **metrics(test[TARGET], pred, test['IsHoliday']),
                   'czas_s': round(fit_s, 1), 'n_cech': len(variant.features), 'n_train': len(Xtr)}
            print(f"[{d}] {m:22s} WMAE={row['WMAE']:9.1f}  R2={row['R2']:.4f}  ({fit_s:.0f}s)", flush=True)
            if save_models:
                save_model(model, f'{d}__{m}'.replace(' ', '_'))
            pending.append(row)
            pending = _flush_rows(pending, path)   # zapis po każdym modelu

    if pending:   # ostatnia deska ratunku - wyniki trafiają do osobnego pliku
        backup = path.with_name(f'{path.stem}_niezapisane_{int(time.time())}.csv')
        pd.DataFrame(pending).to_csv(backup, index=False)
        print(f"UWAGA: {len(pending)} wyników zapisano do {backup.name} zamiast {path.name}", flush=True)
    if failed:
        print(f"Modele zakończone błędem ({len(failed)}): " + '; '.join(f'{d}/{m}' for d, m, _ in failed), flush=True)
    results = load_results(path)
    return results.sort_values('WMAE').reset_index(drop=True), preds


def _flush_rows(rows, path):
    """Dopisuje wiersze do CSV pod blokadą (bezpieczne przy kilku procesach).
    Zwraca wiersze, których nie udało się zapisać - zostaną dopisane przy następnej próbie."""
    try:
        with locked(path):
            results = load_results(path)   # świeży odczyt - inny proces mógł coś dopisać
            new = pd.DataFrame(rows)
            keep = ~results.set_index(KEYS).index.isin(new.set_index(KEYS).index)
            atomic_to_csv(pd.concat([results[keep], new], ignore_index=True), path)
        return []
    except (PermissionError, TimeoutError) as e:
        print(f"UWAGA: nie udało się zapisać {path.name} ({e}) - ponowię przy następnym modelu", flush=True)
        return rows


FAMILIES = ['XGBoost', 'LightGBM', 'Sieć z embeddingami', 'Średnia']


def best_per_family(results, dataset, families=FAMILIES):
    """Najlepsza wersja każdej rodziny modeli w wariancie danych (domyślna / strojona /
    z wielu ziaren / średnia), posortowana wg WMAE. Rodzina = początek nazwy modelu."""
    r = results[results['dane'] == dataset]
    picks = [r[r['model'].str.startswith(f)].nsmallest(1, 'WMAE') for f in families]
    return pd.concat(picks).sort_values('WMAE')['model'].tolist()


def pivot(results, metric='WMAE'):
    """Tabela dane x model dla wybranej metryki."""
    return results.pivot(index='dane', columns='model', values=metric)


def plot_heatmap(results, metric='WMAE', ax=None):
    import matplotlib.pyplot as plt
    p = pivot(results, metric)
    ax = ax or plt.subplots(figsize=(1.6 * len(p.columns) + 3, 0.6 * len(p) + 2))[1]
    # skala kolorów przycięta do 10-90 percentyla - jedna skrajna wartość (np. regresja bez
    # historii) nie spłaszcza reszty; liczby w polach są zawsze dokładne
    vmin, vmax = np.nanpercentile(p.values, [10, 90])
    im = ax.imshow(p.values, cmap='viridis_r' if metric != 'R2' else 'viridis', aspect='auto',
                   vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(p.columns))); ax.set_xticklabels(p.columns, rotation=30, ha='right')
    ax.set_yticks(range(len(p))); ax.set_yticklabels(p.index)
    mid = (vmin + vmax) / 2
    for i in range(p.shape[0]):
        for j in range(p.shape[1]):
            v = p.values[i, j]
            if not np.isnan(v):
                dark = (v > mid) if metric != 'R2' else (v < mid)
                ax.text(j, i, f'{v:.3f}' if metric == 'R2' else f'{v:,.0f}'.replace(',', ' '),
                        ha='center', va='center', fontsize=9, color='white' if dark else 'black')
    ax.grid(False)
    ax.set_title(f'{metric}: wariant danych x model')
    ax.figure.colorbar(im, ax=ax, shrink=0.8, label=metric)
    return ax
