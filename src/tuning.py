"""Strojenie hiperparametrów: losowe przeszukiwanie + walidacja czasowa na zbiorze TRENINGOWYM.

Zbiór testowy (ostatnie TEST_WEEKS tygodni) nie jest tu w ogóle używany - wybór
hiperparametrów na teście zawyżałby wyniki. Walidacja to kolejne okna czasowe
z końca treningu (expanding window), np. dla n_folds=3 i val_weeks=8:

    fold 1: trening [start .. T-24)  walidacja [T-24 .. T-16)
    fold 2: trening [start .. T-16)  walidacja [T-16 .. T-8)
    fold 3: trening [start .. T-8)   walidacja [T-8 .. T)

Użycie:
    trials = tune('LightGBM', dataset='bazowy', n_iter=30)
    register_tuned('LightGBM', 'bazowy')      # dodaje 'LightGBM (strojony)' do MODELS
    run_experiments(models=['LightGBM', 'LightGBM (strojony)'])
"""
import hashlib
import json
import time
from functools import partial

import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.model_selection import ParameterSampler

from .config import RANDOM_STATE, RESULTS_DIR, TARGET, TEST_WEEKS
from .data import clean, load_raw, time_split
from .datasets import DATASETS
from .evaluation import wmae
from .models import MODELS, NEEDS_IMPUTE
from .storage import atomic_to_csv

TUNING_DIR = RESULTS_DIR / 'strojenie'


def _free_gpu():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass

# Przestrzenie poszukiwań: listy = wybór z listy, rozkłady scipy = losowanie z rozkładu.
SPACES = {
    # Zakresy poszerzone po przebiegu z 18.11.2026 - najlepsze próby leżały na granicach:
    # XGBoost max_depth=12 (górna), LightGBM colsample_bytree~0.50 (dolna), num_leaves 217-230.
    'LightGBM': dict(
        n_estimators=randint(300, 2500), learning_rate=loguniform(0.01, 0.2),
        num_leaves=randint(31, 512), min_child_samples=randint(2, 100),
        subsample=uniform(0.6, 0.4), colsample_bytree=uniform(0.3, 0.7),
        reg_lambda=loguniform(1e-3, 10)),
    'XGBoost': dict(
        n_estimators=randint(300, 2500), learning_rate=loguniform(0.01, 0.2),
        max_depth=randint(4, 17), min_child_weight=loguniform(1, 50),
        subsample=uniform(0.6, 0.4), colsample_bytree=uniform(0.3, 0.7),
        reg_lambda=loguniform(1e-3, 10)),
    'HistGradientBoosting': dict(
        max_iter=randint(300, 1500), learning_rate=loguniform(0.01, 0.2),
        max_leaf_nodes=randint(31, 256), min_samples_leaf=randint(5, 100),
        l2_regularization=loguniform(1e-3, 10)),
    'Random Forest': dict(   # drogi: ~kilka minut na jedno dopasowanie
        n_estimators=randint(100, 400), max_depth=[12, 18, 24, None],
        min_samples_leaf=randint(1, 10), max_features=[0.3, 0.5, 0.8, 1.0]),
    'MLP': dict(
        # architektura sieci jest hiperparametrem jak każdy inny
        hidden=[(64,), (128,), (128, 64), (256, 128), (256, 128, 64),
                (512, 256, 128), (256, 256, 256), (512, 256, 128, 64)],
        activation=['relu', 'tanh'],
        alpha=loguniform(1e-6, 1e-2),              # regularyzacja L2
        learning_rate_init=loguniform(1e-4, 1e-2),
        batch_size=[256, 512, 1024, 2048]),
    # Poszerzone: obie wygrane próby miały hidden=(256, 256, 256) (najszersza „płaska”),
    # emb_dim_max=32 (maksimum) i weight_decay blisko górnej granicy 1e-3.
    'Sieć z embeddingami': dict(
        hidden=[(128, 64), (256, 128), (256, 128, 64), (512, 256, 128),
                (512, 256, 128, 64), (256, 256, 256), (512, 512, 512), (256, 256, 256, 256)],
        activation=['relu', 'gelu', 'silu'],
        dropout=[0.0, 0.1, 0.2, 0.3],
        emb_dim_max=[4, 8, 16, 32, 64],
        loss=['wmae', 'huber'],
        lr=loguniform(3e-4, 3e-3),
        weight_decay=loguniform(1e-6, 1e-2),
        batch_size=[512, 1024, 2048]),
}


def time_folds(dates, n_folds=3, val_weeks=TEST_WEEKS):
    """Indeksy (train, val) dla kolejnych okien czasowych z końca szeregu."""
    dates = pd.Series(pd.to_datetime(dates)).reset_index(drop=True)
    weeks = np.sort(dates.unique())
    if len(weeks) <= n_folds * val_weeks:
        raise ValueError('Za mało tygodni na tyle foldów walidacyjnych')
    for k in range(n_folds, 0, -1):
        start = weeks[-k * val_weeks]
        end = weeks[-(k - 1) * val_weeks] if k > 1 else None
        val = (dates >= start) & ((dates < end) if end is not None else True)
        yield np.flatnonzero(dates < start), np.flatnonzero(val)


def _to_builtin(v):
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, tuple):
        return list(v)
    return v


def space_signature(space, fixed=None):
    """Stały opis przestrzeni (repr rozkładów scipy zawiera adres w pamięci - nie nadaje się)."""
    def desc(v):
        if hasattr(v, 'dist'):
            return f'{v.dist.name}{tuple(v.args)}{sorted(v.kwds.items())}'
        return repr(v)
    parts = [f'{k}={desc(space[k])}' for k in sorted(space)]
    parts += [f'fixed.{k}={v!r}' for k, v in sorted((fixed or {}).items())]
    return hashlib.md5(';'.join(parts).encode()).hexdigest()[:10]


def _paths(model, dataset):
    stem = f'{dataset}__{model}'.replace(' ', '_')
    return TUNING_DIR / f'{stem}.csv', TUNING_DIR / f'{stem}.json'


def tune(model, dataset='bazowy', n_iter=20, n_folds=3, val_weeks=TEST_WEEKS,
         space=None, fixed=None, df_clean=None, seed=RANDOM_STATE):
    """Losowe przeszukiwanie hiperparametrów modelu `model` (nazwa z MODELS).

    Wynik każdej próby to średni WMAE po foldach. Próby są zapisywane do CSV na bieżąco;
    ponowne wywołanie z tymi samymi argumentami dokańcza przerwane przeszukiwanie.
    `fixed` - parametry stałe dla wszystkich prób (np. max_iter=20 do szybkiego testu).
    Zwraca tabelę prób posortowaną od najlepszej.
    """
    space = space or SPACES[model]
    fixed = fixed or {}
    if df_clean is None:
        df_clean = clean(load_raw(), verbose=False)

    variant = DATASETS[dataset]
    train, _ = time_split(variant.build(df_clean), TEST_WEEKS, verbose=False)  # test odrzucony
    if variant.train_filter is not None:
        train = variant.train_filter(train)
    train = train.reset_index(drop=True)
    X, y, hol = train[variant.features], train[TARGET].to_numpy(), train['IsHoliday'].to_numpy()
    folds = list(time_folds(train['Date'], n_folds, val_weeks))

    csv_path, json_path = _paths(model, dataset)
    TUNING_DIR.mkdir(parents=True, exist_ok=True)
    trials = pd.read_csv(csv_path) if csv_path.exists() else pd.DataFrame()
    sig = f'{space_signature(space, fixed)}_s{seed}_n{n_iter}_f{n_folds}'
    if len(trials) and set(trials.get('przestrzen', pd.Series(dtype=str)).astype(str)) != {sig}:
        # próby z innej przestrzeni / liczby prób / foldów nie mogą się mieszać z nowymi
        old = csv_path.with_name(f'{csv_path.stem}__stara_przestrzen_{time.strftime("%Y%m%d_%H%M%S")}.csv')
        csv_path.replace(old)
        print(f'[{dataset}/{model}] zmieniona przestrzeń strojenia - stare próby przeniesione do {old.name}, '
              f'zaczynam od nowa', flush=True)
        trials = pd.DataFrame()
    candidates = list(ParameterSampler(space, n_iter, random_state=seed))

    for i, params in enumerate(candidates):
        if i < len(trials):
            continue   # już policzone w poprzednim uruchomieniu
        t0, scores, error = time.time(), [], ''
        try:
            for tr, va in folds:
                Xtr, Xva = X.iloc[tr], X.iloc[va]
                if model in NEEDS_IMPUTE:
                    med = Xtr.median()
                    Xtr, Xva = Xtr.fillna(med), Xva.fillna(med)
                est = MODELS[model](**{**params, **fixed})   # `fixed` nadpisuje wylosowane
                est.fit(Xtr, y[tr])
                score = wmae(y[va], est.predict(Xva), hol[va])
                if not np.isfinite(score):
                    raise ValueError('WMAE = NaN/inf (model się rozbiegł)')
                scores.append(score)
        except Exception as e:   # zła kombinacja parametrów / brak pamięci GPU - idziemy dalej
            error, scores = f'{type(e).__name__}: {e}'[:300], []
            _free_gpu()
        row = {'proba': i, 'WMAE_sr': np.mean(scores) if scores else np.nan,
               'WMAE_std': np.std(scores) if scores else np.nan,
               **{f'fold{j+1}': s for j, s in enumerate(scores)},
               'czas_s': round(time.time() - t0, 1),
               'params': json.dumps({k: _to_builtin(v) for k, v in params.items()}),
               'blad': error, 'przestrzen': sig}
        trials = pd.concat([trials, pd.DataFrame([row])], ignore_index=True)
        atomic_to_csv(trials, csv_path)
        status = f"BŁĄD {error}" if error else f"WMAE={row['WMAE_sr']:9.1f} (±{row['WMAE_std']:.0f})"
        print(f"[{dataset}/{model}] próba {i+1}/{n_iter}: {status}  ({row['czas_s']:.0f}s)  "
              f"{row['params']}", flush=True)

    trials = trials.sort_values('WMAE_sr', na_position='last').reset_index(drop=True)
    if trials['WMAE_sr'].isna().all():
        raise RuntimeError(f'Wszystkie próby strojenia {model} zakończyły się błędem - '
                           f'szczegóły w kolumnie "blad" w {csv_path}')
    best = {**json.loads(trials.loc[0, 'params']), **fixed}
    tmp = json_path.with_suffix('.tmp')
    tmp.write_text(json.dumps(best, indent=2, ensure_ascii=False), encoding='utf-8')
    tmp.replace(json_path)
    print(f"Najlepsze ({model}, {dataset}): WMAE={trials.loc[0, 'WMAE_sr']:.1f}  {best}")
    return trials


def best_params(model, dataset='bazowy'):
    path = _paths(model, dataset)[1]
    params = json.loads(path.read_text(encoding='utf-8'))
    for k in ('hidden', 'activation', 'dropout'):   # JSON zamienia krotki na listy
        if isinstance(params.get(k), list):
            params[k] = tuple(params[k])
    return params


def register_tuned(model, dataset='bazowy', name=None):
    """Dodaje do MODELS wersję modelu z najlepszymi parametrami, np. 'LightGBM (strojony)'."""
    name = name or f'{model} (strojony)'
    MODELS[name] = partial(MODELS[model], **best_params(model, dataset))
    if model in NEEDS_IMPUTE:
        NEEDS_IMPUTE.add(name)
    return name
