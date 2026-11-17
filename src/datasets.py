"""Rejestr wariantów danych do eksperymentów.

Każdy wariant to przepis: jak z oczyszczonej ramki zbudować cechy i których kolumn
użyć w modelu. Nowy wariant dodaje się jednym wywołaniem `register(...)` - tutaj
albo bezpośrednio w notebooku.

Zbiór testowy jest zawsze ten sam (ostatnie tygodnie), więc wyniki wariantów są
porównywalne. Filtrowanie wierszy (`train_filter`) działa tylko na treningu.
"""
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from .config import FEATURE_COLS, HIST_COLS, MD_COLS, TARGET
from .features import add_features


@dataclass
class DatasetVariant:
    name: str
    features: list                           # kolumny wejściowe modelu
    build: Callable = add_features           # df_clean -> df z cechami
    train_filter: Optional[Callable] = None  # train -> train (np. usunięcie outlierów)
    description: str = ''


DATASETS = {}


def register(name, features, build=add_features, train_filter=None, description=''):
    DATASETS[name] = DatasetVariant(name, list(features), build, train_filter, description)
    return DATASETS[name]


def without(*cols):
    drop = set(cols)
    return [c for c in FEATURE_COLS if c not in drop]


# --- Przykładowe cechy dodatkowe -------------------------------------------------
def add_extra_lags(df):
    """Dłuższa historia: lag 2, 4 i 52 tygodnie (sezonowość roczna) + odchylenie z 4 tyg."""
    df = add_features(df)
    grp = df.groupby(['Store', 'Dept'])[TARGET]
    for k in (2, 4, 52):
        df[f'lag_{k}'] = grp.shift(k)
    df['roll_std_4'] = grp.transform(lambda s: s.shift(1).rolling(4, min_periods=2).std())
    return df


EXTRA_LAGS = ['lag_2', 'lag_4', 'lag_52', 'roll_std_4']


def add_log_markdowns(df):
    """MarkDowny są silnie prawoskośne - log1p."""
    df = add_features(df)
    for c in MD_COLS + ['MarkDown_total']:
        df[c] = np.log1p(df[c].clip(lower=0))
    return df


# --- Warianty --------------------------------------------------------------------
register('bazowy', FEATURE_COLS, description='Wszystkie cechy z notebooka bazowego')
register('bez_historii', without(*HIST_COLS), description='Bez lagów i średnich historycznych')
register('bez_markdown', without(*MD_COLS, 'MarkDown_total', 'MarkDown_active_count', 'has_any_markdown'),
         description='Bez cech promocyjnych')
register('bez_makro', without('Temperature', 'Fuel_Price', 'CPI', 'Unemployment'),
         description='Bez zmiennych makro/pogodowych')
register('dodatkowe_lagi', FEATURE_COLS + EXTRA_LAGS, build=add_extra_lags,
         description='Bazowy + lag 2/4/52 i odchylenie kroczące')
register('log_markdown', FEATURE_COLS, build=add_log_markdowns,
         description='Bazowy z log1p na MarkDownach')
register('bez_ujemnej_sprzedazy', FEATURE_COLS, train_filter=lambda t: t[t[TARGET] >= 0],
         description='Trening bez tygodni ze zwrotami netto')
