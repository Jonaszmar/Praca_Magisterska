"""Rejestr wariantów danych do eksperymentów.

Każdy wariant to przepis: jak z oczyszczonej ramki zbudować cechy i których kolumn
użyć w modelu. Nowy wariant dodaje się jednym wywołaniem `register(...)` - tutaj
albo bezpośrednio w notebooku.

Zbiór testowy jest zawsze ten sam (ostatnie tygodnie), więc wyniki wariantów są
porównywalne. Filtrowanie wierszy (`train_filter`) działa tylko na treningu.
"""
from dataclasses import dataclass
from typing import Callable, Optional

import pandas as pd

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


MACRO_COLS = ['Temperature', 'Fuel_Price', 'CPI', 'Unemployment']


def add_lags_after_holiday(df):
    """Dodatkowe lagi + flaga `po_swiecie` (poprzedni tydzień był świętem).

    lag_1 w tygodniu po święcie niesie świąteczny skok sprzedaży - bez tej flagi model
    przenosi go na zwykły tydzień (po Labor Day 2012 przeszacowanie średnio o ~440).
    Flaga zależy tylko od kalendarza, więc nie ma przecieku z przyszłości.
    """
    df = add_extra_lags(df)
    holidays = df.loc[df['IsHoliday'].astype(bool), 'Date'].unique()
    df['po_swiecie'] = (df['Date'] - pd.Timedelta(weeks=1)).isin(holidays).astype(int)
    return df


def add_extended(df):
    """lagi_bez_makro_po_swiecie + trzy cechy wybrane na walidacji (foldy czasowe z treningu,
    test nietknięty; 18.11.2026): CV WMAE LightGBM -26, XGBoost -15.
    Sprawdzone i odrzucone (zyski się nie sumowały): roll_mean_8/13, lag_51, lag_53, zmiana r/r.
    """
    df = add_lags_after_holiday(df)   # posortowane po Store, Dept, Date (add_history_features)
    grp = df.groupby(['Store', 'Dept'])[TARGET]
    holidays = df.loc[df['IsHoliday'].astype(bool), 'Date'].unique()
    df['przed_swietem'] = (df['Date'] + pd.Timedelta(weeks=1)).isin(holidays).astype(int)
    df['ewm_4'] = grp.transform(lambda s: s.shift(1).ewm(span=4).mean())   # tylko przeszłość
    df['lag1_minus_lag2'] = df['lag_1'] - df['lag_2']                          # bieżący trend
    return df


# --- Warianty --------------------------------------------------------------------
register('bazowy', FEATURE_COLS, description='Wszystkie cechy z notebooka bazowego')
register('bez_historii', without(*HIST_COLS), description='Bez lagów i średnich historycznych')
register('bez_markdown', without(*MD_COLS, 'MarkDown_total', 'MarkDown_active_count', 'has_any_markdown'),
         description='Bez cech promocyjnych')
register('bez_makro', without(*MACRO_COLS),
         description='Bez zmiennych makro/pogodowych')
register('dodatkowe_lagi', FEATURE_COLS + EXTRA_LAGS, build=add_extra_lags,
         description='Bazowy + lag 2/4/52 i odchylenie kroczące')
register('lagi_bez_makro_po_swiecie', without(*MACRO_COLS) + EXTRA_LAGS + ['po_swiecie'],
         build=add_lags_after_holiday,
         description='Dodatkowe lagi, bez makro, + flaga tygodnia po święcie')
register('rozszerzone', DATASETS['lagi_bez_makro_po_swiecie'].features
         + ['przed_swietem', 'ewm_4', 'lag1_minus_lag2'], build=add_extended,
         description='lagi_bez_makro_po_swiecie + tydzień przed świętem, EWM(4), różnica lag1-lag2')
# Usunięte po przebiegu z 18.11.2026 (wyniki w results/archiwum/): 'log_markdown' dawał wyniki
# identyczne z bazowym dla drzew (log1p nie zmienia kolejności), 'bez_ujemnej_sprzedazy'
# różnił się od bazowego o < 1% WMAE.
