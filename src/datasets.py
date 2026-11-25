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
def by_date(df, col, weeks):
    """Wartość `col` z tego samego szeregu (Store, Dept) sprzed dokładnie `weeks` tygodni; brak -> NaN.

    605 z 3331 szeregów ma brakujące tygodnie; shift(52) po wierszach sięgałby wtedy dalej
    niż rok wstecz (13 tys. błędnych lag_52). Krótkie lagi zostają po wierszach („ostatnia
    znana sprzedaż”) - po dacie dawały puste wartości, uzupełniane średnią działu, i na
    walidacji wypadały gorzej (LightGBM +24 WMAE). Luki są w szeregach o znikomej sprzedaży.
    """
    s = df.set_index(['Store', 'Dept', 'Date'])[col]
    prev = pd.MultiIndex.from_arrays([df['Store'], df['Dept'], df['Date'] - pd.Timedelta(weeks=weeks)])
    return s.reindex(prev).to_numpy()


def add_extra_lags(df):
    """Dłuższa historia: lag 2, 4 i 52 tygodnie (sezonowość roczna) + odchylenie z 4 tyg."""
    df = add_features(df)
    grp = df.groupby(['Store', 'Dept'])[TARGET]
    for k in (2, 4):
        df[f'lag_{k}'] = grp.shift(k)
    df['lag_52'] = by_date(df, TARGET, 52)   # ten sam tydzień rok temu - po dacie, nie po wierszach
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
    """Dodatkowe lagi + po_swiecie + cechy wybrane na walidacji (foldy czasowe z treningu,
    test nietknięty; 18.11.2026): CV WMAE LightGBM -26, XGBoost -15.
    Sprawdzone i odrzucone (zyski się nie sumowały): roll_mean_8/13, lag_51, lag_53, zmiana r/r.
    Odrzucone 20.11.2026: dev_51, roll_mean_52, trend sprzedaży całego sklepu / działu
    (gorzej na oknie sezonowym), wagi świąt x5 w treningu (lepsze okno z Labor Day, gorsze reszta).
    """
    df = add_lags_after_holiday(df)   # posortowane po Store, Dept, Date (add_history_features)
    grp = df.groupby(['Store', 'Dept'])[TARGET]
    holidays = df.loc[df['IsHoliday'].astype(bool), 'Date'].unique()
    df['przed_swietem'] = (df['Date'] + pd.Timedelta(weeks=1)).isin(holidays).astype(int)
    df['ewm_4'] = grp.transform(lambda s: s.shift(1).ewm(span=4).mean())   # tylko przeszłość
    df['lag1_minus_lag2'] = df['lag_1'] - df['lag_2']                          # bieżący trend
    # Dodane 20.11.2026 (walidacja: 3 ostatnie okna + okno sezonowe 2011, test nietknięty):
    # dev_52 - jak bardzo ten tydzień rok temu odbiegał od ówczesnej średniej z 4 tygodni
    # (kształt sezonu bez poziomu sprzedaży); LightGBM-L1 -19, XGBoost -30 (razem z lag_52 po dacie).
    df['dev_52'] = df['lag_52'] - by_date(df, 'roll_mean_4', 52)
    # Wielkanoc to jedyne ważne święto, które przesuwa się o kilka tygodni między latami
    # (2011: 24.04, 2012: 8.04), więc lag_52/dev_52 trafiają wtedy w inny tydzień względem świąt.
    # Odległość od Wielkanocy teraz i rok temu; LightGBM-L1 -23, XGBoost -32.
    df['easter_rel'] = easter_rel(df['Date'])
    df['easter_rel_52'] = easter_rel(df['Date'] - pd.Timedelta(weeks=52))
    return df


EASTER = pd.to_datetime(['2010-04-04', '2011-04-24', '2012-04-08', '2013-03-31'])


def easter_rel(dates, max_weeks=3):
    """Odległość (w tygodniach) tygodnia sprzedaży od tygodnia z Niedzielą Wielkanocną.

    Data w danych to piątek KOŃCZĄCY tydzień (np. Thanksgiving, czwartek 25.11.2010 -> 26.11.2010),
    więc tydzień z Niedzielą Wielkanocną ma datę Wielkanoc + 5 dni. Poza oknem ±max_weeks: max_weeks + 1."""
    d = pd.to_datetime(pd.Series(dates)).to_numpy()
    rel = np.full(len(d), float(max_weeks + 1))
    for e in EASTER:
        r = (d - np.datetime64(e + pd.Timedelta(days=5))) / np.timedelta64(7, 'D')
        near = np.abs(r) <= max_weeks
        rel[near] = r[near]
    return rel


# --- Warianty --------------------------------------------------------------------
register('bazowy', FEATURE_COLS, description='Wszystkie cechy z notebooka bazowego')
register('bez_historii', without(*HIST_COLS), description='Bez lagów i średnich historycznych')
register('bez_markdown', without(*MD_COLS, 'MarkDown_total', 'MarkDown_active_count', 'has_any_markdown'),
         description='Bez cech promocyjnych')
register('bez_makro', without(*MACRO_COLS),
         description='Bez zmiennych makro/pogodowych')
register('dodatkowe_lagi', FEATURE_COLS + EXTRA_LAGS, build=add_extra_lags,
         description='Bazowy + lag 2/4/52 i odchylenie kroczące')
register('rozszerzone', without(*MACRO_COLS) + EXTRA_LAGS
         + ['po_swiecie', 'przed_swietem', 'ewm_4', 'lag1_minus_lag2', 'dev_52', 'easter_rel', 'easter_rel_52'],
         build=add_extended,
         description='Dodatkowe lagi, bez makro, + tydzień po / przed świętem, EWM(4), różnica lag1-lag2, '
                     'odchylenie rok temu (dev_52), tydzień względem Wielkanocy (teraz i rok temu)')
# Usunięty 20.11.2026: 'lagi_bez_makro_po_swiecie' - podzbiór 'rozszerzone' (ten sam zestaw bez
# przed_swietem, ewm_4, lag1-lag2, dev_52, Wielkanocy), wyniki praktycznie jak 'rozszerzone'.
# Usunięte po przebiegu z 18.11.2026 (wyniki w results/archiwum/): 'log_markdown' dawał wyniki
# identyczne z bazowym dla drzew (log1p nie zmienia kolejności), 'bez_ujemnej_sprzedazy'
# różnił się od bazowego o < 1% WMAE.
