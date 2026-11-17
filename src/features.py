"""Inżynieria cech."""
import pandas as pd

from .config import MD_COLS, TARGET

# Konkretne święta (inaczej wpływają na sprzedaż niż zwykłe IsHoliday=1)
HOLIDAYS = {
    'is_superbowl': ['2010-02-12', '2011-02-11', '2012-02-10', '2013-02-08'],
    'is_labor_day': ['2010-09-10', '2011-09-09', '2012-09-07', '2013-09-06'],
    'is_thanksgiving': ['2010-11-26', '2011-11-25', '2012-11-23', '2013-11-29'],
    'is_christmas': ['2010-12-31', '2011-12-30', '2012-12-28', '2013-12-27'],
}


def add_calendar_features(df):
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['WeekOfYear'] = df['Date'].dt.isocalendar().week.astype(int)
    for name, dates in HOLIDAYS.items():
        df[name] = df['Date'].isin(pd.to_datetime(dates)).astype(int)
    return df


def add_markdown_features(df):
    df['MarkDown_total'] = df[MD_COLS].sum(axis=1)
    df['MarkDown_active_count'] = (df[MD_COLS] > 0).sum(axis=1)
    df['has_any_markdown'] = (df['MarkDown_total'] > 0).astype(int)
    return df


def add_history_features(df):
    """Cechy historyczne WYŁĄCZNIE z przeszłości - .shift(1) przed każdym oknem,
    żeby wartość dla danego tygodnia nigdy nie zawierała informacji z tego
    samego lub przyszłego tygodnia."""
    df = df.sort_values(['Store', 'Dept', 'Date']).reset_index(drop=True)
    grp = df.groupby(['Store', 'Dept'])[TARGET]
    df['lag_1'] = grp.shift(1)
    df['roll_mean_4'] = grp.transform(lambda s: s.shift(1).rolling(4, min_periods=1).mean())
    df['expanding_mean_hist'] = grp.transform(lambda s: s.shift(1).expanding(min_periods=1).mean())
    return df


def add_features(df):
    df = add_calendar_features(df.copy())
    df = add_markdown_features(df)
    return add_history_features(df)
