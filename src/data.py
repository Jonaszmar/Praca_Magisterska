"""Wczytanie, czyszczenie i podział danych."""
import pandas as pd

from .config import RAW_CSV, TEST_WEEKS, TARGET, HIST_COLS


def load_raw(path=RAW_CSV):
    return pd.read_csv(path, index_col=0)


def clean(df_raw, verbose=True):
    df = df_raw.copy()
    df['Date'] = pd.to_datetime(df['Date'])

    # Ujemne MarkDown2/MarkDown3 - drobne korekty/błędy danych źródłowych - przycinamy do 0
    for col in ['MarkDown2', 'MarkDown3']:
        n_neg = (df[col] < 0).sum()
        df[col] = df[col].clip(lower=0)
        if verbose:
            print(f"{col}: przycięto {n_neg} ujemnych wartości do 0")

    # Ujemna Weekly_Sales to realny fakt biznesowy (zwroty netto przewyższające sprzedaż
    # w danym tygodniu), a nie błąd - zostawiamy wartości, ale oznaczamy flagą.
    df['is_negative_sales'] = (df[TARGET] < 0).astype(int)
    if verbose:
        print(f"\nOznaczono {df['is_negative_sales'].sum()} wierszy "
              f"({df['is_negative_sales'].mean()*100:.2f}%) z ujemną sprzedażą jako zwroty netto.")

    for col in ['Store', 'Dept', 'Type', 'IsHoliday']:
        df[col] = df[col].astype(int)

    if verbose:
        print("\nDuplikaty po czyszczeniu:", df.duplicated().sum())
    return df.sort_values(['Store', 'Dept', 'Date']).reset_index(drop=True)


def time_split(df, test_weeks=TEST_WEEKS, verbose=True):
    """Podział czasowy: ostatnie `test_weeks` tygodni to test.

    Braki w cechach historycznych uzupełniane średnią działu z TRENINGU (bez wycieku).
    """
    cutoff_date = sorted(df['Date'].unique())[-test_weeks]
    train = df[df['Date'] < cutoff_date].copy()
    test = df[df['Date'] >= cutoff_date].copy()

    dept_mean_train = train.groupby('Dept')[TARGET].mean()
    global_mean_train = train[TARGET].mean()
    for part in (train, test):
        for c in HIST_COLS:
            part[c] = part[c].fillna(part['Dept'].map(dept_mean_train)).fillna(global_mean_train)

    if verbose:
        print("Data odcięcia (cutoff):", pd.Timestamp(cutoff_date).date())
        print(f"Train: {len(train):,} wierszy, {train['Date'].min().date()} - {train['Date'].max().date()}")
        print(f"Test:  {len(test):,} wierszy, {test['Date'].min().date()} - {test['Date'].max().date()}")
        print("Braki po uzupełnieniu (train):", train[HIST_COLS].isna().sum().sum())
        print("Braki po uzupełnieniu (test):", test[HIST_COLS].isna().sum().sum())
    return train, test
