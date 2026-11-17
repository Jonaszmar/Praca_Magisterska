"""Testy statystyczne na danych (normalność, różnice między grupami, korelacje, stacjonarność)."""
import numpy as np
import pandas as pd
from scipy import stats

from .config import RANDOM_STATE, TARGET

ALPHA = 0.05


def _sample(s, n, seed=RANDOM_STATE):
    s = pd.Series(s).dropna()
    return s.sample(n, random_state=seed) if len(s) > n else s


def normality_tests(df, cols, sample=5000, alpha=ALPHA):
    """Shapiro-Wilk + D'Agostino K² dla każdej kolumny.

    Shapiro-Wilk jest wiarygodny do ok. 5000 obserwacji, dlatego liczony jest na
    losowej próbie `sample` wierszy. H0: rozkład jest normalny.
    """
    rows = []
    for c in cols:
        x = _sample(df[c], sample)
        w, p_sw = stats.shapiro(x)
        k2, p_k2 = stats.normaltest(x)
        rows.append({'zmienna': c, 'n': len(x), 'skośność': stats.skew(x), 'kurtoza': stats.kurtosis(x),
                     'Shapiro W': w, 'Shapiro p': p_sw, "D'Agostino K2": k2, "D'Agostino p": p_k2,
                     'normalny?': bool(p_sw > alpha and p_k2 > alpha)})
    return pd.DataFrame(rows).set_index('zmienna')


def group_test(df, group_col, value_col=TARGET, alpha=ALPHA):
    """Test różnic między grupami (nieparametryczny, bo rozkłady nie są normalne).

    2 grupy -> Mann-Whitney U, >2 grupy -> Kruskal-Wallis. H0: grupy mają ten sam rozkład.
    """
    groups = [g[value_col].dropna().values for _, g in df.groupby(group_col)]
    if len(groups) == 2:
        name, (stat, p) = 'Mann-Whitney U', stats.mannwhitneyu(*groups, alternative='two-sided')
        # rank-biserial: wielkość efektu w [-1, 1]
        effect = 1 - 2 * stat / (len(groups[0]) * len(groups[1]))
    else:
        name, (stat, p) = 'Kruskal-Wallis', stats.kruskal(*groups)
        n = sum(len(g) for g in groups)
        effect = (stat - len(groups) + 1) / (n - len(groups))  # epsilon^2
    return pd.Series({'test': name, 'grupy': len(groups), 'statystyka': stat, 'p': p,
                      'wielkość efektu': effect, 'istotne?': bool(p < alpha)},
                     name=f'{value_col} ~ {group_col}')


def group_tests(df, group_cols, value_col=TARGET, alpha=ALPHA):
    return pd.DataFrame([group_test(df, g, value_col, alpha) for g in group_cols])


def correlation_tests(df, cols, target=TARGET, alpha=ALPHA):
    """Korelacja Spearmana każdej zmiennej z targetem. H0: brak zależności monotonicznej."""
    rows = []
    for c in cols:
        rho, p = stats.spearmanr(df[c], df[target], nan_policy='omit')
        rows.append({'zmienna': c, 'rho Spearmana': rho, 'p': p, 'istotne?': bool(p < alpha)})
    return (pd.DataFrame(rows).set_index('zmienna')
            .sort_values('rho Spearmana', key=np.abs, ascending=False))


def stationarity_test(series, alpha=ALPHA):
    """Test ADF (Augmented Dickey-Fuller). H0: szereg jest niestacjonarny. Wymaga statsmodels."""
    from statsmodels.tsa.stattools import adfuller
    stat, p, lags, n, crit, _ = adfuller(pd.Series(series).dropna())
    return pd.Series({'ADF': stat, 'p': p, 'opóźnienia': lags, 'n': n,
                      'kryt. 5%': crit['5%'], 'stacjonarny?': bool(p < alpha)})


def residual_tests(y_true, y_pred, sample=5000, alpha=ALPHA):
    """Normalność i nieobciążoność reszt modelu (Shapiro-Wilk + test t średniej = 0)."""
    res = np.asarray(y_true) - np.asarray(y_pred)
    w, p_sw = stats.shapiro(_sample(res, sample))
    t, p_t = stats.ttest_1samp(res, 0)
    return pd.Series({'średnia reszt': res.mean(), 'Shapiro W': w, 'Shapiro p': p_sw,
                      'reszty normalne?': bool(p_sw > alpha), 't (średnia=0)': t, 'p (średnia=0)': p_t,
                      'nieobciążony?': bool(p_t > alpha)})
