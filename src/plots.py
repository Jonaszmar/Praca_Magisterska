"""Wykresy oceny predykcji na zbiorze testowym.

Wejście: tabele z `load_preds(dane, model)` - kolumny Store, Dept, Date, IsHoliday, y_true, y_pred.
Większość funkcji przyjmuje słownik {nazwa_modelu: tabela}, żeby porównać kilka modeli naraz.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator

from .config import CAT, INK2, MUTED, TARGET
from .evaluation import metrics

_mln = FuncFormatter(lambda v, _: f'{v / 1e6:.0f} mln')
_tys = FuncFormatter(lambda v, _: f'{v / 1e3:.0f} tys.')


def summary(preds):
    """Tabela metryk dla każdego modelu (WMAE, MAE, RMSE, R2) + błąd sumy sprzedaży sieci."""
    rows = []
    for name, p in preds.items():
        m = metrics(p['y_true'], p['y_pred'], p['IsHoliday'])
        tot = p.groupby('Date')[['y_true', 'y_pred']].sum()
        m['błąd sumy tyg. %'] = 100 * (tot['y_pred'] - tot['y_true']).abs().mean() / tot['y_true'].mean()
        rows.append({'model': name, **m})
    return pd.DataFrame(rows).set_index('model').sort_values('WMAE')


def plot_total(preds, history=None, history_weeks=30, ax=None):
    """Łączna tygodniowa sprzedaż wszystkich sklepów: rzeczywista vs przewidziana.

    history - pełne dane (np. df_clean), żeby pokazać `history_weeks` tygodni treningu przed testem.
    """
    ax = ax or plt.subplots(figsize=(11, 4.5))[1]
    first = next(iter(preds.values()))
    start = first['Date'].min()
    if history is not None:
        h = history[history['Date'] < start].groupby('Date')[TARGET].sum().iloc[-history_weeks:]
        ax.plot(h.index, h.values, color=MUTED, lw=1.5, label='historia (trening)')
    true = first.groupby('Date')['y_true'].sum()
    ax.plot(true.index, true.values, color='black', lw=2.2, marker='o', ms=4, label='rzeczywista')
    for (name, p), c in zip(preds.items(), CAT):
        s = p.groupby('Date')['y_pred'].sum()
        ax.plot(s.index, s.values, color=c, lw=1.8, ls='--', marker='o', ms=3, label=name)
    ax.axvline(start, color=INK2, ls=':', lw=1)
    ax.text(start, ax.get_ylim()[1], ' początek testu', va='top', fontsize=9, color=INK2)
    for d in first.loc[first['IsHoliday'] == 1, 'Date'].unique():
        ax.axvspan(d - pd.Timedelta(days=3), d + pd.Timedelta(days=3), color=CAT[3], alpha=0.15)
    ax.yaxis.set_major_formatter(_mln)
    ax.set_title('Łączna tygodniowa sprzedaż sieci - zbiór testowy (żółte = tydzień świąteczny)')
    ax.set_ylabel('sprzedaż')
    ax.legend(fontsize=9)
    return ax


def plot_series(preds, store, dept, history=None, history_weeks=52, ax=None):
    """Sprzedaż jednego działu w jednym sklepie: rzeczywista vs przewidziana."""
    ax = ax or plt.subplots(figsize=(11, 4))[1]
    first = next(iter(preds.values()))
    sel = lambda p: p[(p['Store'] == store) & (p['Dept'] == dept)].sort_values('Date')
    if history is not None:
        h = history[(history['Store'] == store) & (history['Dept'] == dept)
                    & (history['Date'] < first['Date'].min())].sort_values('Date').iloc[-history_weeks:]
        ax.plot(h['Date'], h[TARGET], color=MUTED, lw=1.5, label='historia (trening)')
    t = sel(first)
    ax.plot(t['Date'], t['y_true'], color='black', lw=2.2, marker='o', ms=4, label='rzeczywista')
    for (name, p), c in zip(preds.items(), CAT):
        s = sel(p)
        ax.plot(s['Date'], s['y_pred'], color=c, lw=1.8, ls='--', marker='o', ms=3, label=name)
    ax.axvline(t['Date'].min(), color=INK2, ls=':', lw=1)
    ax.yaxis.set_major_formatter(_tys)
    ax.set_title(f'Sklep {store}, dział {dept}')
    ax.legend(fontsize=9)
    return ax


def plot_examples(preds, history=None, pairs=None, n=6, seed=0):
    """Siatka kilku par (sklep, dział). Domyślnie losowe spośród 30% największych działów."""
    first = next(iter(preds.values()))
    if pairs is None:
        size = first.groupby(['Store', 'Dept'])['y_true'].mean()
        big = list(size[size >= size.quantile(0.7)].index)
        idx = np.random.default_rng(seed).choice(len(big), min(n, len(big)), replace=False)
        pairs = [big[i] for i in idx]
    rows = int(np.ceil(len(pairs) / 2))
    fig, axes = plt.subplots(rows, 2, figsize=(14, 3.4 * rows), squeeze=False)
    for ax, (s, d) in zip(axes.flat, pairs):
        plot_series(preds, s, d, history=history, history_weeks=20, ax=ax)
        ax.legend().set_visible(ax is axes.flat[0])
        ax.tick_params(axis='x', labelrotation=30, labelsize=8)
    for ax in list(axes.flat)[len(pairs):]:
        ax.set_visible(False)
    fig.tight_layout()
    return fig


def plot_scatter(p, name='', ax=None):
    """Przewidziana vs rzeczywista dla każdej obserwacji; idealnie punkty leżą na przekątnej."""
    ax = ax or plt.subplots(figsize=(6, 6))[1]
    hol = p['IsHoliday'] == 1
    ax.scatter(p.loc[~hol, 'y_true'], p.loc[~hol, 'y_pred'], s=4, alpha=0.15, color=CAT[0], label='zwykły tydzień')
    ax.scatter(p.loc[hol, 'y_true'], p.loc[hol, 'y_pred'], s=6, alpha=0.4, color=CAT[1], label='tydzień świąteczny')
    lim = [min(p['y_true'].min(), p['y_pred'].min()), max(p['y_true'].max(), p['y_pred'].max())]
    ax.plot(lim, lim, color='black', lw=1, ls='--', label='idealna predykcja')
    ax.xaxis.set_major_formatter(_tys); ax.yaxis.set_major_formatter(_tys)
    ax.xaxis.set_major_locator(MaxNLocator(5)); ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.set_xlabel('rzeczywista sprzedaż'); ax.set_ylabel('przewidziana sprzedaż')
    ax.set_title(f'Przewidziana vs rzeczywista {name}'.strip())
    ax.legend(fontsize=9, markerscale=3)
    return ax


def plot_errors(p, name='', bins=100):
    """Rozkład błędów (rzeczywista - przewidziana) i błąd procentowy w zależności od wielkości działu."""
    err = p['y_true'] - p['y_pred']
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    lo, hi = err.quantile([0.005, 0.995])
    axes[0].hist(err.clip(lo, hi), bins=bins, color=CAT[0], alpha=0.8)
    axes[0].axvline(0, color='black', lw=1)
    axes[0].axvline(err.mean(), color=CAT[1], lw=1.5, ls='--', label=f'średni błąd = {err.mean():,.0f}'.replace(',', ' '))
    axes[0].set_title(f'Rozkład błędów {name}'.strip()); axes[0].set_xlabel('rzeczywista − przewidziana')
    axes[0].xaxis.set_major_formatter(_tys); axes[0].legend()

    g = p.assign(ae=err.abs()).groupby(['Store', 'Dept']).agg(sprzedaz=('y_true', 'mean'), mae=('ae', 'mean'))
    g = g[g['sprzedaz'] > 100]
    axes[1].scatter(g['sprzedaz'], 100 * g['mae'] / g['sprzedaz'], s=6, alpha=0.4, color=CAT[0])
    axes[1].set_xscale('log'); axes[1].set_yscale('log')
    axes[1].set_xlabel('średnia tygodniowa sprzedaż działu (log)')
    axes[1].set_ylabel('MAE jako % sprzedaży (log)')
    axes[1].set_title('Błąd względny vs wielkość działu')
    fig.tight_layout()
    return fig


def plot_error_by(preds, by='Dept', top=20, ax=None):
    """Działy/sklepy, na których modele mylą się najbardziej (MAE w USD), porównanie modeli."""
    tab = pd.DataFrame({name: (p['y_true'] - p['y_pred']).abs().groupby(p[by]).mean()
                        for name, p in preds.items()})
    tab = tab.loc[tab.mean(axis=1).sort_values(ascending=False).index[:top]]
    ax = ax or plt.subplots(figsize=(12, 4.5))[1]
    tab.plot.bar(ax=ax, color=CAT[:len(tab.columns)], width=0.8)
    ax.yaxis.set_major_formatter(_tys)
    ax.set_title(f'{top} pozycji ({by}) z największym błędem MAE')
    ax.set_ylabel('MAE'); ax.set_xlabel(by)
    return ax


def plot_error_by_week(preds, ax=None):
    """MAE w każdym tygodniu testu - czy model psuje się z czasem albo w święta?"""
    ax = ax or plt.subplots(figsize=(11, 3.8))[1]
    for (name, p), c in zip(preds.items(), CAT):
        w = (p['y_true'] - p['y_pred']).abs().groupby(p['Date']).mean()
        ax.plot(w.index, w.values, marker='o', color=c, label=name)
    first = next(iter(preds.values()))
    for d in first.loc[first['IsHoliday'] == 1, 'Date'].unique():
        ax.axvspan(d - pd.Timedelta(days=3), d + pd.Timedelta(days=3), color=CAT[3], alpha=0.15)
    ax.set_title('MAE w kolejnych tygodniach testu (żółte = tydzień świąteczny)')
    ax.set_ylabel('MAE'); ax.legend(fontsize=9)
    return ax
