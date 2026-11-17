"""Stałe projektu: ścieżki, paleta kolorów, lista cech."""
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / 'data'
MODELS_DIR = ROOT / 'models'
RESULTS_DIR = ROOT / 'results'
RAW_CSV = DATA_DIR / 'walmart_cleaned.csv'

RANDOM_STATE = 42
TEST_WEEKS = 8
TARGET = 'Weekly_Sales'

MD_COLS = ['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5']
HIST_COLS = ['lag_1', 'roll_mean_4', 'expanding_mean_hist']
FEATURE_COLS = [
    'Store', 'Dept', 'Type', 'Size', 'IsHoliday',
    'Temperature', 'Fuel_Price', 'CPI', 'Unemployment',
    *MD_COLS,
    'MarkDown_total', 'MarkDown_active_count', 'has_any_markdown',
    'Year', 'Month', 'WeekOfYear',
    'is_superbowl', 'is_labor_day', 'is_thanksgiving', 'is_christmas',
    *HIST_COLS,
]

# --- Spójna paleta kolorów dla wszystkich wykresów ---
CAT = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7', '#e34948']
INK = '#0b0b0b'
INK2 = '#52514e'
MUTED = '#898781'
GRID = '#e1e0d9'
SURFACE = '#fcfcfb'
DIVERGING = LinearSegmentedColormap.from_list('diverging', ['#e34948', '#f0efec', '#2a78d6'])


def setup_style():
    """Ustawia styl matplotlib i opcje wyświetlania pandas."""
    plt.rcParams.update({
        'figure.facecolor': SURFACE,
        'axes.facecolor': SURFACE,
        'axes.edgecolor': GRID,
        'axes.labelcolor': INK2,
        'text.color': INK,
        'xtick.color': MUTED,
        'ytick.color': MUTED,
        'grid.color': GRID,
        'axes.grid': True,
        'grid.linewidth': 0.8,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'font.size': 11,
        'figure.dpi': 100,
    })
    pd.set_option('display.max_columns', 50)
    pd.set_option('display.width', 160)
