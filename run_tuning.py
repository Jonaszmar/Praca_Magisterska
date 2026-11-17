"""Strojenie modeli i porównanie wersji domyślnej ze strojoną na zbiorze testowym.

    python run_tuning.py "Sieć z embeddingami"
    python run_tuning.py LightGBM XGBoost --n-iter 25

Przerwane strojenie można wznowić tym samym poleceniem (próby są zapisywane na bieżąco).
"""
import argparse
import warnings

from src.data import clean, load_raw
from src.experiment import run_experiments
from src.tuning import register_tuned, tune

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument('models', nargs='+')
parser.add_argument('--dataset', default='bazowy')
parser.add_argument('--n-iter', type=int, default=25)
parser.add_argument('--n-folds', type=int, default=3)
args = parser.parse_args()

df_clean = clean(load_raw(), verbose=False)
for model in args.models:
    tune(model, dataset=args.dataset, n_iter=args.n_iter, n_folds=args.n_folds, df_clean=df_clean)
    tuned = register_tuned(model, args.dataset)
    results, _ = run_experiments(datasets=[args.dataset], models=[model, tuned], df_clean=df_clean)
    sel = (results['dane'] == args.dataset) & results['model'].isin([model, tuned])
    print(results[sel][['model', 'WMAE', 'MAE', 'R2', 'czas_s']]
          .to_string(index=False), flush=True)
