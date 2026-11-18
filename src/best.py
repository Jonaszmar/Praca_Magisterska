"""Zapis najlepszych modeli (wg WMAE na teście) do models/najlepsze/ i commit do gita.

Modele z siatki nie są zapisywane w trakcie przebiegu (Random Forest to ~640 MB na wariant),
więc najlepsze pary (dane, model) trenujemy tu ponownie - z tym samym ziarnem, na tych samych
danych - i sprawdzamy, że WMAE zgadza się z tabelą wyników.

    from src.best import save_best_models
    save_best_models(top=5)

Wczytanie: joblib.load('models/najlepsze/01__bazowy__LightGBM_(strojony).joblib')
"""
import subprocess

import joblib
import pandas as pd

from .config import MODELS_DIR, ROOT, TARGET
from .data import clean, load_raw
from .datasets import DATASETS
from .evaluation import metrics
from .experiment import RESULTS_CSV, load_results, prepare
from .models import MODELS, NEEDS_IMPUTE

BEST_DIR = MODELS_DIR / 'najlepsze'
GIT_LIMIT_MB = 95   # GitHub odrzuca pliki > 100 MB


def _resolve(model):
    """'LightGBM (strojony)' nie jest w MODELS, dopóki nie wczytamy najlepszych parametrów."""
    if model not in MODELS and model.endswith(' (strojony)'):
        from .tuning import register_tuned
        base = model[:-len(' (strojony)')]
        return base, register_tuned
    return None, None


def save_best_models(top=5, df_clean=None, results_path=RESULTS_CSV, commit=True):
    res = load_results(results_path)
    if not len(res):
        print('Brak wyników - nie ma czego zapisać.')
        return pd.DataFrame()
    best = res.sort_values('WMAE').head(top).reset_index(drop=True)
    if df_clean is None:
        df_clean = clean(load_raw(), verbose=False)

    BEST_DIR.mkdir(parents=True, exist_ok=True)
    for old in BEST_DIR.glob('*.joblib'):   # poprzedni zestaw jest w historii gita
        old.unlink()

    rows, too_big = [], []
    for i, r in best.iterrows():
        d, m = r['dane'], r['model']
        base, register = _resolve(m)
        if register:
            register(base, d)   # parametry ze strojenia na tym samym wariancie danych
        X_train, y_train, X_test, test = prepare(DATASETS[d], df_clean)
        if m in NEEDS_IMPUTE:
            med = X_train.median()
            X_train, X_test = X_train.fillna(med), X_test.fillna(med)
        model = MODELS[m]()
        model.fit(X_train, y_train)
        wmae = metrics(test[TARGET], model.predict(X_test), test['IsHoliday'])['WMAE']

        path = BEST_DIR / f'{i + 1:02d}__{d}__{m}.joblib'.replace(' ', '_')
        joblib.dump(model, path, compress=3)
        mb = path.stat().st_size / 2**20
        if mb > GIT_LIMIT_MB:
            too_big.append(path)
        print(f'#{i + 1} [{d}] {m}: WMAE w tabeli {r["WMAE"]:.1f}, po ponownym treningu {wmae:.1f}  '
              f'-> {path.name} ({mb:.1f} MB)', flush=True)
        rows.append({**r.to_dict(), 'WMAE_ponowny_trening': round(wmae, 2),
                     'plik': path.name, 'rozmiar_MB': round(mb, 1),
                     'w_gicie': mb <= GIT_LIMIT_MB, 'cechy': ','.join(DATASETS[d].features)})

    ranking = pd.DataFrame(rows)
    ranking.to_csv(BEST_DIR / 'ranking.csv', index=False)
    # zbyt duże pliki zostają lokalnie, ale poza gitem
    (BEST_DIR / '.gitignore').write_text(''.join(f'{p.name}\n' for p in too_big), encoding='utf-8')
    if too_big:
        print(f'UWAGA: > {GIT_LIMIT_MB} MB, tylko lokalnie (nie w gicie): {[p.name for p in too_big]}')
    if commit:
        git_commit_best(top)
    return ranking


def git_commit_best(top):
    """Commituje wyłącznie models/najlepsze/ - inne zmiany w repo zostają nietknięte."""
    rel = BEST_DIR.relative_to(ROOT).as_posix()
    git = lambda *a: subprocess.run(['git', *a], cwd=ROOT, capture_output=True, text=True)
    try:
        git('add', '-A', '--', rel)
        if not git('diff', '--cached', '--quiet', '--', rel).returncode:
            print('Git: najlepsze modele bez zmian - nic do commitowania.')
            return
        out = git('commit', '-m', f'Najlepsze modele (top {top} wg WMAE)', '--', rel)
        print('Git: ' + (out.stdout.strip().splitlines() or [''])[0] if out.returncode == 0
              else f'Git: commit nieudany - {out.stderr.strip()}')
    except FileNotFoundError:
        print('Git: nie znaleziono polecenia git - pliki zapisane, commit pominięty.')
