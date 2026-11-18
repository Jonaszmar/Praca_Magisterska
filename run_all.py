"""Cały pipeline jednym poleceniem - do puszczenia na noc / bez nadzoru.

    python run_all.py --fresh          # od zera (stare wyniki przenoszone do results/archiwum/)
    python run_all.py                  # wznowienie: policzone rzeczy są pomijane
    python run_all.py --only-best      # tylko krok końcowy (średnie + najlepsze) na istniejących wynikach

Uruchamia równolegle dwa procesy:
  CPU: siatka (wszystkie warianty danych x modele klasyczne) -> strojenie LightGBM i XGBoost
  GPU: siatka dla sieci z embeddingami -> strojenie sieci
Po każdym strojeniu: trening wersji strojonej, ocena na teście, zapis predykcji i modelu.

Na koniec: średnia XGB+LGBM+Sieć dla każdego wariantu (src/ensemble.py), potem top-N modeli (wg WMAE na teście) -> models/najlepsze/ + commit do gita (src/best.py).

Wyniki:  results/eksperymenty.csv, results/predykcje/, results/strojenie/, models/
Logi:    results/logi/  (run_all_cpu_*.log, run_all_gpu_*.log, status_*.txt)
Oglądanie wyników: 03_eksperymenty.ipynb (sekcje 4-6) i 04_wyniki.ipynb.
"""
import argparse
import os
import shutil
import subprocess
import sys
import time
import traceback
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

from src.config import RESULTS_DIR
from src.models import GRID_CPU as CPU_GRID, GRID_GPU as GPU_GRID

LOG_DIR = RESULTS_DIR / 'logi'
CPU_TUNE = ['LightGBM', 'XGBoost']
GPU_TUNE = ['Sieć z embeddingami']


class Tee:
    """Wypisuje jednocześnie na konsolę i do pliku, z godziną na początku każdej linii."""

    def __init__(self, path, console):
        self.file = open(path, 'a', encoding='utf-8', buffering=1)
        self.console = console
        self.new_line = True

    def write(self, s):
        for part in s.splitlines(keepends=True):
            if self.new_line and part.strip():
                part = datetime.now().strftime('%H:%M:%S ') + part
            self.new_line = part.endswith('\n')
            self.file.write(part)
            if self.console is not None:
                try:
                    self.console.write(part)
                except Exception:
                    pass   # konsola zamknięta - log w pliku i tak powstaje
        return len(s)

    def flush(self):
        self.file.flush()
        if self.console is not None:
            try:
                self.console.flush()
            except Exception:
                pass


def keep_awake():
    """Windows: blokuje uśpienie komputera, dopóki działa proces (ekran może zgasnąć)."""
    if sys.platform == 'win32':
        import ctypes
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)


def archive_results():
    """Przenosi (nie kasuje!) poprzednie wyniki do results/archiwum/<data>/."""
    dest = RESULTS_DIR / 'archiwum' / datetime.now().strftime('%Y%m%d_%H%M%S')
    moved = []
    for name in ['eksperymenty.csv', 'strojenie', 'predykcje', 'logi']:
        src = RESULTS_DIR / name
        if src.exists():
            dest.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dest / name))
            moved.append(name)
    print(f'Poprzednie wyniki przeniesione do {dest}: {moved}' if moved else 'Brak poprzednich wyników.')


def run_part(part, args):
    from src.data import clean, load_raw
    from src.datasets import DATASETS
    from src.experiment import run_experiments
    from src.tuning import register_tuned, tune

    if part == 'gpu':
        import torch
        torch.set_num_threads(1)   # CPU zostaje dla boostingów z drugiego procesu
        print(f"PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}"
              + (f" ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else ' - UWAGA: liczę na CPU'))
    grid, to_tune = (GPU_GRID, GPU_TUNE) if part == 'gpu' else (CPU_GRID, CPU_TUNE)
    datasets = list(DATASETS)
    status = []

    def phase(name, fn):
        t0 = time.time()
        print(f'\n===== {name} =====', flush=True)
        try:
            fn()
            status.append(f'OK    {name}  ({(time.time() - t0) / 60:.0f} min)')
        except Exception:
            traceback.print_exc(file=sys.stdout)
            status.append(f'BŁĄD  {name}  ({(time.time() - t0) / 60:.0f} min) - szczegóły w logu')
        print(f'----- koniec: {status[-1]}', flush=True)
        (LOG_DIR / f'status_{part}.txt').write_text('\n'.join(status) + '\n', encoding='utf-8')

    df_clean = clean(load_raw(), verbose=False)
    if not args.skip_grid:
        phase(f'Siatka: {len(datasets)} wariantów x {grid}',
              lambda: run_experiments(datasets=datasets, models=grid, df_clean=df_clean))
    if not args.skip_tuning:
        for ds in args.tune_datasets:
            for m in to_tune:
                def tune_and_test(m=m, ds=ds):
                    tune(m, dataset=ds, n_iter=args.n_iter, n_folds=args.n_folds, df_clean=df_clean)
                    tuned = register_tuned(m, ds)
                    run_experiments(datasets=[ds], models=[tuned], df_clean=df_clean, force=True,
                                    save_models=True)
                phase(f'Strojenie {m} na "{ds}" ({args.n_iter} prób x {args.n_folds} foldy)', tune_and_test)
    return status


def finish(args):
    """Krok końcowy po siatce i strojeniu: średnie modeli, potem zapis najlepszych."""
    from src.best import save_best_models
    from src.ensemble import ENSEMBLE_NAME, add_ensembles
    from src.experiment import load_results
    steps = [(ENSEMBLE_NAME, add_ensembles),
             (f'Zapis {args.top} najlepszych modeli do models/najlepsze/',
              lambda: save_best_models(top=args.top, commit=not args.no_commit))]
    for name, fn in steps:
        print(f'\n===== {name} =====', flush=True)
        try:
            fn()
        except Exception:
            traceback.print_exc(file=sys.stdout)
            print('Krok nieudany - powtórz: python run_all.py --only-best', flush=True)
    res = load_results()
    if len(res):
        print('\nTOP 15 (WMAE na teście):')
        print(res.sort_values('WMAE')[['dane', 'model', 'WMAE', 'R2']].head(15).to_string(index=False))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--part', choices=['all', 'cpu', 'gpu'], default='all')
    p.add_argument('--fresh', action='store_true', help='zacznij od zera (archiwizuje stare wyniki)')
    p.add_argument('--n-iter', type=int, default=40, help='liczba prób strojenia na model (domyślnie 40)')
    p.add_argument('--n-folds', type=int, default=3)
    p.add_argument('--tune-datasets', nargs='+',
                   default=['bazowy', 'dodatkowe_lagi', 'lagi_bez_makro_po_swiecie'])
    p.add_argument('--skip-grid', action='store_true')
    p.add_argument('--skip-tuning', action='store_true')
    p.add_argument('--top', type=int, default=5, help='ile najlepszych modeli zapisać (domyślnie 5)')
    p.add_argument('--only-best', action='store_true',
                   help='tylko krok końcowy: średnie modeli + zapis najlepszych, i zakończ')
    p.add_argument('--no-commit', action='store_true', help='nie commituj najlepszych modeli do gita')
    args = p.parse_args()

    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
    from src.datasets import DATASETS
    unknown = [d for d in args.tune_datasets if d not in DATASETS]
    if unknown:
        p.error(f'nieznane warianty danych {unknown}; dostępne: {list(DATASETS)}')

    keep_awake()
    if args.only_best:
        finish(args)
        return
    if args.fresh and args.part == 'all':
        archive_results()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    if args.part != 'all':
        console = sys.stdout if args.part == 'cpu' else None   # proces GPU pisze tylko do pliku
        sys.stdout = sys.stderr = Tee(LOG_DIR / f'run_all_{args.part}_{stamp}.log', console)
        status = run_part(args.part, args)
        print('\n'.join(['', f'PODSUMOWANIE ({args.part}):'] + status), flush=True)
        return

    # --part all: dwa procesy potomne - GPU w tle (tylko log), CPU na konsoli
    sys.stdout = Tee(LOG_DIR / f'run_all_main_{stamp}.log', sys.stdout)
    passthrough = [f'--n-iter={args.n_iter}', f'--n-folds={args.n_folds}', '--tune-datasets', *args.tune_datasets]
    passthrough += ['--skip-grid'] * args.skip_grid + ['--skip-tuning'] * args.skip_tuning
    gpu_err = open(LOG_DIR / f'gpu_stderr_{stamp}.txt', 'w', encoding='utf-8')   # błędy przed startem logu
    gpu = subprocess.Popen([sys.executable, '-u', __file__, '--part', 'gpu', *passthrough],
                           env={**os.environ, 'PYTHONIOENCODING': 'utf-8'},
                           stdout=gpu_err, stderr=gpu_err)
    print(f'Start: {datetime.now():%Y-%m-%d %H:%M}. Proces GPU (PID {gpu.pid}) loguje do {LOG_DIR}.')
    t0 = time.time()
    try:
        cpu_rc = subprocess.call([sys.executable, '-u', __file__, '--part', 'cpu', *passthrough],
                                 env={**os.environ, 'PYTHONIOENCODING': 'utf-8'})
        print(f'\nCPU skończone ({(time.time() - t0) / 3600:.1f} h), czekam na GPU...', flush=True)
        gpu_rc = gpu.wait()
    except KeyboardInterrupt:
        gpu.terminate()
        raise

    print(f'\n===== KONIEC: {datetime.now():%Y-%m-%d %H:%M}, łącznie {(time.time() - t0) / 3600:.1f} h =====')
    for part, rc in [('cpu', cpu_rc), ('gpu', gpu_rc)]:
        f = LOG_DIR / f'status_{part}.txt'
        print(f'\n[{part}] kod wyjścia {rc}')
        print(f.read_text(encoding='utf-8') if f.exists() else '  (brak pliku statusu - proces padł na starcie, patrz log)')

    finish(args)
    (LOG_DIR / 'GOTOWE.txt').write_text(f'Zakończono {datetime.now():%Y-%m-%d %H:%M}\n', encoding='utf-8')


if __name__ == '__main__':
    main()
