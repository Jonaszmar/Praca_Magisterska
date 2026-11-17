"""Bezpieczny zapis plików wyników przy kilku procesach naraz (np. CPU i GPU równolegle).

- `locked(path)`: blokada na czas odczyt-modyfikacja-zapis (plik .lock tworzony atomowo).
- `atomic_to_csv(df, path)`: zapis do pliku tymczasowego i podmiana - przerwanie w trakcie
  zapisu (zanik prądu, Ctrl+C) nie zostawi uszkodzonego CSV.
"""
import os
import time
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def locked(path, timeout=600, stale=300):
    lock = Path(f'{path}.lock')
    lock.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    while True:
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            break
        except FileExistsError:
            try:   # blokada po procesie, który padł - usuwamy po `stale` sekundach
                if time.time() - lock.stat().st_mtime > stale:
                    lock.unlink(missing_ok=True)
                    continue
            except FileNotFoundError:
                continue
            if time.time() - t0 > timeout:
                raise TimeoutError(f'Nie udało się zablokować {path}')
            time.sleep(0.2)
    try:
        yield
    finally:
        lock.unlink(missing_ok=True)


def atomic_to_csv(df, path, retries=20, **kwargs):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f'{path.name}.{os.getpid()}.tmp')
    df.to_csv(tmp, **{'index': False, **kwargs})
    for i in range(retries):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:   # Windows: plik chwilowo otwarty (np. w Excelu/innym procesie)
            time.sleep(0.5 * (i + 1))
    raise PermissionError(f'Nie można zapisać {path} - czy plik jest otwarty w innym programie?')
