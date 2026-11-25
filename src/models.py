"""Rejestr modeli. Nowy model: `MODELS['nazwa'] = funkcja_zwracająca_estymator`."""
import time

import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import MODELS_DIR, RANDOM_STATE


def make_emb_nn(**params):
    """Sieć PyTorch z embeddingami Store/Dept/Type - szczegóły w src/nn.py."""
    from .nn import EmbeddingMLPRegressor   # wymaga `pip install torch`
    return EmbeddingMLPRegressor(**params)


def make_rf(**params):
    """Random Forest - tylko do notatnika 02_modelowanie (poza siatką: zdominowany przez
    boosting na każdym wariancie danych, ~13 min i ~640 MB na wariant)."""
    defaults = dict(n_estimators=300, max_depth=18, min_samples_leaf=2,
                    n_jobs=-1, random_state=RANDOM_STATE)
    return RandomForestRegressor(**{**defaults, **params})


def make_xgb(**params):
    from xgboost import XGBRegressor
    defaults = dict(n_estimators=600, max_depth=8, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8,
                    random_state=RANDOM_STATE, n_jobs=-1)
    return XGBRegressor(**{**defaults, **params})


def make_lgbm(**params):
    from lightgbm import LGBMRegressor
    defaults = dict(n_estimators=600, learning_rate=0.05, num_leaves=127,
                    subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                    random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)
    return LGBMRegressor(**{**defaults, **params})


class ResidualRegressor:
    """Model uczy się odchylenia sprzedaży od bieżącego poziomu szeregu, nie samej sprzedaży.

    Poziom = roll_mean_4 (średnia z 4 poprzednich tygodni), gdy brak - lag_1, gdy brak - 0
    (np. wariant bez_historii). Obie cechy są tylko z przeszłości, więc nie ma przecieku.
    Predykcja = poziom + przewidziane odchylenie.
    """
    BASE_COLS = ('roll_mean_4', 'lag_1')

    def __init__(self, estimator, base_cols=BASE_COLS):
        self.estimator = estimator
        self.base_cols = tuple(base_cols)

    def _base(self, X):
        import numpy as np
        cols = [c for c in self.base_cols if c in X.columns]
        if not cols:
            return np.zeros(len(X))
        base = X[cols[0]]
        for c in cols[1:]:
            base = base.fillna(X[c])
        return base.fillna(0).to_numpy(dtype=float)

    def fit(self, X, y, **fit_params):
        import numpy as np
        self.estimator.fit(X, np.asarray(y, dtype=float) - self._base(X), **fit_params)
        return self

    def predict(self, X):
        return self.estimator.predict(X) + self._base(X)

    def __repr__(self):
        return f'ResidualRegressor({type(self.estimator).__name__})'


# Wersje „-L1”: cel = sprzedaż - roll_mean_4, funkcja straty L1 (metryka WMAE to też błąd
# bezwzględny). Na foldach z treningu (test nietknięty, 20.11.2026, lagi_bez_makro_po_swiecie):
# LightGBM 1368 -> 1305, XGBoost 1355 -> 1316; strojony LightGBM na rozszerzone 1313 -> 1274.
# Osobno: L1 -42 / -36, cel względem roll_mean_4 -28 / -11. Wagi świąt x5 w treningu: bez zysku.
def make_lgbm_l1(**params):
    return ResidualRegressor(make_lgbm(**{'objective': 'l1', **params}))


def make_xgb_l1(**params):
    return ResidualRegressor(make_xgb(**{'objective': 'reg:absoluteerror', **params}))


def make_emb_nn_resid(**params):
    """Sieć z embeddingami ucząca się reszty względem roll_mean_4 (strata jak w sieci: WMAE).
    Walidacja 20.11.2026 (domyślne parametry, 4 okna): 1446 -> 1357 WMAE; na starych cechach 1625 -> 1405."""
    return ResidualRegressor(make_emb_nn(**params))


MODELS = {
    'Regresja liniowa': lambda: make_pipeline(StandardScaler(), LinearRegression()),
    'XGBoost': make_xgb,
    'LightGBM': make_lgbm,   # wymaga `pip install lightgbm`
    'XGBoost-L1': make_xgb_l1,      # cel względem roll_mean_4 + strata L1
    'LightGBM-L1': make_lgbm_l1,
    'Sieć z embeddingami': make_emb_nn,                    # wymaga `pip install torch`
    'Sieć z embeddingami-reszta': make_emb_nn_resid,
}

# Modele liczone w siatce eksperymentów (run_experiments z models=None, run_all.py).
# Regresja liniowa = prosty punkt odniesienia. Zwykłe XGBoost/LightGBM/sieć (domyślne parametry)
# pokazują, ile daje uczenie reszty względem roll_mean_4 (wersje -L1 / -reszta).
# Usunięte 20.11.2026 (wyniki w results/archiwum/): Drzewo decyzyjne, Random Forest,
# HistGradientBoosting - zdominowane przez XGBoost/LightGBM na każdym wariancie danych
# (HGB to ten sam algorytm co LightGBM), RF ~13 min na wariant; Ridge (= regresja liniowa
# przy 400 tys. wierszy), sklearnowe MLP (zdominowane przez sieć z embeddingami).
GRID_GPU = ['Sieć z embeddingami', 'Sieć z embeddingami-reszta']
GRID_CPU = ['Regresja liniowa', 'XGBoost', 'LightGBM', 'XGBoost-L1', 'LightGBM-L1']
GRID = GRID_CPU + GRID_GPU

# Modele, które nie przyjmują NaN w cechach (braki uzupełniane medianą z treningu)
NEEDS_IMPUTE = {'Regresja liniowa'}


class SeedBagRegressor:
    """Ten sam model trenowany z kilkoma ziarnami losowości, predykcja = średnia.

    Zmniejsza wariancję wynikającą z losowej inicjalizacji / próbkowania. Na walidacji
    (ostatni fold treningu, 18.11.2026) sieć z embeddingami: 1343 -> 1269 WMAE przy 5 ziarnach,
    LightGBM -10, XGBoost -5.
    `factory(random_state=s)` musi zwracać nowy estymator.
    """

    def __init__(self, factory, seeds=(42, 43, 44, 45, 46)):
        self.factory = factory
        self.seeds = tuple(seeds)

    def fit(self, X, y):
        self.estimators_ = [self.factory(random_state=s).fit(X, y) for s in self.seeds]
        return self

    def predict(self, X):
        import numpy as np
        return np.mean([e.predict(X) for e in self.estimators_], axis=0)

    def __repr__(self):
        return f'SeedBagRegressor({len(self.seeds)} ziaren)'


def fit_timed(model, X, y, name=None):
    t0 = time.time()
    model.fit(X, y)
    print(f"Czas trenowania {name or type(model).__name__}: {time.time()-t0:.1f}s")
    return model


def save_model(model, name):
    MODELS_DIR.mkdir(exist_ok=True)
    path = MODELS_DIR / f'{name}.joblib'
    joblib.dump(model, path)
    return path


def load_model(name):
    return joblib.load(MODELS_DIR / f'{name}.joblib')
