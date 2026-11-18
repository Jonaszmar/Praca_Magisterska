"""Rejestr modeli. Nowy model: `MODELS['nazwa'] = funkcja_zwracająca_estymator`."""
import time

import joblib
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

from .config import MODELS_DIR, RANDOM_STATE

# Cechy kategoryczne zakodowane liczbami - dla sieci kodujemy je one-hot
CATEGORICAL = ['Store', 'Dept', 'Type']


def _cat_cols(X):
    return [c for c in CATEGORICAL if c in X.columns]


def _num_cols(X):
    return [c for c in X.columns if c not in CATEGORICAL]


def make_mlp(hidden=(128, 64), onehot=True, **params):
    """Perceptron wielowarstwowy; `hidden` = liczby neuronów w kolejnych warstwach ukrytych.

    Np. hidden=(256, 128, 64) -> trzy warstwy ukryte. Wejście: one-hot dla Store/Dept/Type
    i standaryzacja reszty; cel (sprzedaż) też standaryzowany - bez tego sieć uczy się źle.
    """
    defaults = dict(hidden_layer_sizes=tuple(hidden), activation='relu', alpha=1e-4,
                    learning_rate_init=1e-3, batch_size=512, max_iter=200,
                    early_stopping=True, validation_fraction=0.1, n_iter_no_change=10,
                    random_state=RANDOM_STATE)
    pre = (ColumnTransformer([('cat', OneHotEncoder(handle_unknown='ignore'), _cat_cols),
                              ('num', StandardScaler(), _num_cols)])
           if onehot else StandardScaler())
    net = make_pipeline(pre, MLPRegressor(**{**defaults, **params}))
    return TransformedTargetRegressor(regressor=net, transformer=StandardScaler())


def make_emb_nn(**params):
    """Sieć PyTorch z embeddingami Store/Dept/Type - szczegóły w src/nn.py."""
    from .nn import EmbeddingMLPRegressor   # wymaga `pip install torch`
    return EmbeddingMLPRegressor(**params)


def make_rf(**params):
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


def make_hgb(**params):
    defaults = dict(max_iter=600, learning_rate=0.05, max_leaf_nodes=127, random_state=RANDOM_STATE)
    return HistGradientBoostingRegressor(**{**defaults, **params})


MODELS = {
    'Regresja liniowa': lambda: make_pipeline(StandardScaler(), LinearRegression()),
    'Ridge': lambda: make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
    'Drzewo decyzyjne': lambda: DecisionTreeRegressor(max_depth=14, min_samples_leaf=5,
                                                      random_state=RANDOM_STATE),
    'Random Forest': make_rf,
    'HistGradientBoosting': make_hgb,
    'XGBoost': make_xgb,
    'LightGBM': make_lgbm,   # wymaga `pip install lightgbm`
    'MLP': make_mlp,                                        # 2 warstwy: 128-64
    'MLP 3 warstwy': lambda: make_mlp(hidden=(256, 128, 64)),
    'Sieć z embeddingami': make_emb_nn,                    # wymaga `pip install torch`
}

# Modele liczone w siatce eksperymentów (run_experiments z models=None, run_all.py).
# Sklearnowe MLP zostają w rejestrze (można je trenować/stroić ręcznie), ale nie w siatce -
# są wolne na CPU i zdominowane przez sieć z embeddingami.
GRID_GPU = ['Sieć z embeddingami']
# Ridge usunięty z siatki: przy ~400 tys. wierszy alpha=1 nic nie zmienia - wyniki były
# identyczne z regresją liniową (przebieg z 18.11.2026). Zostaje w rejestrze.
GRID_CPU = ['Regresja liniowa', 'Drzewo decyzyjne', 'Random Forest',
            'HistGradientBoosting', 'XGBoost', 'LightGBM']
GRID = GRID_CPU + GRID_GPU

# Modele, które nie przyjmują NaN w cechach (braki uzupełniane medianą z treningu)
NEEDS_IMPUTE = {'Regresja liniowa', 'Ridge', 'Drzewo decyzyjne', 'Random Forest',
                'MLP', 'MLP 3 warstwy'}


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
