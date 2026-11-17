"""Sieć neuronowa z embeddingami dla cech kategorycznych (PyTorch), zgodna z API sklearn.

Architektura:

    Store ─► Embedding(45 -> d1) ─┐
    Dept  ─► Embedding(81 -> d2) ─┼─► concat ─► [Linear ─► BatchNorm ─► aktywacja ─► Dropout] x N ─► Linear ─► ŷ
    Type  ─► Embedding(3  -> d3) ─┤
    cechy liczbowe (standaryzowane) ┘

Embedding uczy gęstej reprezentacji sklepu/działu (podobne działy -> bliskie wektory),
zamiast setek rzadkich kolumn one-hot. Dzięki zgodności ze sklearn model działa
w `run_experiments` i w `tune` jak każdy inny.
"""
import copy
import time

import numpy as np
import pandas as pd
import torch
from sklearn.base import BaseEstimator, RegressorMixin
from torch import nn

from .config import RANDOM_STATE

ACTIVATIONS = {'relu': nn.ReLU, 'gelu': nn.GELU, 'silu': nn.SiLU, 'tanh': nn.Tanh,
               'leaky_relu': nn.LeakyReLU, 'elu': nn.ELU}


def _per_layer(value, n):
    """Wartość wspólna dla wszystkich warstw albo osobna dla każdej (lista/krotka)."""
    if isinstance(value, (list, tuple)):
        if len(value) != n:
            raise ValueError(f'Oczekiwano {n} wartości (po jednej na warstwę), dostano {len(value)}')
        return list(value)
    return [value] * n


class EmbeddingNet(nn.Module):
    def __init__(self, cardinalities, emb_dims, n_num, hidden, activation, dropout, batch_norm):
        super().__init__()
        self.embeddings = nn.ModuleList(nn.Embedding(c, d) for c, d in zip(cardinalities, emb_dims))
        layers, width = [], sum(emb_dims) + n_num
        for units, act, p in zip(hidden, _per_layer(activation, len(hidden)),
                                 _per_layer(dropout, len(hidden))):
            layers.append(nn.Linear(width, units))
            if batch_norm:
                layers.append(nn.BatchNorm1d(units))
            layers.append(ACTIVATIONS[act]())
            if p > 0:
                layers.append(nn.Dropout(p))
            width = units
        layers.append(nn.Linear(width, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x_cat, x_num):
        emb = [e(x_cat[:, i]) for i, e in enumerate(self.embeddings)]
        return self.mlp(torch.cat(emb + [x_num], dim=1)).squeeze(1)


class EmbeddingMLPRegressor(RegressorMixin, BaseEstimator):
    """Regresor: embeddingi dla `cat_cols` + perceptron wielowarstwowy.

    hidden      - neurony w kolejnych warstwach ukrytych, np. (256, 128, 64)
    activation  - jedna nazwa albo lista na warstwę, np. ('relu', 'relu', 'tanh')
    dropout     - jedna wartość albo lista na warstwę, np. (0.3, 0.2, 0.0)
    emb_dim_max - górny limit wymiaru embeddingu; wymiar = min(emb_dim_max, (liczba_kategorii+1)//2)
    loss        - 'wmae' (L1 z wagą 5 dla tygodni świątecznych = metryka oceny), 'mae', 'mse', 'huber'
    Wczesne zatrzymanie: walidacja na najpóźniejszych `val_fraction` wierszy (wg Year, WeekOfYear),
    czyli zgodnie z czasem, nie losowo.
    """

    def __init__(self, cat_cols=('Store', 'Dept', 'Type'), hidden=(256, 128, 64),
                 activation='relu', dropout=0.2, batch_norm=True, emb_dim_max=16,
                 loss='wmae', lr=1e-3, weight_decay=1e-5, batch_size=1024, epochs=60,
                 patience=6, val_fraction=0.1, device='auto', random_state=RANDOM_STATE,
                 verbose=False):
        self.cat_cols = cat_cols
        self.hidden = hidden
        self.activation = activation
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.emb_dim_max = emb_dim_max
        self.loss = loss
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.val_fraction = val_fraction
        self.device = device
        self.random_state = random_state
        self.verbose = verbose

    # --- przygotowanie danych -------------------------------------------------
    def _device(self):
        if self.device == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.device(self.device)

    def _encode(self, X):
        """DataFrame -> (indeksy kategorii, standaryzowane liczby). Nieznana kategoria -> 0."""
        x_cat = np.stack([X[c].map(self.cat_maps_[c]).fillna(0).to_numpy(np.int64)
                          for c in self.cat_cols_], axis=1) if self.cat_cols_ else \
            np.zeros((len(X), 0), np.int64)
        x_num = X[self.num_cols_].astype(float).fillna(self.num_median_)
        x_num = ((x_num - self.num_mean_) / self.num_std_).to_numpy(np.float32)
        return torch.from_numpy(x_cat), torch.from_numpy(x_num)

    def _loss_fn(self, pred, y, w):
        if self.loss in ('wmae', 'mae'):
            err = (pred - y).abs()
        elif self.loss == 'mse':
            err = (pred - y) ** 2
        elif self.loss == 'huber':
            err = nn.functional.huber_loss(pred, y, reduction='none')
        else:
            raise ValueError(f'Nieznana funkcja straty: {self.loss}')
        return (err * w).sum() / w.sum()

    # --- trening ------------------------------------------------------------
    def fit(self, X, y):
        if not isinstance(X, pd.DataFrame):
            raise TypeError('EmbeddingMLPRegressor wymaga DataFrame (potrzebne nazwy kolumn)')
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        dev = self._device()
        y = np.asarray(y, dtype=np.float32)

        self.cat_cols_ = [c for c in self.cat_cols if c in X.columns]
        self.num_cols_ = [c for c in X.columns if c not in self.cat_cols_]
        # indeks 0 zarezerwowany na kategorie niewidziane w treningu
        self.cat_maps_ = {c: {v: i + 1 for i, v in enumerate(np.sort(X[c].dropna().unique()))}
                          for c in self.cat_cols_}
        cards = [len(self.cat_maps_[c]) + 1 for c in self.cat_cols_]
        emb_dims = [min(self.emb_dim_max, (k + 1) // 2) for k in cards]
        num = X[self.num_cols_].astype(float)
        self.num_median_ = num.median()
        self.num_mean_ = num.fillna(self.num_median_).mean()
        self.num_std_ = num.fillna(self.num_median_).std().replace(0, 1).fillna(1)
        self.y_mean_, self.y_std_ = float(y.mean()), float(y.std() or 1)

        x_cat, x_num = self._encode(X)
        y_t = torch.from_numpy((y - self.y_mean_) / self.y_std_)
        w_t = torch.ones(len(X))
        if self.loss == 'wmae' and 'IsHoliday' in X.columns:
            w_t = torch.from_numpy(np.where(X['IsHoliday'].to_numpy() == 1, 5.0, 1.0).astype(np.float32))

        # podział na trening / walidację do wczesnego zatrzymania - zgodnie z czasem
        n = len(X)
        if {'Year', 'WeekOfYear'} <= set(X.columns):
            order = np.lexsort((X['WeekOfYear'].to_numpy(), X['Year'].to_numpy()))
        else:
            order = np.random.permutation(n)
        n_val = int(n * self.val_fraction) if self.patience else 0
        tr_idx, va_idx = order[:n - n_val], order[n - n_val:]

        self.model_ = EmbeddingNet(cards, emb_dims, len(self.num_cols_), list(self.hidden),
                                   self.activation, self.dropout, self.batch_norm).to(dev)
        opt = torch.optim.AdamW(self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2)

        data = [t.to(dev) for t in (x_cat, x_num, y_t, w_t)]
        tr_idx_t = torch.from_numpy(tr_idx).to(dev)
        va = [t[torch.from_numpy(va_idx).to(dev)] for t in data] if n_val else None

        self.history_ = []
        best, best_state, bad = np.inf, None, 0
        for epoch in range(self.epochs):
            t0 = time.time()
            self.model_.train()
            perm = tr_idx_t[torch.randperm(len(tr_idx_t), device=dev)]
            total, count = 0.0, 0
            for s in range(0, len(perm), self.batch_size):
                b = perm[s:s + self.batch_size]
                if len(b) < 2 and self.batch_norm:
                    continue   # BatchNorm nie działa na batchu o rozmiarze 1
                xc, xn, yb, wb = (t[b] for t in data)
                loss = self._loss_fn(self.model_(xc, xn), yb, wb)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += loss.item() * len(b)
                count += len(b)
            rec = {'epoka': epoch + 1, 'train_loss': total / count, 'czas_s': time.time() - t0}

            if va is not None:
                val_loss = self._eval_loss(*va)
                rec['val_loss'] = val_loss
                sched.step(val_loss)
                if val_loss < best - 1e-5:
                    best, bad = val_loss, 0
                    best_state = copy.deepcopy(self.model_.state_dict())
                else:
                    bad += 1
            self.history_.append(rec)
            if self.verbose:
                print(f"epoka {rec['epoka']:3d}  train={rec['train_loss']:.4f}  "
                      f"val={rec.get('val_loss', float('nan')):.4f}  ({rec['czas_s']:.1f}s)")
            if va is not None and bad >= self.patience:
                break

        if best_state is not None:
            self.model_.load_state_dict(best_state)
        self.model_.cpu().eval()   # przechowywany na CPU - łatwy zapis joblib i predykcja wszędzie
        self.history_ = pd.DataFrame(self.history_)
        return self

    @torch.no_grad()
    def _eval_loss(self, xc, xn, y, w):
        self.model_.eval()
        preds = torch.cat([self.model_(xc[s:s + 8192], xn[s:s + 8192])
                           for s in range(0, len(y), 8192)])
        return self._loss_fn(preds, y, w).item()

    @torch.no_grad()
    def predict(self, X):
        x_cat, x_num = self._encode(X)
        dev = self._device()
        self.model_.to(dev).eval()
        out = torch.cat([self.model_(x_cat[s:s + 8192].to(dev), x_num[s:s + 8192].to(dev)).cpu()
                         for s in range(0, len(X), 8192)])
        self.model_.cpu()
        return out.numpy() * self.y_std_ + self.y_mean_

    def embedding_weights(self, col):
        """Macierz embeddingu dla kolumny (DataFrame: wiersz = kategoria) - do analizy/wizualizacji."""
        i = self.cat_cols_.index(col)
        W = self.model_.embeddings[i].weight.detach().numpy()[1:]   # bez indeksu 'nieznany'
        return pd.DataFrame(W, index=list(self.cat_maps_[col]))
