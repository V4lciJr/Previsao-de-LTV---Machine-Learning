"""
Projeto Hashtag Lifetime Value — CRISP-DM
=========================================
Fase 4 — Modelagem: treino dos 4 modelos e validação cruzada (K-Fold = 5, R²).

Modelos comparados:
    1. DummyRegressor (média) — baseline
    2. Regressão Linear
    3. Regressão Polinomial (grau 2) + Linear
    4. Random Forest

Uso:
    python modelagem.py
"""

import time

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

# --- variáveis importadas da fase de preparação ---------------------------
from preparacao_modelagem import (
    PASTA,
    RANDOM_STATE,
    X_train,
    construir_preprocessador,
    y_train,
)

N_SPLITS = 5
METRICA = "r2"


def construir_modelos() -> dict:
    """Os 4 estimadores que serão comparados."""
    return {
        "1. DummyRegressor (média)": DummyRegressor(strategy="mean"),
        "2. Regressão Linear": LinearRegression(),
        "3. Regressão Polinomial (grau 2) + Linear": Pipeline(
            [
                ("poly", PolynomialFeatures(degree=2, include_bias=False)),
                ("lin", LinearRegression()),
            ]
        ),
        "4. Random Forest": RandomForestRegressor(
            random_state=RANDOM_STATE, n_jobs=-1
        ),
    }


def montar_pipeline(modelo) -> Pipeline:
    """Encadeia o pré-processador com o estimador.

    O pré-processador é sempre uma instância nova e não ajustada: assim o
    StandardScaler e o OneHotEncoder são refeitos dentro de cada fold,
    usando apenas os dados de treino daquele fold (sem vazamento).
    """
    return Pipeline(
        steps=[
            ("preprocessador", construir_preprocessador()),
            ("modelo", clone(modelo)),
        ]
    )


def validacao_cruzada(modelos: dict) -> pd.DataFrame:
    """Roda o K-Fold e devolve a tabela com o R² de cada fold."""
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    linhas, tempos = {}, {}
    for nome, modelo in modelos.items():
        inicio = time.time()
        scores = cross_val_score(
            montar_pipeline(modelo),
            X_train,
            y_train,
            cv=kf,
            scoring=METRICA,
            n_jobs=1,
        )
        tempos[nome] = time.time() - inicio
        linhas[nome] = scores
        print(
            f"   {nome:45} média R² = {scores.mean():.4f}"
            f"   ({tempos[nome]:.1f}s)"
        )

    tabela = pd.DataFrame(
        linhas, index=[f"Fold {i}" for i in range(1, N_SPLITS + 1)]
    ).T
    folds = [f"Fold {i}" for i in range(1, N_SPLITS + 1)]
    tabela["Média"] = tabela[folds].mean(axis=1)
    tabela["Desvio"] = tabela[folds].std(axis=1)
    tabela["Mín"] = tabela[folds].min(axis=1)
    tabela["Máx"] = tabela[folds].max(axis=1)
    tabela["Tempo (s)"] = [round(tempos[n], 1) for n in tabela.index]
    return tabela


def main() -> None:
    print("=" * 78)
    print("FASE 4 — MODELAGEM E VALIDAÇÃO CRUZADA")
    print("=" * 78)
    print(f"   dados de treino : {X_train.shape[0]:,} linhas x {X_train.shape[1]} features".replace(",", "."))
    print(f"   validação       : KFold(n_splits={N_SPLITS}, shuffle=True, "
          f"random_state={RANDOM_STATE})")
    print(f"   métrica         : {METRICA.upper()}\n")

    tabela = validacao_cruzada(construir_modelos())

    print("\n" + "=" * 78)
    print(f"RESULTADO COMPLETO — R² POR FOLD (k={N_SPLITS})")
    print("=" * 78)
    with pd.option_context(
        "display.width", 250, "display.float_format", lambda v: f"{v:,.4f}"
    ):
        print(tabela.to_string())

    melhor = tabela["Média"].idxmax()
    print(f"\n   Melhor R² médio: {melhor}  ({tabela.loc[melhor, 'Média']:.4f})")

    destino = f"{PASTA}/resultados_validacao_cruzada.csv"
    tabela.to_csv(destino, encoding="utf-8-sig", sep=";", decimal=",")
    print(f"   Tabela salva em: {destino}")


if __name__ == "__main__":
    main()
