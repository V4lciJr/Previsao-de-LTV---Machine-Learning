"""
Projeto Hashtag Lifetime Value — CRISP-DM
=========================================
Fase 3 — Preparação dos dados para a modelagem.

Entrada: ltv_base_tratada.csv — base já tratada nas etapas anteriores
         (deduplicada por ID, sem as colunas ID e Renda, tipos corrigidos,
         cardinalidade reduzida e data convertida em mes_compra /
         dia_semana_compra).
"""

import os

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# --------------------------------------------------------------------------
# Constantes do projeto
# --------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.20

PASTA = os.path.dirname(os.path.abspath(__file__))
ARQUIVO_BASE = os.path.join(PASTA, "ltv_base_tratada.csv")
ARQUIVO_ARTEFATOS = os.path.join(PASTA, "prep_ltv.joblib")

ALVO = "LTV"

# valor_1_compra é a única contínua de verdade -> StandardScaler
COLS_ESCALAR = ["valor_1_compra"]
# recorrente_1_compra já é binária 0/1 -> passthrough (escalar destruiria a
# leitura direta do coeficiente e não traz ganho)
COLS_PASSTHROUGH = ["recorrente_1_compra"]
# mes_compra e dia_semana_compra entram aqui de propósito: são ordinais
# cíclicas, não numéricas — dezembro não é "maior" que janeiro
COLS_CATEGORICAS = [
    "mes_compra",
    "dia_semana_compra",
    "Produto Fonte",
    "Fonte Campanha",
    "Sexo",
    "Formacao",
]

ORDEM_MES = [
    "Janeiro", "Fevereiro", "Março", "Abril", "Maio", "Junho",
    "Julho", "Agosto", "Setembro", "Outubro", "Novembro", "Dezembro",
]
ORDEM_DIA = [
    "Segunda", "Terça", "Quarta", "Quinta", "Sexta", "Sábado", "Domingo",
]


# --------------------------------------------------------------------------
# Funções
# --------------------------------------------------------------------------
def carregar_base(caminho: str = ARQUIVO_BASE) -> pd.DataFrame:
    """Lê a base tratada e devolve mes/dia da semana como Categorical ordenada."""
    df = pd.read_csv(caminho, encoding="utf-8")
    df["mes_compra"] = pd.Categorical(
        df["mes_compra"], categories=ORDEM_MES, ordered=True
    )
    df["dia_semana_compra"] = pd.Categorical(
        df["dia_semana_compra"], categories=ORDEM_DIA, ordered=True
    )
    return df


def separar_alvo(df: pd.DataFrame):
    """Separa a matriz de features X do vetor alvo y."""
    return df.drop(columns=[ALVO]), df[ALVO]


def construir_preprocessador() -> ColumnTransformer:
    """Monta o ColumnTransformer com os três ramos do pré-processamento.

    Devolve sempre uma instância NOVA e NÃO ajustada — importante para que
    cada fold da validação cruzada refaça o fit apenas com seus próprios
    dados de treino, sem vazamento.
    """
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), COLS_ESCALAR),
            ("bin", "passthrough", COLS_PASSTHROUGH),
            (
                "cat",
                OneHotEncoder(
                    drop="first",
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                COLS_CATEGORICAS,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def separar_treino_teste(X, y):
    """Split 80/20 com semente fixa."""
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)


# --------------------------------------------------------------------------
# Objetos prontos para importação
# --------------------------------------------------------------------------
df = carregar_base()
X, y = separar_alvo(df)
preprocessador = construir_preprocessador()
pipeline_prep = Pipeline(steps=[("preprocessador", construir_preprocessador())])
X_train, X_test, y_train, y_test = separar_treino_teste(X, y)


# --------------------------------------------------------------------------
# Relatório (só roda em execução direta)
# --------------------------------------------------------------------------
def _relatorio() -> None:
    import joblib

    print("=" * 72)
    print("FASE 3 — PREPARAÇÃO DOS DADOS PARA A MODELAGEM")
    print("=" * 72)

    print("\n[1] ALVO E FEATURES")
    print(f"   base        : {df.shape[0]:,} linhas x {df.shape[1]} colunas".replace(",", "."))
    print(f"   alvo (y)    : {ALVO}  |  shape {y.shape}  |  tipo {y.dtype}")
    print(f"   features (X): {X.shape[1]} colunas")
    for c in X.columns:
        print(f"                 - {c}")

    print("\n[2] PRÉ-PROCESSADOR")
    print(f"   StandardScaler -> {COLS_ESCALAR}")
    print(f"   passthrough    -> {COLS_PASSTHROUGH}  (binária, já em escala 0/1)")
    print(f"   OneHotEncoder  -> {COLS_CATEGORICAS}")

    print(f"\n[3] SPLIT TREINO/TESTE (test_size={TEST_SIZE}, random_state={RANDOM_STATE})")
    print(f"   treino: {X_train.shape[0]:,} linhas ({X_train.shape[0]/len(X)*100:.0f}%)".replace(",", "."))
    print(f"   teste : {X_test.shape[0]:,} linhas ({X_test.shape[0]/len(X)*100:.0f}%)".replace(",", "."))
    print(f"   LTV médio  treino R$ {y_train.mean():.2f} | teste R$ {y_test.mean():.2f}"
          f" | diferença R$ {abs(y_train.mean() - y_test.mean()):.2f}")
    print(f"   LTV desvio treino R$ {y_train.std():.2f} | teste R$ {y_test.std():.2f}")

    # fit APENAS no treino — evita vazamento do conjunto de teste
    pipeline_prep.fit(X_train)
    Xtr = pipeline_prep.transform(X_train)
    Xte = pipeline_prep.transform(X_test)
    nomes = pipeline_prep.named_steps["preprocessador"].get_feature_names_out()

    print("\n[4] MATRIZ RESULTANTE")
    print(f"   X_train: {X_train.shape} -> {Xtr.shape}")
    print(f"   X_test : {X_test.shape} -> {Xte.shape}")
    print(f"   {X.shape[1]} features originais -> {Xtr.shape[1]} colunas")

    print(f"\n   FEATURES GERADAS ({len(nomes)}):")
    for i, nome in enumerate(nomes, 1):
        print(f"      {i:2}. {nome}")

    ct = pipeline_prep.named_steps["preprocessador"]
    scaler = ct.named_transformers_["num"]
    print("\n[5] PARÂMETROS APRENDIDOS")
    print("   StandardScaler (ajustado só no treino):")
    print(f"      média  = R$ {scaler.mean_[0]:.4f}")
    print(f"      desvio = R$ {scaler.scale_[0]:.4f}")

    ohe = ct.named_transformers_["cat"]
    print("   OneHotEncoder (drop='first') — categoria de referência:")
    for col, cats in zip(COLS_CATEGORICAS, ohe.categories_):
        print(f"      {col:20} ref = \"{cats[0]}\"  ({len(cats)} cat -> {len(cats)-1} dummies)")

    joblib.dump(
        {
            "pipeline_prep": pipeline_prep,
            "X_train": X_train, "X_test": X_test,
            "y_train": y_train, "y_test": y_test,
            "nomes_features": nomes,
            "RANDOM_STATE": RANDOM_STATE,
        },
        ARQUIVO_ARTEFATOS,
    )
    print(f"\nArtefatos salvos em: {ARQUIVO_ARTEFATOS}")


if __name__ == "__main__":
    _relatorio()
