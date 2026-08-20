# 💰 Hashtag Lifetime Value — Previsão de LTV para otimização do CAC

Projeto de ciência de dados estruturado pela metodologia **CRISP-DM**, desenvolvido na
disciplina *Regressões Lineares* (Módulo III — Data Analysis and Predictive Insights)
da pós-graduação em Data Analytics e Inteligência Artificial Aplicada a Negócios.

## 🎯 Problema de negócio

A Hashtag Treinamentos precisa decidir **quanto pode pagar para adquirir cada novo aluno**
sem comprometer a rentabilidade futura. Otimizar campanhas apenas pelo menor custo por
compra leva a comprar clientes baratos que não renovam — e a desligar campanhas caras que
atraem clientes de alto valor.

A solução é estimar o **LTV esperado já na data da primeira compra**, e converter essa
previsão em um teto de CAC por segmento.

- 📈 **Tipo de tarefa:** regressão supervisionada (alvo contínuo, em R$)
- 👤 **Unidade de análise:** o cliente (1 linha = 1 cliente)
- 🎯 **Alvo:** `LTV` — valor acumulado pelo cliente
- 🚫 **Restrição central:** somente features disponíveis no instante da primeira compra
  (nenhuma variável posterior — isso seria *data leakage*)

## 📏 Métricas de sucesso

| Métrica | Por que |
|---|---|
| 💵 **MAE** | Erro médio em reais — direto de interpretar pelo negócio |
| ⚖️ **RMSE** | Penaliza mais os erros grandes |
| 📐 **R²** | Proporção da variância do LTV explicada pelo modelo |

## 📁 Estrutura

```
codigo/
├── 🗃️ ltv_base_tratada.csv     # base de entrada (38.753 clientes x 9 colunas)
├── 🐍 preparacao_modelagem.py  # Fase 3 — pré-processamento e split treino/teste
├── 🐍 modelagem.py             # Fase 4 — treino dos modelos e validação cruzada
├── 📦 requirements.txt
└── 📄 README.md
```

`modelagem.py` **importa** de `preparacao_modelagem.py`. Não há duplicação de lógica:
existe uma única definição do pré-processamento e uma única definição do split, e as
duas são reaproveitadas por qualquer script posterior (avaliação, deploy).

## ▶️ Como rodar

```bash
pip install -r requirements.txt

python preparacao_modelagem.py   # relatório da preparação + prep_ltv.joblib
python modelagem.py              # validação cruzada dos 4 modelos
```

`modelagem.py` roda sozinho — ao importar o módulo de preparação, os dados já vêm
carregados e separados.

## 🗃️ A base

38.753 clientes, compras entre 01/01/2023 e 31/05/2024. Já passou pelos tratamentos das
etapas anteriores: deduplicação por ID, remoção das colunas `ID` e `Renda`, correção de
tipos, consolidação de valores mascarados, redução de cardinalidade e extração de
`mes_compra` / `dia_semana_compra` a partir da data.

| Coluna | Tipo | Papel |
|---|---|---|
| 🎯 `LTV` | float | **alvo** |
| `valor_1_compra` | float | numérica contínua |
| `recorrente_1_compra` | int (0/1) | binária |
| `mes_compra` | texto (12) | categórica |
| `dia_semana_compra` | texto (7) | categórica |
| `Produto Fonte` | texto (10) | categórica |
| `Fonte Campanha` | texto (7) | categórica |
| `Sexo` | texto (3) | categórica |
| `Formacao` | texto (6) | categórica |

## 🔧 Pré-processamento

`ColumnTransformer` com três ramos:

| Ramo | Transformador | Colunas |
|---|---|---|
| 🔢 `num` | `StandardScaler` | `valor_1_compra` |
| 🔘 `bin` | `passthrough` | `recorrente_1_compra` |
| 🏷️ `cat` | `OneHotEncoder(drop='first')` | as 6 categóricas |

`recorrente_1_compra` não é escalada de propósito: já é 0/1, na mesma escala das dummies,
e padronizá-la só atrapalharia a leitura do coeficiente. `mes_compra` e
`dia_semana_compra` vão para o One-Hot, e não para o scaler, porque são **ordinais
cíclicas** — dezembro não é "maior" que janeiro.

Resultado: **8 features originais → 41 colunas**.

Split: 80/20 (`random_state=42`) → 31.002 linhas de treino, 7.751 de teste.

## 📊 Resultados — validação cruzada (K-Fold = 5, R²)

| Modelo | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | **Média** | Desvio |
|---|---|---|---|---|---|---|---|
| 🎲 DummyRegressor (média) | −0,0001 | −0,0026 | −0,0011 | −0,0000 | −0,0004 | **−0,0009** | 0,0011 |
| 🏆 **Regressão Linear** | 0,8489 | 0,8440 | 0,8533 | 0,8486 | 0,8572 | **0,8504** | 0,0050 |
| 🥈 Polinomial (g2) + Linear | 0,8431 | 0,8398 | 0,8502 | 0,8456 | 0,8534 | **0,8464** | 0,0054 |
| 🌳 Random Forest | 0,8215 | 0,8148 | 0,8314 | 0,8253 | 0,8340 | **0,8254** | 0,0077 |

🥇 A **regressão linear** lidera, com o menor desvio entre folds. Os termos quadráticos não
trazem ganho e o Random Forest — flexível e não-linear — fica atrás das duas regressões:
a estrutura do problema é linear.

⚠️ **Nota técnica:** a matriz polinomial tem posto 622 de 902 colunas (multicolinearidade
severa, inevitável porque o quadrado de uma dummy é igual a ela mesma). O
`LinearRegression` do scikit-learn não explode porque resolve por mínimos quadrados com
pseudo-inversa; um solver por inversão direta produziria R² fortemente negativo e exigiria
regularização (`Ridge`).

## 🔁 Reprodutibilidade

Todas as fontes de aleatoriedade estão travadas em `random_state=42`
(`train_test_split`, `KFold(shuffle=True)`, `RandomForestRegressor`). Os resultados
acima foram conferidos contra uma implementação manual do K-Fold, com o R² calculado
pela fórmula — diferença máxima de 1,11e−16 (ruído de ponto flutuante).

🐍 Ambiente de referência: Python 3.11, scikit-learn 1.8, pandas 3.0, NumPy 2.4.
