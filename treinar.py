# Treino e comparação dos modelos (Tema 05 - Preço de imóveis, regressão)
# X = área habitável, quartos, ano de construção e bairro | y = preço
# Execução: python treinar.py

import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, TargetEncoder
from sklearn.tree import DecisionTreeRegressor

# CONFIGURAÇÕES

PASTA_PROJETO = Path(__file__).parent
CAMINHO_DADOS = PASTA_PROJETO / "perth_house_prices_tratado.csv"
PASTA_MODELOS = PASTA_PROJETO / "models"

COLUNAS_NUMERICAS = ["area_habitavel", "quartos", "ano_construcao"]
COLUNAS_CATEGORICAS = ["bairro"]
COLUNAS_X = COLUNAS_NUMERICAS + COLUNAS_CATEGORICAS
COLUNA_Y = "preco"

# 80% treino e 20% teste; o random_state fixo deixa a divisão sempre igual
TAMANHO_TESTE = 0.20
RANDOM_STATE = 42


# LEITURA DOS DADOS

def carregar_dados():
    df = pd.read_csv(CAMINHO_DADOS)
    print(f"Base tratada carregada: {len(df)} registros e {df.shape[1]} colunas")

    # Remove duplicadas antes da divisão, para a mesma casa não ficar no treino e no teste
    duplicadas = df.duplicated().sum()
    df = df.drop_duplicates()
    print(f"Linhas duplicadas removidas: {duplicadas}")
    print(f"Registros usados no Machine Learning: {len(df)}")
    return df


# PRÉ-PROCESSAMENTO

def criar_preprocessador(codificacao_bairro):
    # Numéricas: preenche ausentes com a mediana do treino
    # Bairro: transforma o texto em número
    #   "onehot": uma coluna 0/1 para cada bairro (usado na regressão linear)
    #   "preco_medio": o bairro vira o preço médio dele no treino (usado nas árvores,
    #   que funcionam mal com centenas de colunas 0/1)
    # Dentro do Pipeline, o fit acontece só com os dados de treino
    if codificacao_bairro == "onehot":
        codificador = OneHotEncoder(handle_unknown="ignore")
    else:
        codificador = TargetEncoder(
            target_type="continuous",
            cv=KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
        )

    return ColumnTransformer([
        ("numericas", SimpleImputer(strategy="median"), COLUNAS_NUMERICAS),
        ("bairro", codificador, COLUNAS_CATEGORICAS),
    ])


# MODELOS

def criar_modelos():
    # Baseline: sempre prevê a mediana do treino, serve só de referência
    # min_samples_leaf (mínimo de casas por folha) evita que a árvore decore o treino;
    # os valores foram escolhidos com validação cruzada apenas no treino
    return {
        "Baseline (mediana)": (DummyRegressor(strategy="median"), "onehot"),
        "Linear Regression": (LinearRegression(), "onehot"),
        "Decision Tree Regressor": (
            DecisionTreeRegressor(min_samples_leaf=60, random_state=RANDOM_STATE),
            "preco_medio",
        ),
        # 100 árvores; a previsão é a média delas
        "Random Forest Regressor": (
            RandomForestRegressor(n_estimators=100, min_samples_leaf=10, random_state=RANDOM_STATE, n_jobs=-1),
            "preco_medio",
        ),
    }


# AVALIAÇÃO

def calcular_metricas(y_real, y_previsto):
    # MAE: erro médio em AUD (métrica principal)
    # RMSE: como o MAE, mas pesa mais os erros grandes
    # R²: quanto da variação do preço o modelo explica (1 = perfeito)
    return {
        "mae": float(mean_absolute_error(y_real, y_previsto)),
        "rmse": float(np.sqrt(mean_squared_error(y_real, y_previsto))),
        "r2": float(r2_score(y_real, y_previsto)),
    }


def erro_por_faixa_de_preco(y_real, y_previsto):
    # Divide o teste em 4 faixas de preço real e calcula o erro de cada uma
    tabela = pd.DataFrame({"real": y_real, "previsto": y_previsto})
    tabela["erro_absoluto"] = (tabela["real"] - tabela["previsto"]).abs()
    tabela["faixa"] = pd.qcut(
        tabela["real"], q=4,
        labels=["25% mais baratas", "25%-50%", "50%-75%", "25% mais caras"],
    )
    resumo = tabela.groupby("faixa", observed=True).agg(
        casas=("real", "size"),
        preco_medio=("real", "mean"),
        mae=("erro_absoluto", "mean"),
    )
    resumo["erro_percentual"] = resumo["mae"] / resumo["preco_medio"] * 100
    return resumo


def main():
    print(80 * "=")
    print("TREINAMENTO DOS MODELOS - PREÇO DE IMÓVEIS")
    print(80 * "=")

    df = carregar_dados()
    print()

    # SEPARAÇÃO DE X E y (o preço não entra em X)
    X = df[COLUNAS_X]
    y = df[COLUNA_Y]

    # DIVISÃO TREINO / TESTE
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TAMANHO_TESTE, random_state=RANDOM_STATE
    )
    print(f"Treino: {len(X_train)} casas ({1 - TAMANHO_TESTE:.0%}) | "
          f"Teste: {len(X_test)} casas ({TAMANHO_TESTE:.0%}) | random_state={RANDOM_STATE}")
    print(f"Preço no teste: média AU$ {y_test.mean():,.0f} | mediana AU$ {y_test.median():,.0f}")
    print()

    # TREINO E AVALIAÇÃO DE CADA MODELO
    resultados = {}
    pipelines = {}
    previsoes_teste = pd.DataFrame({"preco_real": y_test.values}, index=y_test.index)
    previsoes_teste = pd.concat([X_test, previsoes_teste], axis=1)

    for nome, (estimador, codificacao_bairro) in criar_modelos().items():
        pipeline = Pipeline([
            ("preprocessamento", criar_preprocessador(codificacao_bairro)),
            ("modelo", estimador),
        ])

        # Validação cruzada em 5 partes do treino, para ver se o resultado é estável
        mae_cv = -cross_val_score(
            pipeline, X_train, y_train, cv=5, scoring="neg_mean_absolute_error"
        ).mean()

        # Treina só com o treino e avalia no treino e no teste
        pipeline.fit(X_train, y_train)

        metricas_treino = calcular_metricas(y_train, pipeline.predict(X_train))
        y_previsto = pipeline.predict(X_test)
        metricas_teste = calcular_metricas(y_test, y_previsto)

        resultados[nome] = {
            "codificacao_bairro": codificacao_bairro,
            "treino": metricas_treino,
            "validacao_cruzada_treino": {"mae": float(mae_cv)},
            "teste": metricas_teste,
        }
        pipelines[nome] = pipeline
        previsoes_teste[f"previsto_{nome}"] = y_previsto

    # TABELA DE COMPARAÇÃO
    tabela = pd.DataFrame({
        nome: {
            "MAE treino": r["treino"]["mae"],
            "MAE valid. cruzada": r["validacao_cruzada_treino"]["mae"],
            "MAE teste": r["teste"]["mae"],
            "RMSE teste": r["teste"]["rmse"],
            "R² teste": r["teste"]["r2"],
        }
        for nome, r in resultados.items()
    }).T
    # Quanto maior a diferença, mais o modelo decorou o treino (overfitting)
    tabela["Diferença MAE (teste - treino)"] = tabela["MAE teste"] - tabela["MAE treino"]

    print("COMPARAÇÃO DOS MODELOS (mesma divisão treino/teste):")
    with pd.option_context("display.float_format", "{:,.2f}".format,
                           "display.width", 250, "display.max_columns", None):
        print(tabela)
    print()

    # ESCOLHA DO MODELO FINAL: menor MAE no teste (o baseline não entra)
    candidatos = {n: r for n, r in resultados.items() if not n.startswith("Baseline")}
    escolhido = min(candidatos, key=lambda n: candidatos[n]["teste"]["mae"])
    mae_escolhido = resultados[escolhido]["teste"]["mae"]
    mae_baseline = resultados["Baseline (mediana)"]["teste"]["mae"]
    melhora = (1 - mae_escolhido / mae_baseline) * 100

    print(f"MODELO ESCOLHIDO: {escolhido}")
    print(f"  MAE no teste: AU$ {mae_escolhido:,.0f} "
          f"({mae_escolhido / y_test.mean():.1%} do preço médio)")
    print(f"  Erra {melhora:.1f}% menos que o baseline (AU$ {mae_baseline:,.0f})")
    print()

    # ERRO DO MODELO ESCOLHIDO POR FAIXA DE PREÇO
    faixas = erro_por_faixa_de_preco(y_test, previsoes_teste[f"previsto_{escolhido}"])
    print(f"ERRO DO MODELO ESCOLHIDO POR FAIXA DE PREÇO REAL ({escolhido}):")
    with pd.option_context("display.float_format", "{:,.2f}".format, "display.width", 200):
        print(faixas)
    print()

    # SALVAMENTO
    PASTA_MODELOS.mkdir(exist_ok=True)

    # O pipeline salvo leva o pré-processamento junto, então o Flask usa o mesmo preparo
    joblib.dump(pipelines[escolhido], PASTA_MODELOS / "modelo.joblib", compress=3)

    # Real x previsto no teste, para os gráficos
    previsoes_teste.to_csv(PASTA_MODELOS / "previsoes_teste.csv", index=False)

    # Informações do treino; bairros e faixas válidas são usados no formulário do Flask
    metricas = {
        "data_treino": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "base_treino": CAMINHO_DADOS.name,
        "registros_usados": len(df),
        "tamanho_treino": len(X_train),
        "tamanho_teste": len(X_test),
        "proporcao_teste": TAMANHO_TESTE,
        "random_state": RANDOM_STATE,
        "versao_scikit_learn": sklearn.__version__,
        "colunas_x": COLUNAS_X,
        "coluna_y": COLUNA_Y,
        "metrica_principal": "MAE",
        "moeda": "AUD (dólar australiano)",
        "resultados": resultados,
        "modelo_escolhido": escolhido,
        "mae_teste_modelo_escolhido": mae_escolhido,
        "erro_por_faixa_de_preco": faixas.reset_index().astype({"faixa": str}).to_dict(orient="records"),
        "bairros": sorted(X_train["bairro"].unique().tolist()),
        "faixas_validas": {
            coluna: {"min": float(X_train[coluna].min()), "max": float(X_train[coluna].max())}
            for coluna in COLUNAS_NUMERICAS
        },
    }
    with open(PASTA_MODELOS / "metricas.json", "w", encoding="utf-8") as arquivo:
        json.dump(metricas, arquivo, ensure_ascii=False, indent=2)

    print("Arquivos gerados em models/:")
    print("  - modelo.joblib        (pipeline com pré-processamento + modelo escolhido)")
    print("  - metricas.json        (resultados, modelo escolhido, bairros e faixas válidas)")
    print("  - previsoes_teste.csv  (preço real x previsto de cada modelo no teste)")


if __name__ == "__main__":
    main()
