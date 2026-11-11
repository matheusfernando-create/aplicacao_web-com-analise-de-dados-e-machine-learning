# Carrega o modelo treinado (treinar.py) e faz a previsão de preço para o Flask
# Uso no Flask:
#   valores, erros = validar_entrada(request.form)
#   if not erros:
#       preco = prever(valores)

import json
from pathlib import Path

import joblib
import pandas as pd

PASTA_MODELOS = Path(__file__).parent / "models"
CAMINHO_MODELO = PASTA_MODELOS / "modelo.joblib"
CAMINHO_METRICAS = PASTA_MODELOS / "metricas.json"

# Guarda o modelo e as métricas na memória depois da primeira leitura
_modelo = None
_metricas = None


def carregar_modelo():
    # Pipeline salvo: pré-processamento + modelo
    global _modelo
    if _modelo is None:
        if not CAMINHO_MODELO.exists():
            raise FileNotFoundError(
                "Modelo não encontrado. Execute 'python treinar.py' antes de iniciar a aplicação."
            )
        _modelo = joblib.load(CAMINHO_MODELO)
    return _modelo


def carregar_metricas():
    # Resultados do treino, bairros e faixas válidas
    global _metricas
    if _metricas is None:
        if not CAMINHO_METRICAS.exists():
            raise FileNotFoundError(
                "Métricas não encontradas. Execute 'python treinar.py' antes de iniciar a aplicação."
            )
        with open(CAMINHO_METRICAS, encoding="utf-8") as arquivo:
            _metricas = json.load(arquivo)
    return _metricas


def listar_bairros():
    # Bairros vistos no treino, para o <select> do formulário
    return carregar_metricas()["bairros"]


def faixas_validas():
    # Mínimo e máximo de cada variável numérica no treino
    return carregar_metricas()["faixas_validas"]


def validar_entrada(dados):
    # Confere os campos do formulário e retorna (valores convertidos, lista de erros)
    faixas = faixas_validas()
    erros = []
    valores = {}

    # Numéricas: precisam ser números dentro da faixa vista no treino
    nomes = {
        "area_habitavel": ("Área habitável", "m²", float),
        "quartos": ("Quartos", "", int),
        "ano_construcao": ("Ano de construção", "", int),
    }
    for coluna, (nome, unidade, tipo) in nomes.items():
        texto = str(dados.get(coluna, "")).strip().replace(",", ".")
        if texto == "":
            erros.append(f"{nome}: campo obrigatório.")
            continue
        try:
            numero = float(texto)
        except ValueError:
            erros.append(f"{nome}: digite um número válido.")
            continue
        if tipo is int and not numero.is_integer():
            erros.append(f"{nome}: digite um número inteiro.")
            continue

        minimo = faixas[coluna]["min"]
        maximo = faixas[coluna]["max"]
        if not minimo <= numero <= maximo:
            erros.append(f"{nome}: informe um valor entre {minimo:g} e {maximo:g} {unidade}".strip() + ".")
            continue
        valores[coluna] = tipo(numero)

    # Bairro: precisa existir no treino
    bairro = str(dados.get("bairro", "")).strip()
    if bairro == "":
        erros.append("Bairro: campo obrigatório.")
    elif bairro not in listar_bairros():
        erros.append("Bairro: escolha um bairro da lista.")
    else:
        valores["bairro"] = bairro

    return valores, erros


def prever(valores):
    # Recebe os valores já validados e devolve o preço previsto em AUD
    modelo = carregar_modelo()
    colunas = carregar_metricas()["colunas_x"]

    # Monta uma linha com as mesmas colunas e a mesma ordem do treino;
    # o pipeline aplica o mesmo pré-processamento usado no treino
    linha = pd.DataFrame([[valores[c] for c in colunas]], columns=colunas)
    preco = modelo.predict(linha)[0]

    # Evita preço negativo
    return max(float(preco), 0.0)


if __name__ == "__main__":
    # Teste rápido: python modelo.py
    exemplo = {"area_habitavel": "172", "quartos": "4", "ano_construcao": "1995", "bairro": "Wandi"}
    valores, erros = validar_entrada(exemplo)
    if erros:
        print("Erros:", erros)
    else:
        metricas = carregar_metricas()
        print("Entrada:", valores)
        print(f"Modelo usado: {metricas['modelo_escolhido']}")
        print(f"Preço previsto: AU$ {prever(valores):,.2f}")
        print(f"Erro médio esperado (MAE no teste): AU$ {metricas['mae_teste_modelo_escolhido']:,.2f}")

    # Exemplo de entrada inválida
    print(validar_entrada({"area_habitavel": "abc", "quartos": "2.5", "ano_construcao": "1500", "bairro": "Inexistente"})[1])
