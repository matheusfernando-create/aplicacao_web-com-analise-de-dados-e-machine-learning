# Leitura, diagnóstico e tratamento dos dados (Tema 05 - Preço de imóveis)
# Execução: python tratamento_dados.py
#
# - O CSV original (perth_house_prices.csv) NÃO é alterado: o resultado vai para
#   perth_house_prices_tratado.csv (raiz e pasta data/).
# - Cada transformação fica registrada em relatorios/registro_tratamento.md,
#   com o motivo e quantas linhas ela afetou.
# - As funções diagnosticar() e tratar() recebem um DataFrame, então também podem
#   ser usadas pelo Flask para um CSV enviado pelo usuário.

from pathlib import Path

import pandas as pd

PASTA_PROJETO = Path(__file__).parent
CAMINHO_ORIGINAL = PASTA_PROJETO / "perth_house_prices.csv"
CAMINHOS_TRATADO = [
    PASTA_PROJETO / "perth_house_prices_tratado.csv",        # lido pelo treinar.py
    PASTA_PROJETO / "data" / "perth_house_prices_tratado.csv",
]
CAMINHO_REGISTRO = PASTA_PROJETO / "relatorios" / "registro_tratamento.md"

# Colunas pedidas no tema 5: X = área, quartos, ano de construção, bairro | y = preço
COLUNAS_TEMA = {
    "FLOOR_AREA": "area_habitavel",
    "BEDROOMS": "quartos",
    "BUILD_YEAR": "ano_construcao",
    "SUBURB": "bairro",
    "PRICE": "preco",
}

AREA_MINIMA = 30  # m²: abaixo disso não é uma casa habitável (ex.: 1 m², 13 m²)


# DIAGNÓSTICO

def valores_impossiveis(df):
    """Conta valores fisicamente impossíveis ou suspeitos (aceita nomes originais ou tratados)."""
    col = lambda nome: nome if nome in df.columns else COLUNAS_TEMA[nome]

    regras = {
        f"área habitável < {AREA_MINIMA} m²": df[col("FLOOR_AREA")] < AREA_MINIMA,
        "preço <= 0": df[col("PRICE")] <= 0,
        "quartos <= 0": df[col("BEDROOMS")] <= 0,
        "ano de construção no futuro": df[col("BUILD_YEAR")] > pd.Timestamp.today().year,
    }
    if "LAND_AREA" in df.columns:
        regras["terreno = 999.999 m² (código de 'sem informação')"] = df["LAND_AREA"] == 999999
    if "DATE_SOLD" in df.columns and "BUILD_YEAR" in df.columns:
        ano_venda = pd.to_datetime(df["DATE_SOLD"].str.strip(), format="%m-%Y", errors="coerce").dt.year
        regras["vendida antes do ano de construção (venda na planta)"] = df["BUILD_YEAR"] > ano_venda
    return {nome: int(mascara.sum()) for nome, mascara in regras.items()}


def diagnosticar(df, titulo):
    """Imprime e devolve o diagnóstico: tamanho, tipos, ausentes, duplicatas e impossíveis."""
    diag = {
        "linhas": len(df),
        "colunas": df.shape[1],
        "tipos": df.dtypes.astype(str).to_dict(),
        "ausentes": df.isna().sum()[lambda s: s > 0].to_dict(),
        "duplicatas": int(df.duplicated().sum()),
        "impossiveis": valores_impossiveis(df),
    }

    print(f"\n{'=' * 80}\n{titulo}\n{'=' * 80}")
    print(f"Tamanho: {diag['linhas']} linhas x {diag['colunas']} colunas")
    print("\nTipos das colunas:")
    print(df.dtypes.to_string())
    print("\nValores ausentes:")
    print(pd.Series(diag["ausentes"], dtype=int).to_string() if diag["ausentes"] else "nenhum")
    print(f"\nLinhas duplicadas: {diag['duplicatas']}")
    print("\nValores impossíveis / suspeitos:")
    for regra, qtd in diag["impossiveis"].items():
        print(f"  {regra}: {qtd}")
    return diag


# TRATAMENTO

def tratar(df):
    """Aplica o tratamento e devolve (df_tratado, registro). O df recebido não é alterado."""
    registro = []

    def etapa(descricao, motivo, antes, depois):
        registro.append({"etapa": descricao, "motivo": motivo,
                         "antes": antes, "depois": depois, "removidas": antes - depois})

    df = df.copy()

    # 1. Seleção de colunas
    n = len(df)
    df = df[list(COLUNAS_TEMA)]
    etapa("Mantidas só as 5 colunas do tema (FLOOR_AREA, BEDROOMS, BUILD_YEAR, SUBURB, PRICE)",
          "O tema define X = área, quartos, ano e bairro; y = preço. As outras colunas "
          "(garagem, ranking de escola etc.) não entram e têm muitos ausentes.", n, len(df))

    # 2. Ausentes: só BUILD_YEAR tem ausentes entre as colunas mantidas
    n = len(df)
    ausentes_ano = int(df["BUILD_YEAR"].isna().sum())
    df = df.dropna()
    etapa(f"Excluídas as linhas sem BUILD_YEAR ({ausentes_ano} casas, {ausentes_ano / n:.1%})",
          "Exclusão em vez de preenchimento: preencher com a mediana (1995) daria a casas "
          "antigas e novas a mesma idade, inventando um dado que influencia o preço. "
          "Com mais de 30 mil linhas restantes, a perda é pequena.", n, len(df))

    # 3. Valores impossíveis
    n = len(df)
    df = df[df["FLOOR_AREA"] >= AREA_MINIMA]
    etapa(f"Excluídas as casas com área habitável < {AREA_MINIMA} m²",
          "Valores como 1 m² e 13 m² são erros de digitação: nenhuma casa à venda tem esse tamanho.",
          n, len(df))

    # 4. Conversão de tipo
    n = len(df)
    df["BUILD_YEAR"] = df["BUILD_YEAR"].astype(int)
    etapa("BUILD_YEAR convertido de float para int",
          "Vinha como decimal (ex.: 2003.0) só por causa dos ausentes; ano é número inteiro.",
          n, len(df))

    # 5. Renomear
    df = df.rename(columns=COLUNAS_TEMA)
    etapa("Colunas renomeadas para português (area_habitavel, quartos, ano_construcao, bairro, preco)",
          "São os nomes usados pelo modelo e pelo formulário do Flask.", len(df), len(df))

    # 6. Duplicatas (feito depois de reduzir as colunas: aí surgem linhas idênticas)
    n = len(df)
    df = df.drop_duplicates()
    etapa("Excluídas as linhas duplicadas (mesmos 5 valores)",
          "Linhas idênticas pesariam em dobro nas estatísticas e poderiam cair no treino e "
          "no teste ao mesmo tempo (vazamento).", n, len(df))

    return df.reset_index(drop=True), registro


def salvar_registro(registro, diag_antes, diag_depois):
    linhas = [
        "# Registro do tratamento dos dados",
        "",
        "Gerado automaticamente por `tratamento_dados.py`. O CSV original "
        "(`perth_house_prices.csv`) não é alterado.",
        "",
        f"- **Antes:** {diag_antes['linhas']} linhas x {diag_antes['colunas']} colunas",
        f"- **Depois:** {diag_depois['linhas']} linhas x {diag_depois['colunas']} colunas",
        f"- **Removidas:** {diag_antes['linhas'] - diag_depois['linhas']} linhas "
        f"({(diag_antes['linhas'] - diag_depois['linhas']) / diag_antes['linhas']:.1%})",
        "",
        "## Diagnóstico da base original",
        "",
        "| Coluna | Tipo | Ausentes |",
        "|---|---|---|",
    ]
    for coluna, tipo in diag_antes["tipos"].items():
        linhas.append(f"| {coluna} | {tipo} | {diag_antes['ausentes'].get(coluna, 0)} |")
    linhas += ["", f"Linhas duplicadas (todas as colunas): {diag_antes['duplicatas']}", "",
               "| Valor impossível / suspeito | Linhas |", "|---|---|"]
    linhas += [f"| {regra} | {qtd} |" for regra, qtd in diag_antes["impossiveis"].items()]
    linhas += [
        "",
        "Observações:",
        "",
        "- `LAND_AREA = 999.999` e as vendas na planta não afetam o modelo: terreno e data de venda "
        "não fazem parte do tema, então essas colunas são descartadas e as linhas ficam.",
        "- Vender antes de terminar a obra é comum (venda na planta), por isso não é tratado como erro.",
        "",
        "## Transformações (em ordem)",
        "",
        "| # | Transformação | Motivo | Antes | Depois | Removidas |",
        "|---|---|---|---|---|---|",
    ]
    for i, r in enumerate(registro, 1):
        linhas.append(f"| {i} | {r['etapa']} | {r['motivo']} | {r['antes']} | {r['depois']} | {r['removidas']} |")
    linhas += ["", "## Base tratada", "", "| Coluna | Tipo | Ausentes |", "|---|---|---|"]
    for coluna, tipo in diag_depois["tipos"].items():
        linhas.append(f"| {coluna} | {tipo} | {diag_depois['ausentes'].get(coluna, 0)} |")
    linhas += ["", f"Duplicatas restantes: {diag_depois['duplicatas']}", ""]

    CAMINHO_REGISTRO.parent.mkdir(exist_ok=True)
    CAMINHO_REGISTRO.write_text("\n".join(linhas), encoding="utf-8")


def main():
    df_original = pd.read_csv(CAMINHO_ORIGINAL)

    print("PRIMEIROS 5 REGISTROS DA BASE ORIGINAL:")
    print(df_original.head())

    diag_antes = diagnosticar(df_original, "DIAGNÓSTICO DA BASE ORIGINAL")
    df_tratado, registro = tratar(df_original)
    diag_depois = diagnosticar(df_tratado, "DIAGNÓSTICO DA BASE TRATADA")

    print(f"\n{'=' * 80}\nTRANSFORMAÇÕES APLICADAS\n{'=' * 80}")
    for i, r in enumerate(registro, 1):
        print(f"{i}. {r['etapa']}: {r['antes']} -> {r['depois']} (removidas: {r['removidas']})")
        print(f"   Motivo: {r['motivo']}")

    print(f"\nTAMANHO ANTES: {diag_antes['linhas']} x {diag_antes['colunas']}")
    print(f"TAMANHO DEPOIS: {diag_depois['linhas']} x {diag_depois['colunas']}")

    for caminho in CAMINHOS_TRATADO:
        caminho.parent.mkdir(exist_ok=True)
        df_tratado.to_csv(caminho, index=False)
    salvar_registro(registro, diag_antes, diag_depois)
    print(f"\nBase tratada salva em: {', '.join(str(c.relative_to(PASTA_PROJETO)) for c in CAMINHOS_TRATADO)}")
    print(f"Registro das transformações: {CAMINHO_REGISTRO.relative_to(PASTA_PROJETO)}")


if __name__ == "__main__":
    main()
