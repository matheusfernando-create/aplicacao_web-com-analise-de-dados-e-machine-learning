import io
from datetime import datetime
from pathlib import Path

import pandas as pd

BASE = Path(__file__).parent
CAMINHO_EXEMPLO = BASE / "perth_house_prices_tratado.csv" 

COLUNAS_OBRIGATORIAS = ["area_habitavel", "quartos", "ano_construcao", "bairro", "preco"]
COLUNAS_NUMERICAS = ["area_habitavel", "quartos", "ano_construcao", "preco"]
COLUNAS_BRUTAS = ["floor_area", "bedrooms", "build_year", "suburb", "price"]

AREA_MINIMA = 30            
ANO_MINIMO = 1800
MINIMO_REGISTROS = 50       
REMOVER_DUPLICATAS = True  


def formatar_numero(valor, casas=0, milhar=True):
    """1234.5 -> '1.234,5' (padrão brasileiro)."""
    if valor is None or pd.isna(valor):
        return "-"
    texto = f"{valor:,.{casas}f}"
    if not milhar:
        texto = texto.replace(",", "")
    return texto.replace(",", "§").replace(".", ",").replace("§", ".")


def formatar_moeda(valor):
    """612000 -> 'A$ 612.000' (dólar australiano)."""
    if valor is None or pd.isna(valor):
        return "-"
    return "A$ " + formatar_numero(round(valor), 0)


def _para_numero(serie):
    """Converte para número aceitando vírgula decimal. Texto inválido vira NaN."""
    if not pd.api.types.is_numeric_dtype(serie):  # texto (object no pandas 2, str no pandas 3)
        serie = serie.astype(str).str.strip().str.replace(",", ".", regex=False)
    return pd.to_numeric(serie, errors="coerce")


def ler_csv(conteudo):
    """
    Lê os bytes de um CSV e valida a estrutura.
    Retorna (DataFrame, []) se estiver tudo certo ou (None, [mensagens de erro]).
    """
    if not conteudo or not conteudo.strip():
        return None, ["O arquivo está vazio. Envie um CSV com cabeçalho e pelo menos uma linha de dados."]

    texto = None
    for codificacao in ("utf-8-sig", "latin-1"):
        try:
            texto = conteudo.decode(codificacao)
            break
        except UnicodeDecodeError:
            continue

    primeira_linha = texto.splitlines()[0]
    separador = max([",", ";", "\t"], key=primeira_linha.count)
    if primeira_linha.count(separador) == 0:
        return None, ["Não foi possível identificar o separador de colunas. "
                      "Use vírgula (,) ou ponto e vírgula (;) entre as colunas."]

    try:
        df = pd.read_csv(io.StringIO(texto), sep=separador)
    except (pd.errors.ParserError, pd.errors.EmptyDataError):
        return None, ["O arquivo não pôde ser lido como CSV. "
                      "Confira se todas as linhas têm o mesmo número de colunas."]

    df.columns = [str(c).strip().lower() for c in df.columns]

    faltando = [c for c in COLUNAS_OBRIGATORIAS if c not in df.columns]
    if faltando:
        erros = [f"Faltam colunas obrigatórias: {', '.join(faltando)}. "
                 f"O arquivo precisa ter: {', '.join(COLUNAS_OBRIGATORIAS)}."]
        if all(c in df.columns for c in COLUNAS_BRUTAS):
            erros.append("Este parece ser o arquivo bruto do Kaggle. "
                         "Rode o tratamento_dados.py e envie o CSV tratado.")
        return None, erros

    if df.empty:
        return None, ["O arquivo tem cabeçalho, mas nenhuma linha de dados."]

    erros = []
    for coluna in COLUNAS_NUMERICAS:
        if _para_numero(df[coluna]).notna().sum() == 0:
            erros.append(f"A coluna '{coluna}' deve conter números, mas nenhum valor numérico foi encontrado.")
    if erros:
        return None, erros

    return df, []


def limpar(df):
    """
    Aplica a limpeza e devolve (df_limpo, relatorio).
    O relatório lista cada transformação e quantas linhas ela removeu.
    O CSV original não é alterado (o app.py guarda uma cópia dele em uploads/).
    """
    ano_atual = datetime.now().year
    d = df[COLUNAS_OBRIGATORIAS].copy()

    relatorio = {
        "registros_antes": int(len(d)),
        "colunas_enviadas": [str(c) for c in df.columns],
        "diagnostico": [
            {"coluna": c, "tipo": str(d[c].dtype), "ausentes": int(d[c].isna().sum())}
            for c in COLUNAS_OBRIGATORIAS
        ],
        "duplicatas": int(d.duplicated().sum()),
        "passos": [],
    }

    def registrar(etapa, detalhe, removidas=0):
        relatorio["passos"].append({"etapa": etapa, "detalhe": detalhe, "removidas": int(removidas)})

    registrar("Selecionar colunas",
              "Mantidas apenas as colunas do modelo: " + ", ".join(COLUNAS_OBRIGATORIAS) + ".")

    invalidos = 0
    for coluna in COLUNAS_NUMERICAS:
        ausentes_antes = d[coluna].isna().sum()
        d[coluna] = _para_numero(d[coluna])
        invalidos += int(d[coluna].isna().sum() - ausentes_antes)
    registrar("Converter números",
              f"{invalidos} valores não numéricos foram tratados como ausentes (vírgula decimal é aceita).")

    d["bairro"] = d["bairro"].astype(str).str.strip().replace({"": None, "nan": None})

    def filtrar(etapa, detalhe, mascara_manter):
        nonlocal d
        antes = len(d)
        d = d[mascara_manter(d)]
        registrar(etapa, detalhe, antes - len(d))

    filtrar("Remover ausentes", "Linhas com qualquer coluna do modelo vazia não entram no treino nem na análise.",
            lambda x: x.notna().all(axis=1))
    filtrar("Área impossível", f"Removidos imóveis com área habitável menor que {AREA_MINIMA} m².",
            lambda x: x["area_habitavel"] >= AREA_MINIMA)
    filtrar("Quartos impossíveis", "Removidos imóveis com menos de 1 quarto.",
            lambda x: x["quartos"] >= 1)
    filtrar("Ano impossível", f"Removidos anos de construção fora do intervalo {ANO_MINIMO} a {ano_atual}.",
            lambda x: x["ano_construcao"].between(ANO_MINIMO, ano_atual))
    filtrar("Preço impossível", "Removidos imóveis com preço menor ou igual a zero.",
            lambda x: x["preco"] > 0)

    if REMOVER_DUPLICATAS:
        antes = len(d)
        d = d.drop_duplicates()
        registrar("Duplicatas", "Linhas idênticas removidas (mesma regra do treinar.py).", antes - len(d))
    else:
        registrar("Duplicatas",
                  f"{relatorio['duplicatas']} linhas idênticas foram mantidas: imóveis diferentes "
                  "podem ter a mesma área, quartos, ano e bairro.")

    d["quartos"] = d["quartos"].astype(int)
    d["ano_construcao"] = d["ano_construcao"].astype(int)
    registrar("Ajustar tipos", "quartos e ano_construcao convertidos para inteiro; bairro sem espaços nas pontas.")

    d = d.reset_index(drop=True)
    relatorio["registros_depois"] = int(len(d))
    return d, relatorio


def carregar_base_exemplo():
    """Lê e limpa o CSV de exemplo (também usado pelo treinar.py)."""
    df, erros = ler_csv(CAMINHO_EXEMPLO.read_bytes())
    if erros:
        raise ValueError("; ".join(erros))
    return limpar(df)


def resumo_bairros(df):
    """Mediana de preço por bairro, só com bairros que têm vendas suficientes."""
    agrupado = df.groupby("bairro")["preco"].agg(imoveis="count", mediana="median", media="mean")
    validos = agrupado
    minimo = 1
    for minimo in (30, 5, 1):
        validos = agrupado[agrupado["imoveis"] >= minimo]
        if not validos.empty:
            break
    return validos.sort_values("mediana", ascending=False), minimo


def analisar(df):
    """Estatísticas, achados e interpretações dos gráficos, tudo calculado a partir do df."""
    preco = df["preco"]

    formatos = {
        "area_habitavel": ("Área habitável (m²)", 1, True),
        "quartos": ("Quartos", 1, True),
        "ano_construcao": ("Ano de construção", 0, False),
        "preco": ("Preço", 0, True),
    }
    estatisticas = []
    for coluna, (nome, casas, milhar) in formatos.items():
        s = df[coluna]
        if coluna == "preco":
            f = formatar_moeda
            fm = formatar_moeda
        else:
            f = lambda v, c=casas, m=milhar: formatar_numero(v, c, m)
            fm = lambda v, m=milhar: formatar_numero(v, 0, m)
        estatisticas.append({
            "coluna": nome, "media": f(s.mean()), "mediana": f(s.median()), "desvio": f(s.std()),
            "minimo": fm(s.min()), "maximo": fm(s.max()),
        })

    achados = []
    r = df["area_habitavel"].corr(preco)
    if pd.notna(r):
        forca = "forte" if abs(r) >= 0.7 else "moderada" if abs(r) >= 0.4 else "fraca"
        sentido = "positiva" if r > 0 else "negativa"
        achado = f"A correlação entre área habitável e preço é {forca} e {sentido} (r = {formatar_numero(r, 2)})"
        achado += ": imóveis maiores tendem a custar mais." if r > 0 else "."
        achados.append(achado)

    por_bairro, minimo = resumo_bairros(df)
    if len(por_bairro) >= 2 and por_bairro["mediana"].iloc[-1] > 0:
        caro, barato = por_bairro.index[0], por_bairro.index[-1]
        m_caro, m_barato = por_bairro["mediana"].iloc[0], por_bairro["mediana"].iloc[-1]
        achados.append(
            f"O bairro com maior preço mediano ({caro}, {formatar_moeda(m_caro)}) custa cerca de "
            f"{formatar_numero(m_caro / m_barato, 1)} vezes o de menor preço ({barato}, {formatar_moeda(m_barato)}), "
            f"considerando bairros com pelo menos {minimo} vendas. A localização pesa muito no preço.")

    por_quartos = df.groupby("quartos")["preco"].agg(imoveis="count", mediana="median")
    por_quartos = por_quartos[por_quartos["imoveis"] >= 30]
    if len(por_quartos) >= 2:
        q1, q2 = por_quartos.index[0], por_quartos.index[-1]
        achados.append(
            f"A mediana do preço vai de {formatar_moeda(por_quartos['mediana'].iloc[0])} "
            f"({q1} quarto{'s' if q1 > 1 else ''}) até {formatar_moeda(por_quartos['mediana'].iloc[-1])} "
            f"({q2} quartos), considerando quantidades de quartos com pelo menos 30 imóveis.")

    novos, antigos = df[df["ano_construcao"] >= 2000]["preco"], df[df["ano_construcao"] < 2000]["preco"]
    if len(novos) >= 30 and len(antigos) >= 30 and antigos.median() > 0:
        dif = (novos.median() / antigos.median() - 1) * 100
        comparacao = "maior" if dif >= 0 else "menor"
        achados.append(
            f"Imóveis construídos a partir de 2000 têm preço mediano de {formatar_moeda(novos.median())}, "
            f"{formatar_numero(abs(dif), 0)}% {comparacao} que os anteriores ({formatar_moeda(antigos.median())}).")

    p25, p75, p95 = preco.quantile(0.25), preco.quantile(0.75), preco.quantile(0.95)
    dist = (f"A metade central dos imóveis custa entre {formatar_moeda(p25)} e {formatar_moeda(p75)} "
            f"(mediana {formatar_moeda(preco.median())}). ")
    if preco.mean() > preco.median() * 1.02:
        dist += (f"A média ({formatar_moeda(preco.mean())}) fica acima da mediana: poucos imóveis muito caros "
                 "puxam a média para cima (cauda longa à direita). ")
    else:
        dist += "Média e mediana são próximas, então a distribuição é relativamente simétrica. "
    if preco.max() > 1.5 * p95:
        dist += f"O maior preço ({formatar_moeda(preco.max())}) está bem acima do percentil 95 ({formatar_moeda(p95)}), são valores extremos."

    if len(por_bairro) >= 2:
        comp = (f"Entre os bairros com pelo menos {minimo} vendas, o preço mediano vai de "
                f"{formatar_moeda(por_bairro['mediana'].iloc[-1])} ({por_bairro.index[-1]}) a "
                f"{formatar_moeda(por_bairro['mediana'].iloc[0])} ({por_bairro.index[0]}). "
                f"Há desequilíbrio entre bairros: de {int(por_bairro['imoveis'].min())} a "
                f"{int(por_bairro['imoveis'].max())} imóveis por bairro, e comparações com poucos imóveis são menos confiáveis.")
    else:
        comp = "Há bairros demais com poucas vendas para comparar, envie uma base maior."

    if pd.notna(r):
        rel = (f"Cada ponto é um imóvel. A correlação é r = {formatar_numero(r, 2)}, então a área explica "
               f"cerca de {formatar_numero(r * r * 100, 0)}% da variação do preço (r²). O restante vem de outros "
               "fatores como bairro e ano de construção, e por isso o modelo usa as quatro variáveis juntas.")
    else:
        rel = "Não foi possível calcular a correlação entre área e preço com estes dados."

    return {
        "estatisticas": estatisticas,
        "achados": achados,
        "interpretacoes": {"distribuicao": dist, "comparacao": comp, "relacao": rel},
        "correlacao": None if pd.isna(r) else float(r),
    }
