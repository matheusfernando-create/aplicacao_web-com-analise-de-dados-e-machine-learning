import base64
import json
from pathlib import Path

import analise

ARQUIVO_INDICE = "graficos.json"


def _fmt(valor):
    return analise.fmt_aud(valor)


def _forca(r):
    return "forte" if abs(r) >= 0.7 else "moderada" if abs(r) >= 0.4 else "fraca"


def interpretar(df, est):
    """Texto de cada gráfico, calculado com os números do DataFrame (vale para qualquer CSV enviado)."""
    e, c = est["preco"], est["correlacao"]
    textos = {}

    # 1. Distribuição
    t = f"Metade das casas custa entre {_fmt(e['q1'])} e {_fmt(e['q3'])} (mediana {_fmt(e['mediana'])}). "
    if e["media"] > e["mediana"] * 1.02:
        t += (f"A média ({_fmt(e['media'])}) fica acima da mediana: poucas casas muito caras puxam a média "
              f"para cima (assimetria {analise.br(e['assimetria'], 2)}). ")
    else:
        t += "Média e mediana são próximas, então a distribuição é relativamente simétrica. "
    t += (f"{analise.br(e['outliers'])} casas passam de {_fmt(e['limite_outlier'])} (Q3 + 1,5 × IQR), "
          f"até {_fmt(e['max'])}; a área cinza do gráfico marca essa região de valores extremos.")
    textos["1_distribuicao_preco.png"] = t

    # 2. Bairros
    caros, baratos = est["bairros_caros"], est["bairros_baratos"]
    if len(caros) and len(baratos) and baratos["mediana"].iloc[0] > 0:
        razao = caros["mediana"].iloc[0] / baratos["mediana"].iloc[0]
        vendas = est["por_bairro"]["vendas"]
        textos["2_preco_por_bairro.png"] = (
            f"A mediana do bairro mais caro ({caros.index[0]}, {_fmt(caros['mediana'].iloc[0])}) é "
            f"{razao:.0f} vezes a do mais barato ({baratos.index[0]}, {_fmt(baratos['mediana'].iloc[0])}). "
            f"O gráfico mostra só bairros com {analise.MIN_VENDAS_BAIRRO}+ vendas: {est['bairros_poucas_vendas']} "
            f"dos {df['bairro'].nunique()} bairros ficaram fora porque a mediana deles é pouco confiável. "
            f"Há desequilíbrio entre bairros, de {int(vendas.min())} a {int(vendas.max())} vendas cada.")
    else:
        textos["2_preco_por_bairro.png"] = "Nenhum bairro tem vendas suficientes para comparar."

    # 3. Área x preço
    r = c.get("area_habitavel")
    if r is not None and r == r:
        textos["3_area_x_preco.png"] = (
            f"A correlação entre área e preço é {_forca(r)} (r = {analise.br(r, 2)}). A linha laranja mostra a "
            f"mediana do preço por faixa de 25 m², e os hexágonos mais escuros concentram mais casas. A nuvem se "
            f"abre para a direita: com a mesma área os preços variam muito, e a área sozinha explica cerca de "
            f"{analise.br(r * r * 100)}% da variação do preço (r²).")
    else:
        textos["3_area_x_preco.png"] = "Não foi possível calcular a correlação entre área e preço com estes dados."

    # 4. Quartos
    pq = est["por_quartos"]
    grandes = pq[pq["vendas"] >= 30]
    t = ""
    if len(grandes) >= 2:
        q1, q2 = int(grandes.index[0]), int(grandes.index[-1])
        t += (f"A mediana vai de {_fmt(grandes['mediana'].iloc[0])} ({q1} quarto{'s' if q1 > 1 else ''}) a "
              f"{_fmt(grandes['mediana'].iloc[-1])} ({q2} quartos). ")
    t += (f"A correlação do número de quartos com o preço é {analise.br(c.get('quartos', float('nan')), 2)}, e quartos e "
          f"área andam juntos (correlação {analise.br(df['area_habitavel'].corr(df['quartos']), 2)}). "
          f"Casas de 3 e 4 quartos somam {analise.br(df['quartos'].isin([3, 4]).mean() * 100)}% da base; "
          "os grupos com poucas casas (n acima das caixas) são menos confiáveis.")
    textos["4_preco_por_quartos.png"] = t

    # 5. Década
    antigas = df.loc[df["ano_construcao"].between(1900, 1949), "preco"]
    novas = df.loc[df["ano_construcao"] >= 2000, "preco"]
    t = f"A correlação do ano de construção com o preço é {analise.br(c.get('ano_construcao', float('nan')), 2)}. "
    if len(antigas) >= 30 and len(novas) >= 30:
        t += (f"Casas construídas de 1900 a 1949 têm mediana de {_fmt(antigas.median())}, contra "
              f"{_fmt(novas.median())} das construídas a partir de 2000. ")
    t += ("Correlação não é causa: o ano pode refletir onde as casas ficam (o bairro), então ele funciona em parte "
          "como indicador de localização. Décadas com poucas casas (n acima das barras) são menos confiáveis.")
    textos["5_preco_por_decada.png"] = t

    # 6. Real x previsto
    textos["6_real_x_previsto.png"] = (
        "Compara o preço real com o preço previsto pelo modelo no conjunto de teste: quanto mais perto da linha "
        "tracejada, melhor a previsão. Este gráfico vem do treino do modelo e não muda com o arquivo enviado.")
    return textos


def gerar(df, pasta_saida):
    """Gera os gráficos do analise.py para o df e salva PNGs + graficos.json em pasta_saida."""
    pasta = Path(pasta_saida)
    pasta.mkdir(parents=True, exist_ok=True)
    for antigo in pasta.glob("*.png"):
        antigo.unlink()
    analise.PASTA_GRAFICOS = pasta  # o analise.py salva aqui, e não em static/graficos/

    est = analise.estatisticas(df)
    textos = interpretar(df, est)
    chamadas = [
        ("Distribuição dos preços", "1_distribuicao_preco.png", lambda: analise.grafico_distribuicao_preco(df, est)),
        ("Comparação entre bairros", "2_preco_por_bairro.png", lambda: analise.grafico_bairros(df, est)),
        ("Relação entre área e preço", "3_area_x_preco.png", lambda: analise.grafico_area_preco(df, est)),
        ("Preço por número de quartos", "4_preco_por_quartos.png", lambda: analise.grafico_quartos(df, est)),
        ("Preço por década de construção", "5_preco_por_decada.png", lambda: analise.grafico_decada(df, est)),
        ("Preço real × previsto (modelo)", "6_real_x_previsto.png", lambda: analise.grafico_real_previsto()),
    ]
    itens = []
    for titulo, arquivo, desenhar in chamadas:
        try:
            gerado = desenhar()
        except Exception:  
            gerado = None
        if gerado is None and arquivo.startswith("6_"):
            continue 
        itens.append({"titulo": titulo, "arquivo": arquivo if gerado else None,
                      "texto": textos[arquivo] if gerado else
                      "Este gráfico não pôde ser gerado com os dados enviados (poucos dados nesta categoria)."})
    (pasta / ARQUIVO_INDICE).write_text(json.dumps(itens, ensure_ascii=False), encoding="utf-8")
    return itens


def carregar(pasta):
    """Lê graficos.json e devolve os itens com a imagem em base64. Lista vazia se ainda não foram gerados."""
    pasta = Path(pasta)
    indice = pasta / ARQUIVO_INDICE
    if not indice.exists():
        return []
    itens = json.loads(indice.read_text(encoding="utf-8"))
    for item in itens:
        arquivo = pasta / item["arquivo"] if item["arquivo"] else None
        item["base64"] = base64.b64encode(arquivo.read_bytes()).decode("ascii") if arquivo and arquivo.exists() else None
    return itens