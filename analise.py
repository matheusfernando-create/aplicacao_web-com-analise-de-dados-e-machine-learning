# Análise exploratória e visualizações (Tema 05 - Preço de imóveis)
# Execução: python analise.py   (depois de python tratamento_dados.py)
#
# Pergunta do projeto: o que determina o preço de uma casa em Perth, usando
# área habitável, quartos, ano de construção e bairro?
#
# Gera:
#   static/graficos/*.png          -> os gráficos (prontos para o Flask exibir)
#   relatorios/analise.md          -> estatísticas, interpretação de cada gráfico e achados
#
# As funções recebem um DataFrame tratado, então o Flask pode chamá-las com o CSV enviado.

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # gera imagens sem abrir janela (necessário no Flask)
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

PASTA_PROJETO = Path(__file__).parent
CAMINHO_TRATADO = PASTA_PROJETO / "perth_house_prices_tratado.csv"
CAMINHO_PREVISOES = PASTA_PROJETO / "models" / "previsoes_teste.csv"
PASTA_GRAFICOS = PASTA_PROJETO / "static" / "graficos"
CAMINHO_RELATORIO = PASTA_PROJETO / "relatorios" / "analise.md"

MIN_VENDAS_BAIRRO = 20  # bairros com menos vendas têm mediana pouco confiável

# Paleta: azul = série principal, laranja = série de contraste, cinza = referência
AZUL, LARANJA, TEXTO, TEXTO_2, FUNDO = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": FUNDO, "axes.facecolor": FUNDO, "savefig.facecolor": FUNDO,
    "axes.edgecolor": "#c9c8c3", "axes.labelcolor": TEXTO_2, "axes.titlecolor": TEXTO,
    "xtick.color": TEXTO_2, "ytick.color": TEXTO_2, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": "#ebeae6",
    "grid.linewidth": 0.8, "axes.axisbelow": True, "font.size": 11,
    "axes.titlesize": 14, "axes.titleweight": "bold", "axes.titlelocation": "left",
})


def fmt_aud(valor):
    """1234567 -> 'AU$ 1,23 mi' / 539000 -> 'AU$ 539 mil'."""
    if abs(valor) >= 1_000_000:
        return f"AU$ {valor / 1_000_000:.2f} mi".replace(".", ",")
    return f"AU$ {valor / 1_000:.0f} mil"


EIXO_AUD = mticker.FuncFormatter(lambda v, _: fmt_aud(v) if v else "0")


def _salvar(fig, nome):
    PASTA_GRAFICOS.mkdir(parents=True, exist_ok=True)
    caminho = PASTA_GRAFICOS / nome
    fig.savefig(caminho, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return caminho


# ESTATÍSTICAS

def estatisticas(df):
    p = df["preco"]
    q1, q3 = p.quantile([0.25, 0.75])
    limite_outlier = q3 + 1.5 * (q3 - q1)
    por_bairro = df.groupby("bairro")["preco"].agg(vendas="size", mediana="median")
    confiaveis = por_bairro[por_bairro["vendas"] >= MIN_VENDAS_BAIRRO]
    return {
        "resumo": df.describe().round(1),
        "preco": {
            "media": p.mean(), "mediana": p.median(), "desvio": p.std(),
            "q1": q1, "q3": q3, "min": p.min(), "max": p.max(),
            "assimetria": p.skew(), "limite_outlier": limite_outlier,
            "outliers": int((p > limite_outlier).sum()),
        },
        "correlacao": df.corr(numeric_only=True)["preco"].drop("preco").round(2),
        "por_quartos": df.groupby("quartos")["preco"].agg(vendas="size", mediana="median").round(0),
        "por_bairro": por_bairro,
        "bairros_caros": confiaveis.nlargest(10, "mediana"),
        "bairros_baratos": confiaveis.nsmallest(10, "mediana"),
        "bairros_poucas_vendas": int((por_bairro["vendas"] < MIN_VENDAS_BAIRRO).sum()),
        "por_decada": (df.assign(decada=df["ano_construcao"] // 10 * 10)
                       .groupby("decada")
                       .agg(vendas=("preco", "size"), mediana_preco=("preco", "median"),
                            mediana_area=("area_habitavel", "median"))),
    }


# GRÁFICOS

def grafico_distribuicao_preco(df, est):
    """Distribuição: histograma do preço."""
    e = est["preco"]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.hist(df["preco"], bins=60, color=AZUL, edgecolor=FUNDO, linewidth=0.6)
    for valor, nome, estilo in [(e["mediana"], "Mediana", "-"), (e["media"], "Média", "--")]:
        ax.axvline(valor, color=TEXTO, linestyle=estilo, linewidth=1.5)
        ax.text(valor, ax.get_ylim()[1] * 0.97, f" {nome}: {fmt_aud(valor)} ",
                color=TEXTO, fontsize=10, va="top", ha="left" if nome == "Média" else "right",
                bbox={"facecolor": FUNDO, "edgecolor": "none", "pad": 1})
    ax.axvspan(e["limite_outlier"], e["max"], color="#ebeae6", zorder=0)
    ax.text(e["limite_outlier"], ax.get_ylim()[1] * 0.55,
            f"  Valores extremos\n  (> {fmt_aud(e['limite_outlier'])}):\n  {e['outliers']} casas",
            color=TEXTO_2, fontsize=10, va="top")
    ax.xaxis.set_major_formatter(EIXO_AUD)
    ax.set(title="Distribuição do preço de venda das casas em Perth",
           xlabel="Preço de venda", ylabel="Número de casas")
    return _salvar(fig, "1_distribuicao_preco.png")


def grafico_bairros(df, est):
    """Comparação: mediana do preço nos 10 bairros mais caros e nos 10 mais baratos."""
    caros = est["bairros_caros"].sort_values("mediana")
    baratos = est["bairros_baratos"].sort_values("mediana")
    fig, ax = plt.subplots(figsize=(10, 7.5))
    y_bar = np.arange(len(baratos))
    y_car = np.arange(len(caros)) + len(baratos) + 1  # 1 linha de espaço entre os grupos
    ax.barh(y_bar, baratos["mediana"], color=LARANJA, height=0.75, label="10 mais baratos")
    ax.barh(y_car, caros["mediana"], color=AZUL, height=0.75, label="10 mais caros")
    ax.axvline(df["preco"].median(), color=TEXTO, linestyle="--", linewidth=1.2)
    ax.text(df["preco"].median(), -1.1, f" mediana geral: {fmt_aud(df['preco'].median())}",
            color=TEXTO, fontsize=9, va="center")
    ax.set_ylim(-1.6, y_car[-1] + 0.8)
    for y, (bairro, linha) in zip(np.r_[y_bar, y_car], pd.concat([baratos, caros]).iterrows()):
        ax.text(linha["mediana"], y, f" {fmt_aud(linha['mediana'])}  ({int(linha['vendas'])} vendas)",
                va="center", fontsize=9, color=TEXTO_2, zorder=3,
                bbox={"facecolor": FUNDO, "edgecolor": "none", "pad": 0.5})
    ax.set_yticks(np.r_[y_bar, y_car], list(baratos.index) + list(caros.index))
    ax.xaxis.set_major_formatter(EIXO_AUD)
    ax.set_xlim(0, caros["mediana"].max() * 1.35)
    ax.grid(axis="y", visible=False)
    alcas, nomes = ax.get_legend_handles_labels()
    ax.legend(alcas[::-1], nomes[::-1], loc="lower right", frameon=False)
    ax.set(title=f"Mediana do preço por bairro (bairros com {MIN_VENDAS_BAIRRO}+ vendas)",
           xlabel="Mediana do preço de venda", ylabel="Bairro")
    return _salvar(fig, "2_preco_por_bairro.png")


def grafico_area_preco(df, est):
    """Relação: área habitável x preço (hexbin, porque são 30 mil pontos)."""
    fig, ax = plt.subplots(figsize=(10, 6))
    dados = df[df["area_habitavel"] <= df["area_habitavel"].quantile(0.995)]
    hb = ax.hexbin(dados["area_habitavel"], dados["preco"], gridsize=55, cmap="Blues",
                   mincnt=1, bins="log", linewidths=0.2, edgecolors=FUNDO)
    fig.colorbar(hb, ax=ax, label="Número de casas (escala log)")
    faixas = pd.cut(dados["area_habitavel"], bins=np.arange(30, dados["area_habitavel"].max() + 25, 25))
    mediana = dados.groupby(faixas, observed=True)["preco"].median()
    centros = [f.mid for f in mediana.index]
    ax.plot(centros, mediana.values, color=LARANJA, linewidth=2.5, label="Mediana do preço por faixa de 25 m²")
    ax.yaxis.set_major_formatter(EIXO_AUD)
    ax.legend(loc="upper left", frameon=False)
    ax.set(title=f"Área habitável × preço (correlação = {est['correlacao']['area_habitavel']:.2f})".replace(".", ","),
           xlabel="Área habitável (m²)", ylabel="Preço de venda")
    return _salvar(fig, "3_area_x_preco.png")


def grafico_quartos(df, est):
    """Relação / desequilíbrio: boxplot do preço por número de quartos (6+ agrupados)."""
    grupo = df["quartos"].clip(upper=6).map(lambda q: "6+" if q == 6 else str(q))
    ordem = ["1", "2", "3", "4", "5", "6+"]
    caixas = [df.loc[grupo == g, "preco"] for g in ordem]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    bp = ax.boxplot(caixas, tick_labels=ordem, patch_artist=True, showfliers=False, widths=0.55,
                    medianprops={"color": TEXTO, "linewidth": 2},
                    whiskerprops={"color": TEXTO_2}, capprops={"color": TEXTO_2})
    for caixa in bp["boxes"]:
        caixa.set(facecolor=AZUL, edgecolor=AZUL, alpha=0.85)
    topo = max(b.get_ydata().max() for b in bp["caps"])
    for i, c in enumerate(caixas, 1):
        ax.text(i, topo * 1.04, f"n = {len(c):,}".replace(",", "."), ha="center", fontsize=9, color=TEXTO_2)
        ax.text(i + 0.3, c.median(), fmt_aud(c.median()), fontsize=8.5, va="center", color=TEXTO,
                bbox={"facecolor": FUNDO, "edgecolor": "none", "pad": 1})
    ax.set_ylim(0, topo * 1.12)
    ax.yaxis.set_major_formatter(EIXO_AUD)
    ax.grid(axis="x", visible=False)
    ax.set(title="Preço de venda por número de quartos (sem valores extremos)",
           xlabel="Quartos", ylabel="Preço de venda")
    return _salvar(fig, "4_preco_por_quartos.png")


def grafico_decada(df, est):
    """Relação: mediana do preço e da área por década de construção (dois painéis, mesmo eixo x)."""
    d = est["por_decada"]
    d = d[d.index >= 1900]  # antes de 1900 há só 18 casas
    rotulos = [f"{int(x)}s" for x in d.index]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 2], "hspace": 0.12})
    ax1.bar(rotulos, d["mediana_preco"], color=AZUL, width=0.75)
    ax1.yaxis.set_major_formatter(EIXO_AUD)
    ax1.set(title="Casas antigas valem mais, apesar de serem menores", ylabel="Mediana do preço")
    for x, (preco, n) in enumerate(zip(d["mediana_preco"], d["vendas"])):
        ax1.text(x, preco, f"n={n}", ha="center", va="bottom", fontsize=8, color=TEXTO_2)
    ax2.bar(rotulos, d["mediana_area"], color=LARANJA, width=0.75)
    ax2.set(ylabel="Mediana da área (m²)", xlabel="Década de construção")
    for ax in (ax1, ax2):
        ax.grid(axis="x", visible=False)
    return _salvar(fig, "5_preco_por_decada.png")


def grafico_real_previsto(caminho=CAMINHO_PREVISOES, modelo="Random Forest Regressor"):
    """Extra (no lugar da matriz de confusão, que é de classificação): real x previsto no teste."""
    if not caminho.exists():
        return None
    prev = pd.read_csv(caminho)
    real, previsto = prev["preco_real"], prev[f"previsto_{modelo}"]
    fig, ax = plt.subplots(figsize=(7.5, 7))
    ax.scatter(real, previsto, s=9, alpha=0.25, color=AZUL, linewidths=0, label="Casa do teste")
    limite = max(real.max(), previsto.max())
    ax.plot([0, limite], [0, limite], color=TEXTO, linestyle="--", linewidth=1.3, label="Previsão perfeita")
    ax.xaxis.set_major_formatter(EIXO_AUD)
    ax.yaxis.set_major_formatter(EIXO_AUD)
    ax.set_xlim(0, limite)
    ax.set_ylim(0, limite)
    ax.legend(loc="upper left", frameon=False)
    mae = (real - previsto).abs().mean()
    ax.set(title=f"Preço real × previsto ({modelo})\nMAE no teste = {fmt_aud(mae)}",
           xlabel="Preço real", ylabel="Preço previsto")
    return _salvar(fig, "6_real_x_previsto.png")


def gerar_graficos(df):
    est = estatisticas(df)
    caminhos = [
        grafico_distribuicao_preco(df, est),
        grafico_bairros(df, est),
        grafico_area_preco(df, est),
        grafico_quartos(df, est),
        grafico_decada(df, est),
        grafico_real_previsto(),
    ]
    return est, [c for c in caminhos if c]


# RELATÓRIO

def br(valor, casas=0):
    """Formato brasileiro: 30463 -> '30.463' | 0.55 -> '0,55'."""
    return f"{valor:,.{casas}f}".replace(",", "X").replace(".", ",").replace("X", ".")

def tabela_markdown(tabela):
    """DataFrame -> tabela markdown (sem depender do pacote tabulate)."""
    linhas = ["| | " + " | ".join(map(str, tabela.columns)) + " |",
              "|---" * (len(tabela.columns) + 1) + "|"]
    for indice, linha in tabela.iterrows():
        linhas.append(f"| {indice} | " + " | ".join(br(v, 1) for v in linha) + " |")
    return "\n".join(linhas)


def escrever_relatorio(df, est):
    e, c = est["preco"], est["correlacao"]
    caros, baratos = est["bairros_caros"], est["bairros_baratos"]
    pq = est["por_quartos"]["mediana"]
    dec = est["por_decada"]
    antigas = dec[(dec.index >= 1900) & (dec.index < 1950)]
    novas = dec[dec.index >= 2000]
    med_antigas = df.loc[df["ano_construcao"].between(1900, 1949), "preco"].median()
    med_novas = df.loc[df["ano_construcao"] >= 2000, "preco"].median()
    razao_bairros = caros["mediana"].iloc[0] / baratos["mediana"].iloc[0]
    acima_1mi = (df["preco"] > 1_000_000).mean()

    texto = f"""# Análise exploratória: preço das casas em Perth

Gerado automaticamente por `analise.py` a partir de `perth_house_prices_tratado.csv`
({br(len(df))} casas, preços em dólar australiano). O tratamento está em
[`registro_tratamento.md`](registro_tratamento.md).

**Pergunta do projeto:** o que determina o preço de uma casa em Perth, usando área
habitável, quartos, ano de construção e bairro? As respostas abaixo justificam as
variáveis escolhidas para o modelo de Machine Learning.

## Estatísticas descritivas

{tabela_markdown(est['resumo'])}

Correlação de cada variável numérica com o preço: área habitável **{br(c['area_habitavel'], 2)}**,
quartos **{br(c['quartos'], 2)}**, ano de construção **{br(c['ano_construcao'], 2)}**.

## Gráficos e interpretação

### 1. Distribuição do preço (histograma)

![Distribuição do preço](../static/graficos/1_distribuicao_preco.png)

- **Concentração:** metade das casas custa entre {fmt_aud(e['q1'])} e {fmt_aud(e['q3'])}
  (1º e 3º quartis); a mediana é {fmt_aud(e['mediana'])}.
- **Dispersão:** o desvio padrão é alto ({fmt_aud(e['desvio'])}), mais da metade da média.
- **Valores extremos:** a cauda direita é longa (assimetria = {br(e['assimetria'], 2)}).
  {br(e['outliers'])} casas ({br((e['outliers'] / len(df)) * 100, 1)}%) passam de {fmt_aud(e['limite_outlier'])}
  (Q3 + 1,5 × IQR), até {fmt_aud(e['max'])}. Por isso a **média ({fmt_aud(e['media'])}) fica acima
  da mediana**, e a mediana é a medida usada nas comparações. As casas caras são reais (não são
  erros), então foram mantidas.

### 2. Preço por bairro (barras)

![Preço por bairro](../static/graficos/2_preco_por_bairro.png)

- A mediana do bairro mais caro ({caros.index[0]}, {fmt_aud(caros['mediana'].iloc[0])}) é
  **{razao_bairros:.0f} vezes** a do mais barato ({baratos.index[0]}, {fmt_aud(baratos['mediana'].iloc[0])}).
  Nenhuma outra variável produz uma diferença desse tamanho.
- **Desequilíbrio:** a base tem {df['bairro'].nunique()} bairros, com 1 a
  {est['por_bairro']['vendas'].max()} vendas cada. {est['bairros_poucas_vendas']} bairros têm menos
  de {MIN_VENDAS_BAIRRO} vendas e ficaram fora do gráfico, porque a mediana deles é pouco confiável.
  O modelo também erra mais nesses bairros.

### 3. Área habitável × preço (dispersão em hexágonos)

![Área x preço](../static/graficos/3_area_x_preco.png)

- É a variável numérica mais ligada ao preço (correlação {br(c['area_habitavel'], 2)}): a linha da
  mediana sobe de forma quase contínua com a área.
- A nuvem se abre para a direita: com a mesma área, uma casa pode custar de AU$ 300 mil a
  mais de AU$ 2 mi. A área sozinha não basta; essa diferença vem principalmente do **bairro**.
- A maior parte das casas fica entre 100 e 250 m² (hexágonos mais escuros).

### 4. Preço por número de quartos (boxplot)

![Preço por quartos](../static/graficos/4_preco_por_quartos.png)

- Casas de 2 e 3 quartos têm praticamente a mesma mediana ({fmt_aud(pq[2])} e {fmt_aud(pq[3])}).
  O preço só sobe de forma clara a partir de 4 quartos ({fmt_aud(pq[4])}) e 5 quartos ({fmt_aud(pq[5])}).
- Correlação baixa ({br(c['quartos'], 2)}): quartos e área andam juntos (correlação
  {br(df['area_habitavel'].corr(df['quartos']), 2)}), então os quartos acrescentam pouca informação além da área.
- **Desequilíbrio:** 3 e 4 quartos somam {df['quartos'].isin([3, 4]).mean():.0%} das casas;
  1 quarto e 6+ quartos têm poucos exemplos, e as caixas deles são menos confiáveis.

### 5. Preço e área por década de construção (barras)

![Preço por década](../static/graficos/5_preco_por_decada.png)

- Casas construídas entre 1900 e 1949 têm mediana de {fmt_aud(med_antigas)}, contra
  {fmt_aud(med_novas)} das construídas a partir de 2000, **mesmo sendo menores**
  (painel de baixo). Por isso a correlação do ano com o preço é negativa ({br(c['ano_construcao'], 2)}).
- Não é a idade que valoriza a casa: as casas antigas ficam nos bairros centrais e
  tradicionais, e as novas nos loteamentos da periferia. O ano funciona em parte como
  um indicador da localização.

### 6. Extra: preço real × previsto pelo modelo

![Real x previsto](../static/graficos/6_real_x_previsto.png)

O alvo é numérico (regressão), então a matriz de confusão não se aplica. O gráfico
equivalente compara o preço real com o previsto no conjunto de teste: os pontos seguem a
diagonal, mas se espalham mais nas casas caras e o modelo subestima as mais caras de todas,
como mostra a tabela de erro por faixa em `MACHINE_LEARNING.md`.

## Principais achados

1. **A localização é o fator que mais pesa no preço.** A mediana muda {razao_bairros:.0f} vezes
   entre bairros ({caros.index[0]} × {baratos.index[0]}), muito mais do que entre casas pequenas e grandes.
   Isso justifica manter o bairro no modelo, mesmo com {df['bairro'].nunique()} categorias.
2. **O preço é assimétrico, com uma cauda de casas caras.** A média ({fmt_aud(e['media'])}) fica
   {fmt_aud(e['media'] - e['mediana'])} acima da mediana ({fmt_aud(e['mediana'])}) e {acima_1mi:.0%}
   das casas passam de AU$ 1 mi. Essas casas têm poucos exemplos e são onde o modelo erra mais em dinheiro.
3. **Área explica mais que quartos.** A área tem correlação {br(c['area_habitavel'], 2)} com o preço,
   contra {br(c['quartos'], 2)} dos quartos; 2 e 3 quartos custam praticamente o mesmo.
4. **Casas antigas valem mais, mas por causa de onde estão.** As construídas antes de 1950 são
   menores e mesmo assim mais caras, porque ficam nos bairros mais valorizados. É um exemplo de
   correlação que não é causa.
5. **A base é desequilibrada em bairros e em número de quartos**, o que deixa as estimativas
   menos confiáveis para bairros com poucas vendas e para casas com 1 ou 6+ quartos.
"""
    CAMINHO_RELATORIO.parent.mkdir(exist_ok=True)
    CAMINHO_RELATORIO.write_text(texto, encoding="utf-8")
    return CAMINHO_RELATORIO


def main():
    df = pd.read_csv(CAMINHO_TRATADO)
    est, caminhos = gerar_graficos(df)

    e = est["preco"]
    print(f"Casas analisadas: {len(df)}")
    print(f"Preço: média {fmt_aud(e['media'])} | mediana {fmt_aud(e['mediana'])} | "
          f"assimetria {e['assimetria']:.2f} | {e['outliers']} valores extremos")
    print("\nCorrelação com o preço:")
    print(est["correlacao"].to_string())
    print("\nMediana por número de quartos:")
    print(est["por_quartos"].to_string())
    print("\n10 bairros mais caros:")
    print(est["bairros_caros"].to_string())
    print("\n10 bairros mais baratos:")
    print(est["bairros_baratos"].to_string())

    relatorio = escrever_relatorio(df, est)
    print("\nGráficos salvos:")
    for c in caminhos:
        print(f"  {c.relative_to(PASTA_PROJETO)}")
    print(f"Relatório: {relatorio.relative_to(PASTA_PROJETO)}")


if __name__ == "__main__":
    main()
