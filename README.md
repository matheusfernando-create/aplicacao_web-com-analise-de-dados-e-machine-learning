# Aplicação web com análise de dados e Machine Learning

Tema 05: preço de imóveis em Perth (Austrália). Pergunta do projeto: **o que determina o preço
de uma casa, usando área habitável, quartos, ano de construção e bairro?**

## Como executar

```bash
pip install -r requirements.txt
python tratamento_dados.py   # diagnóstico + limpeza -> perth_house_prices_tratado.csv
python analise.py            # estatísticas, gráficos e achados
python treinar.py            # treina, compara e salva o modelo (ver MACHINE_LEARNING.md)
```

## 1. Leitura, limpeza e análise com Pandas (`tratamento_dados.py`)

**Diagnóstico** (impresso no terminal, antes e depois do tratamento): tipos das colunas,
valores ausentes, duplicatas e valores impossíveis.

| | Linhas | Colunas |
|---|---|---|
| Base original | 33.656 | 19 |
| Base tratada | 30.463 | 5 |

**Tratamento justificado.** O CSV original não é alterado; o resultado vai para
`perth_house_prices_tratado.csv` e cada etapa fica registrada, com motivo e número de linhas, em
[`relatorios/registro_tratamento.md`](relatorios/registro_tratamento.md):

1. Mantidas só as 5 colunas do tema.
2. **Exclusão** das 3.155 casas sem ano de construção (9,4%). Preencher com a mediana inventaria a
   idade da casa, que influencia o preço.
3. **Exclusão** das 10 casas com área < 30 m² (erros como 1 m² e 13 m²).
4. **Conversão** do ano de construção de decimal para inteiro.
5. Renomeação das colunas para português.
6. **Exclusão** de 28 linhas duplicadas.

## 2. Visualizações e achados (`analise.py`)

Os gráficos ficam em `static/graficos/` e a interpretação completa em
[`relatorios/analise.md`](relatorios/analise.md).

| Tipo | Gráfico |
|---|---|
| Distribuição | Histograma do preço (concentração, dispersão e valores extremos) |
| Comparação | Mediana do preço nos 10 bairros mais caros × 10 mais baratos |
| Relação | Dispersão área habitável × preço |
| Relação / desequilíbrio | Boxplot do preço por número de quartos |
| Relação | Preço e área por década de construção |
| Extra (ML) | Preço real × previsto (equivale à matriz de confusão em regressão) |

**Principais achados:**

1. A localização pesa mais que tudo: a mediana do bairro mais caro é 13 vezes a do mais barato.
2. O preço é assimétrico: média de AU$ 639 mil contra mediana de AU$ 539 mil, por causa de uma
   cauda de casas caras (12% passam de AU$ 1 mi).
3. A área explica mais que os quartos (correlação 0,55 contra 0,26); 2 e 3 quartos custam o mesmo.
4. Casas antigas são menores e mesmo assim mais caras, porque ficam nos bairros centrais.
5. A base é desequilibrada em bairros (1 a 209 vendas) e em quartos (3 e 4 quartos são 87%).

## 3. Machine Learning

Ver [`MACHINE_LEARNING.md`](MACHINE_LEARNING.md).
