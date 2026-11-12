# Registro do tratamento dos dados

Gerado automaticamente por `tratamento_dados.py`. O CSV original (`perth_house_prices.csv`) não é alterado.

- **Antes:** 33656 linhas x 19 colunas
- **Depois:** 30463 linhas x 5 colunas
- **Removidas:** 3193 linhas (9.5%)

## Diagnóstico da base original

| Coluna | Tipo | Ausentes |
|---|---|---|
| ADDRESS | str | 0 |
| SUBURB | str | 0 |
| PRICE | int64 | 0 |
| BEDROOMS | int64 | 0 |
| BATHROOMS | int64 | 0 |
| GARAGE | float64 | 2478 |
| LAND_AREA | int64 | 0 |
| FLOOR_AREA | int64 | 0 |
| BUILD_YEAR | float64 | 3155 |
| CBD_DIST | int64 | 0 |
| NEAREST_STN | str | 0 |
| NEAREST_STN_DIST | int64 | 0 |
| DATE_SOLD | str | 0 |
| POSTCODE | int64 | 0 |
| LATITUDE | float64 | 0 |
| LONGITUDE | float64 | 0 |
| NEAREST_SCH | str | 0 |
| NEAREST_SCH_DIST | float64 | 0 |
| NEAREST_SCH_RANK | float64 | 10952 |

Linhas duplicadas (todas as colunas): 0

| Valor impossível / suspeito | Linhas |
|---|---|
| área habitável < 30 m² | 15 |
| preço <= 0 | 0 |
| quartos <= 0 | 0 |
| ano de construção no futuro | 0 |
| terreno = 999.999 m² (código de 'sem informação') | 3 |
| vendida antes do ano de construção (venda na planta) | 563 |

Observações:

- `LAND_AREA = 999.999` e as vendas na planta não afetam o modelo: terreno e data de venda não fazem parte do tema, então essas colunas são descartadas e as linhas ficam.
- Vender antes de terminar a obra é comum (venda na planta), por isso não é tratado como erro.

## Transformações (em ordem)

| # | Transformação | Motivo | Antes | Depois | Removidas |
|---|---|---|---|---|---|
| 1 | Mantidas só as 5 colunas do tema (FLOOR_AREA, BEDROOMS, BUILD_YEAR, SUBURB, PRICE) | O tema define X = área, quartos, ano e bairro; y = preço. As outras colunas (garagem, ranking de escola etc.) não entram e têm muitos ausentes. | 33656 | 33656 | 0 |
| 2 | Excluídas as linhas sem BUILD_YEAR (3155 casas, 9.4%) | Exclusão em vez de preenchimento: preencher com a mediana (1995) daria a casas antigas e novas a mesma idade, inventando um dado que influencia o preço. Com mais de 30 mil linhas restantes, a perda é pequena. | 33656 | 30501 | 3155 |
| 3 | Excluídas as casas com área habitável < 30 m² | Valores como 1 m² e 13 m² são erros de digitação: nenhuma casa à venda tem esse tamanho. | 30501 | 30491 | 10 |
| 4 | BUILD_YEAR convertido de float para int | Vinha como decimal (ex.: 2003.0) só por causa dos ausentes; ano é número inteiro. | 30491 | 30491 | 0 |
| 5 | Colunas renomeadas para português (area_habitavel, quartos, ano_construcao, bairro, preco) | São os nomes usados pelo modelo e pelo formulário do Flask. | 30491 | 30491 | 0 |
| 6 | Excluídas as linhas duplicadas (mesmos 5 valores) | Linhas idênticas pesariam em dobro nas estatísticas e poderiam cair no treino e no teste ao mesmo tempo (vazamento). | 30491 | 30463 | 28 |

## Base tratada

| Coluna | Tipo | Ausentes |
|---|---|---|
| area_habitavel | int64 | 0 |
| quartos | int64 | 0 |
| ano_construcao | int64 | 0 |
| bairro | str | 0 |
| preco | int64 | 0 |

Duplicatas restantes: 0
