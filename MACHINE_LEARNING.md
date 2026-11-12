# Machine Learning: previsão do preço de imóveis

Parte do Tema 05 (Preço de imóveis). É um problema de **regressão**, porque o alvo é um valor numérico.

## Arquivos

| Arquivo | Função |
|---|---|
| `treinar.py` | Separa X e y, divide treino/teste, treina e compara os modelos, escolhe o final e salva em `models/` |
| `modelo.py` | Usado pelo Flask: carrega o modelo, valida a entrada do formulário e faz a previsão |
| `models/modelo.joblib` | Pipeline salvo (pré-processamento + modelo escolhido) |
| `models/metricas.json` | Resultados de todos os modelos, modelo escolhido, bairros e faixas válidas |
| `models/previsoes_teste.csv` | Preço real × previsto de cada modelo no conjunto de teste (para os gráficos) |

## Como executar

```bash
pip install pandas scikit-learn joblib
python tratamento_dados.py   # gera perth_house_prices_tratado.csv
python treinar.py            # treina, compara e salva o modelo
python modelo.py             # teste rápido de previsão
```

## Base usada

- **Perth House Prices** (Perth, Austrália), vendas de 1988 a 2020 (92% entre 2013 e 2020). Os preços estão em **dólar australiano (AUD)**.
- Arquivo de entrada: `perth_house_prices_tratado.csv`, gerado por `tratamento_dados.py`.
- As **28 linhas duplicadas** são removidas em `tratamento_dados.py` (e `treinar.py` confere de novo antes da divisão). Assim a mesma casa não aparece no treino e no teste ao mesmo tempo.
- **Registros usados: 30.463.**

## X e y

| Variável | Tipo | Por que ajuda a prever o preço |
|---|---|---|
| `area_habitavel` (m²) | numérica | Casas maiores costumam valer mais (correlação de 0,56 com o preço, a mais forte) |
| `quartos` | numérica | Indica o tamanho e o tipo do imóvel |
| `ano_construcao` | numérica | Idade e padrão de construção |
| `bairro` | categórica | A localização pesa muito no preço: o mesmo imóvel vale bem diferente em bairros diferentes |
| **`preco` (y)** | numérica | Preço de venda: é o que o modelo aprende a prever |

O preço **não entra em X**. Endereço, data de venda e outras informações de depois da venda também ficam de fora.

## Treino e teste

- `train_test_split(test_size=0.20, random_state=42)` divide a base em **24.370 casas para treino (80%)** e **6.093 para teste (20%)**.
- Todos os modelos usam **a mesma divisão**, então os resultados são comparáveis.
- O pré-processamento fica dentro de um `Pipeline`. Com isso, o `fit` (mediana dos ausentes e codificação do bairro) usa **somente o treino**, e o mesmo preparo é aplicado no teste e no Flask. Isso evita vazamento de dados.

### Codificação do bairro

O modelo só trabalha com números, então o bairro (são 319) precisa ser convertido:

- **Regressão Linear: One-Hot.** Cada bairro vira uma coluna 0/1, e o modelo aprende um valor para cada bairro.
- **Árvore e Random Forest: TargetEncoder.** O bairro vira o preço médio dele **no treino**, calculado com validação cruzada interna. Com centenas de colunas 0/1, a árvore quase nunca dividia por bairro. Na validação cruzada do treino, o MAE da árvore caiu de cerca de 182 mil (one-hot) para cerca de 114 mil (preço médio).

### Escolha dos parâmetros

O `min_samples_leaf` (número mínimo de casas em cada folha) foi escolhido com **validação cruzada só no treino**. Os valores testados foram 10, 30 e 60 na árvore e 1, 5 e 10 no Random Forest. O conjunto de teste não foi usado nessa escolha.

## Métrica: MAE

**MAE (erro absoluto médio)** é a média de |preço real − preço previsto|, na mesma unidade do preço (AUD). É fácil de explicar: um MAE de 100 mil significa que, em média, o modelo erra o preço em AU$ 100 mil, para mais ou para menos. O RMSE (que pune mais os erros grandes) e o R² aparecem como métricas complementares.

## Resultados

| Modelo | MAE treino | MAE validação cruzada | **MAE teste** | RMSE teste | R² teste |
|---|---|---|---|---|---|
| Baseline (sempre prevê a mediana) | 244.821 | 244.844 | 244.348 | 369.352 | −0,08 |
| Linear Regression | 108.905 | 110.937 | 111.567 | 179.039 | 0,75 |
| Decision Tree Regressor | 107.055 | 113.903 | 114.008 | 180.895 | 0,74 |
| **Random Forest Regressor** | 95.667 | 108.275 | **109.435** | 174.440 | 0,76 |

### Escolha final: Random Forest Regressor

- Tem o **menor MAE** tanto na validação cruzada quanto no teste, e também o menor RMSE.
- Erra **55% menos** que o baseline. O MAE de AU$ 109 mil corresponde a cerca de 17% do preço médio.
- A vantagem sobre a Regressão Linear é pequena (cerca de AU$ 2 mil). A regressão linear é mais simples e quase tão boa, e seria uma alternativa aceitável.
- É o modelo com a **maior diferença entre treino e teste** (95 mil contra 109 mil), então decorou um pouco o treino. Mesmo assim, é o melhor em dados que nunca viu.

### Onde o modelo erra mais

| Faixa de preço real | Preço médio | MAE | Erro relativo |
|---|---|---|---|
| 25% mais baratas | 318.573 | 92.906 | 29% |
| 25% a 50% | 475.264 | 60.345 | 13% |
| 50% a 75% | 641.620 | 85.627 | 13% |
| 25% mais caras | 1.130.162 | 199.223 | 18% |

- Nas **casas mais caras**, o erro em dinheiro é o maior. Há poucos exemplos dessas casas e o preço depende de coisas que não estão em X (terreno, vista, acabamento).
- Nas **casas mais baratas**, o erro relativo é o maior (29%). O modelo tende a puxar o preço para perto da média.
- As casas de preço médio são onde o modelo funciona melhor.

## Limitações

- O modelo usa só 4 variáveis. Tamanho do terreno, banheiros, garagem e distância do centro ficaram de fora, conforme o tema.
- Mistura vendas de 1988 a 2020 (92% entre 2013 e 2020) sem considerar a data, e os preços mudam com o tempo.
- O modelo serve apenas para Perth e para bairros que existem na base. O formulário só aceita bairros e faixas de valores vistos no treino.
- Bairros com poucas vendas têm estimativas menos confiáveis.
- O modelo é **fixo**: o CSV enviado na aplicação atualiza as análises e os gráficos, mas a previsão usa o modelo treinado com a base indicada em `models/metricas.json`.

## Uso no Flask

```python
from modelo import validar_entrada, prever, listar_bairros

# GET: montar o <select> com listar_bairros()
# POST:
valores, erros = validar_entrada(request.form)   # nomes dos campos: area_habitavel, quartos, ano_construcao, bairro
if erros:
    ...  # exibir as mensagens
else:
    preco = prever(valores)                       # float, em AUD
```

`carregar_metricas()` devolve o conteúdo de `metricas.json`, para montar a página de comparação dos modelos.
