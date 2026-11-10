import pandas as pd

# ANÁLISE INICIAL

# Carrega o arquivo CSV em um DataFrame
df_parth_house_prices = pd.read_csv("perth_house_prices.csv")

# Estrutura do DataFrame
print("INFORMAÇÕES GERAIS DO DATAFRAME SEM O TRATAMENTO:")
df_parth_house_prices.info()
print()

# Mostra as 5 primeiras linhas para ver como os dados são
print("PRIMEIROS 5 REGISTROS:")
print(df_parth_house_prices.head())
print()

# Mostra as 5 últimas linhas para conferir o final do arquivo
print("ÚLTIMOS 5 REGISTROS:")
print(df_parth_house_prices.tail())
print()

# Conta quantos valores ausentes (NaN) existem em cada coluna
print("VALORES AUSENTES POR COLUNA:")
print(df_parth_house_prices.isnull().sum())
print()

print(80 * "=")
print()

# TRATAMENTO DOS DADOS

# Mantém somente as colunas pedidas no tema 5
# x = área habitável, quartos, ano de construção e bairro | y = preço
df_parth_house_prices = df_parth_house_prices[["FLOOR_AREA", "BEDROOMS", "BUILD_YEAR", "SUBURB", "PRICE"]]

# Valores ausentes só nas colunas mantidas
print("VALORES AUSENTES NAS COLUNAS MANTIDAS:")
print(df_parth_house_prices.isnull().sum())
print()

# Remove as linhas com valores ausentes
df_parth_house_prices = df_parth_house_prices.dropna()

# Remove áreas impossíveis (menores que 30 m²)
df_parth_house_prices = df_parth_house_prices[df_parth_house_prices["FLOOR_AREA"] >= 30]

# BUILD_YEAR vem como decimal, converte para inteiro
df_parth_house_prices["BUILD_YEAR"] = df_parth_house_prices["BUILD_YEAR"].astype(int)

# Renomeia as colunas
df_parth_house_prices = df_parth_house_prices.rename(columns={
    "FLOOR_AREA": "area_habitavel",
    "BEDROOMS": "quartos",
    "BUILD_YEAR": "ano_construcao",
    "SUBURB": "bairro",
    "PRICE": "preco"
})

# Confere o resultado
print("INFORMAÇÕES GERAIS DO DATAFRAME TRATADO:")
df_parth_house_prices.info()
print()

print("VALORES AUSENTES POR COLUNA (APÓS A LIMPEZA):")
print(df_parth_house_prices.isnull().sum())
print()

print("PRIMEIROS 5 REGISTROS do DATAFRAME TRATADO:")
print(df_parth_house_prices.head())
print()

print("ÚLTIMOS 5 REGISTROS do DATAFRAME TRATADO:")
print(df_parth_house_prices.tail())

# Salva os dados tratados
df_parth_house_prices.to_csv("perth_house_prices_tratado.csv", index=False)