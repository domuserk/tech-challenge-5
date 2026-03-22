import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.ion()

# =====================================
# CONFIG
# =====================================

arquivo = "./data/pede.xlsx"

indicadores = [
    "IAN","IDA","IEG","IAA","IPS","IPP","IPV","INDE"
]

# =====================================
# FUNÇÕES
# =====================================

def limpar_numero(x):
    if pd.isna(x):
        return np.nan
    x = str(x).strip().replace(",", ".")
    try:
        return float(x)
    except:
        return np.nan

def encontrar_coluna(dataset, palavras):
    for palavra in palavras:
        for col in dataset.columns:
            if palavra in col:
                return col
    return None

# =====================================
# 1. CARREGAR EXCEL
# =====================================

print("\nCarregando arquivo...")
sheets = pd.read_excel(arquivo, sheet_name=None)
dfs = []

for nome, df in sheets.items():
    nome_limpo = nome.strip().upper()

    if "2022" in nome_limpo:
        df["ANO"] = 2022
    elif "2023" in nome_limpo:
        df["ANO"] = 2023
    elif "2024" in nome_limpo:
        df["ANO"] = 2024
    else:
        df["ANO"] = np.nan

    dfs.append(df)

dataset = pd.concat(dfs, ignore_index=True)
dataset.columns = dataset.columns.str.strip().str.upper()
dataset = dataset.loc[:, ~dataset.columns.duplicated()]

print("\nValores únicos de ANO:")
print(dataset["ANO"].unique())

# =====================================
# 2. LIMPEZA
# =====================================

dataset.replace(["", " ", "NA", "NAN"], np.nan, inplace=True)
dataset.drop_duplicates(inplace=True)

# =====================================
# 3. DETECTAR COLUNAS
# =====================================

col_mat = encontrar_coluna(dataset, ["MAT"])
col_port = encontrar_coluna(dataset, ["PORT"])
col_ing = encontrar_coluna(dataset, ["ING"])

col_ieg = encontrar_coluna(dataset, ["IEG"])
col_iaa = encontrar_coluna(dataset, ["IAA"])
col_ips = encontrar_coluna(dataset, ["IPS"])
col_ipv = encontrar_coluna(dataset, ["IPV"])
col_inde = encontrar_coluna(dataset, ["INDE"])

col_defas = encontrar_coluna(dataset, ["DEFAS"])

# =====================================
# 4. IAN (CORRIGIDO)
# =====================================

def calcular_ian(defas):
    if pd.isna(defas):
        return np.nan
    if defas >= 0:
        return 10
    elif defas == -1:
        return 7.5
    elif defas == -2:
        return 5
    else:
        return 2.5

if col_defas:
    dataset[col_defas] = dataset[col_defas].apply(limpar_numero)

    dataset["IAN_CALCULADO"] = dataset[col_defas].apply(calcular_ian)

    if "IAN" in dataset.columns:
        dataset["IAN"] = dataset["IAN"].combine_first(dataset["IAN_CALCULADO"])
    else:
        dataset["IAN"] = dataset["IAN_CALCULADO"]

# =====================================
# 5. IDA
# =====================================

if col_mat and col_port and col_ing:
    dataset["IDA"] = dataset[[col_mat, col_port, col_ing]].applymap(limpar_numero).mean(axis=1)

# =====================================
# 6. OUTROS INDICADORES
# =====================================

if col_ieg:
    dataset["IEG"] = dataset[col_ieg].apply(limpar_numero)
if col_iaa:
    dataset["IAA"] = dataset[col_iaa].apply(limpar_numero)
if col_ips:
    dataset["IPS"] = dataset[col_ips].apply(limpar_numero)
if col_ipv:
    dataset["IPV"] = dataset[col_ipv].apply(limpar_numero)
if col_inde:
    dataset["INDE"] = dataset[col_inde].apply(limpar_numero)

# =====================================
# 7. IPP (AUTO-DETECÇÃO POR CONTEÚDO)
# =====================================

def converter_avaliacao(x):
    if pd.isna(x):
        return np.nan

    x = str(x).strip().upper()

    if "BOLSA" in x:
        return 10
    elif "PROMOVIDO" in x:
        return 8
    elif "MANTIDO" in x:
        return 5
    elif "REQUER" in x:
        return 2
    else:
        return np.nan

# 🔥 detectar colunas com conteúdo relevante (não pelo nome)
cols_av = []

for col in dataset.columns:
    if dataset[col].dtype == object:
        amostra = dataset[col].dropna().astype(str).str.upper()

        if any(amostra.str.contains("PROMOVIDO|MANTIDO|BOLSA|REQUER", na=False)):
            cols_av.append(col)

print("\nColunas de avaliação detectadas:")
for c in cols_av:
    print(c)

print("\nContagem de valores por coluna:")
for col in cols_av:
    print(col, "->", dataset[col].notna().sum())

# converter
cols_num = []

for col in cols_av:
    nova_col = col + "_NUM"
    dataset[nova_col] = dataset[col].apply(converter_avaliacao)
    cols_num.append(nova_col)

# calcular média
dataset["IPP"] = dataset[cols_num].mean(axis=1, skipna=True)

print("\nIPP por ano:")
print(dataset.groupby("ANO")["IPP"].count())

# =====================================
# 8. EVOLUÇÃO TEMPORAL
# =====================================

for col in indicadores:
    if col in dataset.columns:

        print(f"\nContagem de valores válidos para {col}:")
        print(dataset.groupby("ANO")[col].count())

        temp = dataset[["ANO", col]].dropna()

        if len(temp) == 0:
            continue

        medias = temp.groupby("ANO")[col].mean().sort_index()
        medias.index = medias.index.astype(int)

        plt.figure(figsize=(8,5))
        plt.plot(medias.index, medias.values, marker="o")

        plt.xticks(medias.index)

        for x, y in zip(medias.index, medias.values):
            plt.text(x, y, f"{y:.2f}", ha='center')

        plt.title(f"Evolução média do {col} por Ano")
        plt.xlabel("Ano")
        plt.ylabel(col)
        plt.grid()
        plt.tight_layout()
        plt.show()

# =====================================
# 9. CORRELAÇÃO
# =====================================

cols_corr = [c for c in indicadores if c in dataset.columns]
corr = dataset[cols_corr].corr()

print("\nCorrelação:")
print(corr)

plt.figure(figsize=(7,6))
plt.imshow(corr, vmin=-1, vmax=1)
plt.colorbar()
plt.xticks(range(len(cols_corr)), cols_corr, rotation=45)
plt.yticks(range(len(cols_corr)), cols_corr)
plt.title("Mapa de correlação")
plt.tight_layout()
plt.show()

print("\nANÁLISE FINALIZADA")