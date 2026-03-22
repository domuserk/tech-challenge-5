import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# ==========================
# 1. CARREGAR DADOS
# ==========================

arquivo = "pede.xlsx"

df = pd.read_excel(arquivo)

df.columns = (
    df.columns
    .str.strip()
    .str.upper()
    .str.replace(" ", "_")
)

# ==========================
# 2. LIMPAR NÚMEROS
# ==========================

def limpar_numero(col):

    return (
        col.astype(str)
        .str.replace(",", ".", regex=False)
        .str.extract(r'([-+]?\d*\.?\d+)')[0]
        .astype(float)
    )

numericas = [
    "INDE_22","CG","CF","CT","IAA","IEG","IPS","IDA",
    "MATEM","PORTUG","INGLÊS","IPV","IAN","DEFAS"
]

for c in numericas:
    if c in df.columns:
        df[c] = limpar_numero(df[c])

# ==========================
# 3. CALCULAR IDA
# ==========================

df["IDA_CALC"] = (
    df[["MATEM","PORTUG","INGLÊS"]]
    .mean(axis=1)
)

# ==========================
# 4. CALCULAR IAN
# ==========================

def extrair_fase(f):

    try:
        return int(str(f).split()[1])
    except:
        return np.nan

df["FASE_NUM"] = limpar_numero(df["FASE"])

df["FASE_IDEAL_NUM"] = df["FASE_IDEAL"].apply(extrair_fase)

df["IAN_CALC"] = df["FASE_NUM"] - df["FASE_IDEAL_NUM"]

# ==========================
# 5. CALCULAR IPV (aprox)
# ==========================

df["IPV_CALC"] = (
    df["INDE_22"] * 0.6 +
    df["IEG"] * 0.4
)

# ==========================
# 6. CRIAR VARIÁVEL DE RISCO
# ==========================

df["RISCO_DEFASAGEM"] = df["DEFAS"].apply(
    lambda x: 1 if pd.notna(x) and x < 0 else 0
)

# ==========================
# 7. FEATURES DO MODELO
# ==========================

features = [
    "IAN_CALC",
    "IDA_CALC",
    "IEG",
    "IAA",
    "IPS",
    "IPV_CALC"
]

X = df[features]
y = df["RISCO_DEFASAGEM"]

# ==========================
# 8. DIVISÃO TREINO TESTE
# ==========================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42
)

# ==========================
# 9. PIPELINE
# ==========================

pipeline = Pipeline([

    ("imputer", SimpleImputer(strategy="mean")),

    ("model", RandomForestClassifier(
        n_estimators=300,
        random_state=42
    ))
])

pipeline.fit(X_train, y_train)

# ==========================
# 10. AVALIAÇÃO
# ==========================

pred = pipeline.predict(X_test)

print(classification_report(y_test, pred))

# ==========================
# 11. SALVAR MODELO
# ==========================

joblib.dump(pipeline, "modelo_risco_defasagem.pkl")

print("Modelo salvo.")