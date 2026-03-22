import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import plotly.graph_objects as go


# Pagina inicial

st.set_page_config(page_title="Analytics Educacional - Dashboard", layout="wide")
st.title("📊 Sistema de Análise Educacional")

arquivo = "pede.xlsx"


# Funções (tratativas do NaN)

def limpar_numero(x):
    if pd.isna(x):
        return np.nan
    x = str(x).strip().replace(",", ".")
    try:
        return float(x)
    except:
        return np.nan


# Carrega o arquivo

@st.cache_data
def carregar_dados():
    sheets = pd.read_excel(arquivo, sheet_name=None)
    dfs = [df for df in sheets.values()]
    dataset = pd.concat(dfs, ignore_index=True)
    dataset.columns = dataset.columns.str.strip().str.upper()
    dataset = dataset.loc[:, ~dataset.columns.duplicated()]
    dataset.replace(["", " ", "NA", "NAN"], np.nan, inplace=True)
    return dataset

dataset = carregar_dados()
st.success(f"Total de {len(dataset)} registros carregados")


# Limpeza dos dados

for col in dataset.columns:
    dataset[col] = dataset[col].apply(limpar_numero)


# Indicadores

indicadores = ["IAN","IDA","IEG","IAA","IPS","IPP","IPV"]

def find_col(nome):
    for col in dataset.columns:
        col_limpa = col.strip().upper()
        if col_limpa == nome:
            return col
        if col_limpa.startswith(nome + " "):
            return col
    return None

mapa = {k: find_col(k) for k in indicadores}
#st.write("Mapa detectado:", mapa)

for k, v in mapa.items():
    if v:
        dataset[k] = pd.to_numeric(dataset[v], errors="coerce")
    else:
        dataset[k] = np.nan


# Calcular IDA

def find_any(cols):
    for c in dataset.columns:
        for nome in cols:
            if nome in c:
                return c
    return None

mat = find_any(["MAT"])
por = find_any(["POR"])
ing = find_any(["ING"])

if mat and por and ing:
    dataset["IDA"] = dataset[[mat, por, ing]].mean(axis=1)


# Calcular IPP

avaliadores = [c for c in dataset.columns if "AVALIADOR" in c]
if avaliadores:
    dataset["IPP"] = dataset[avaliadores].mean(axis=1)

for col in indicadores:
    if dataset[col].isna().all():
        dataset[col] = 5
    else:
        dataset[col] = dataset[col].fillna(dataset[col].mean())


# Calcular INDE

dataset["INDE"] = (
    dataset["IAN"] * 0.1 +
    dataset["IDA"] * 0.2 +
    dataset["IEG"] * 0.2 +
    dataset["IAA"] * 0.1 +
    dataset["IPS"] * 0.2 +
    dataset["IPP"] * 0.1 +
    dataset["IPV"] * 0.2
)
dataset["INDE"] = dataset["INDE"].fillna(dataset["INDE"].mean())


# Modelo de Risco

X = dataset[indicadores]
y = dataset["INDE"]
dataset["RISCO"] = y < y.quantile(0.25)
modelo_risco = RandomForestClassifier(n_estimators=200, random_state=42)
modelo_risco.fit(X, dataset["RISCO"])


# Simulador (Sidebar)

st.sidebar.header("Simular indicadores")
inputs = {col: st.sidebar.slider(col, 0.0, 10.0, 5.0) for col in indicadores}

if st.sidebar.button("Prever"):
    entrada = pd.DataFrame([inputs])

    # 🔹 INDE direto da fórmula
    inde = (
        inputs["IAN"] * 0.1 +
        inputs["IDA"] * 0.2 +
        inputs["IEG"] * 0.2 +
        inputs["IAA"] * 0.1 +
        inputs["IPS"] * 0.2 +
        inputs["IPP"] * 0.1 +
        inputs["IPV"] * 0.2
    )

    # 🔹 Risco híbrido
    risco_modelo = modelo_risco.predict_proba(entrada)[0][1]
    risco_regra = 1 - (inde / 10)
    risco = (risco_modelo + risco_regra) / 2
    risco = max(0, min(risco, 1))

    # 🔹 Classificação
    if inde < 5:
        classe = "🔴 Crítico"
    elif inde < 7:
        classe = "🟡 Atenção"
    else:
        classe = "🟢 Alto desempenho"

    # Dashboard - Layout

    col1, col3 = st.columns([3,3])

    # 🔹 Radar de Indicadores / Gauge de Risco abaixo
    with col1:
        # Radar
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=[inputs[col] for col in indicadores],
            theta=indicadores,
            fill='toself',
            name='Aluno'
        ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0,10])),
            showlegend=False,
            title="📊 Indicadores do Aluno"
        )
        st.plotly_chart(fig, use_container_width=True)

        # Gauge de risco
        fig_risco = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risco*100,
            title={'text': "Risco (%)"},
            gauge={'axis': {'range': [0,100]},
                   'bar': {'color': "red" if risco>0.7 else "green" if risco<0.3 else "yellow"}}
        ))
        st.plotly_chart(fig_risco, use_container_width=True)

    # 🔹 Motivo / Recomendações
    with col3:
        st.subheader("❗ Motivo do risco")
        impacto = {k:v for k,v in inputs.items() if v<6}
        if impacto:
            piores = sorted(impacto, key=impacto.get)[:3]
            for p in piores:
                st.write(f"📉 {p} baixo está impactando o resultado")
        else:
            st.write("✅ Nenhum indicador crítico")

        st.subheader("🎯 Recomendações")
        recomendacoes = []
        if inputs["IAN"] < 6:
            recomendacoes.append("📚 Reforço de base (defasagem)")
        if inputs["IDA"] < 6:
            recomendacoes.append("📝 Acompanhamento em Português/Matemática")
        if inputs["IEG"] < 6:
            recomendacoes.append("🎯 Melhorar engajamento")
        if inputs["IPS"] < 6:
            recomendacoes.append("🧠 Apoio psicossocial")
        if inputs["IPP"] < 6:
            recomendacoes.append("✏️ Intervenção pedagógica")
        if inputs["IPV"] < 6:
            recomendacoes.append("📈 Plano de evolução")

        if not recomendacoes:
            st.success("✅ Aluno com bom desempenho")
        else:
            for r in recomendacoes:
                st.write(r)

st.success("Sistema pronto")