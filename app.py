import streamlit as st
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import squareform

st.set_page_config(page_title="Zoneamento PCS", layout="wide")
st.title("Simulador de Agrupamento de Nós (Google Sheets)")

# 1. COLE AQUI O SEU LINK PUBLICADO DO GOOGLE SHEETS (CSV)
SHEET_CSV_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vRYLscAsrJp-neajDmauJxeYvjXsndk-CTR60ba3soZ1Oe2ue-yXn4nV_OgZ1bSgWjxvJLYwT4_di-U/pub?output=csv"

@st.cache_data(ttl=30) # Atualiza de 30 em 30 segundos se houver mudança
def carregar_dados(url):
    try:
        # Lê o CSV. Tenta identificar automaticamente se o separador é vírgula ou ponto-e-vírgula
        df = pd.read_csv(url, sep=None, engine='python', decimal=',')
        return df
    except Exception as e:
        st.error(f"Erro ao carregar a planilha. Verifique o link publicado. Detalhe: {e}")
        st.stop()

df_dados = carregar_dados(SHEET_CSV_URL)

node_labels = df_dados['Nó'].astype(str).tolist()
num_nodes = len(node_labels)

# Extração dos cenários
s1 = df_dados[['S1_CG1', 'S1_CG2', 'S1_CG3']].values
s2 = df_dados[['S2_CG1', 'S2_CG2', 'S2_CG3']].values
s3 = df_dados[['S3_CG1', 'S3_CG2', 'S3_CG3']].values

scaler = StandardScaler()
z1 = scaler.fit_transform(s1) if len(s1) > 0 else []
z2 = scaler.fit_transform(s2) if len(s2) > 0 else []
z3 = scaler.fit_transform(s3) if len(s3) > 0 else []

# 2. Interface (Barra Lateral)
st.sidebar.header("Parâmetros do Algoritmo")
threshold = st.sidebar.slider("Limiar de Corte (Threshold)", 0.1, 5.0, 2.0, 0.1)
min_nodes = st.sidebar.number_input("Mínimo de Nós por Zona", 1, 10, 1)

st.sidebar.subheader("Cenários Ativos")
use_s1 = st.sidebar.checkbox("Cenário 1: Padrão", True)
use_s2 = st.sidebar.checkbox("Cenário 2: Queda de Pressão", True)
use_s3 = st.sidebar.checkbox("Cenário 3: Válvula de Anel", True)

if st.sidebar.button("🔄 Forçar Atualização da Planilha"):
    st.cache_data.clear()

if not (use_s1 or use_s2 or use_s3):
    st.warning("Selecione pelo menos um cenário para realizar o cálculo.")
elif num_nodes < 2:
    st.warning("Adicione pelo menos 2 nós na planilha para rodar o agrupamento.")
else:
    # 3. Matemática (Chebyshev)
    dist_matrix = np.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            diffs = []
            if use_s1: diffs.append(np.max(np.abs(z1[i] - z1[j])))
            if use_s2: diffs.append(np.max(np.abs(z2[i] - z2[j])))
            if use_s3: diffs.append(np.max(np.abs(z3[i] - z3[j])))
            
            dist_matrix[i, j] = max(diffs)
            dist_matrix[j, i] = dist_matrix[i, j]

    condensed_dist = squareform(dist_matrix)
    Z = linkage(condensed_dist, method='complete')
    clusters = fcluster(Z, threshold, criterion='distance')

    # 4. Visualização
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Dendrograma Multicenário")
        fig, ax = plt.subplots(figsize=(10, 6))
        dendrogram(Z, labels=node_labels, color_threshold=threshold, ax=ax)
        ax.axhline(y=threshold, c='r', ls='--', label=f'Corte = {threshold}')
        ax.set_ylabel("Distância Máxima")
        plt.xticks(rotation=45, ha='right')
        ax.legend()
        st.pyplot(fig)

    with col2:
        st.subheader("Resultado das Zonas")
        df_results = pd.DataFrame({'Nó': node_labels, 'Zona_PCS': clusters})
        zone_counts = df_results['Zona_PCS'].value_counts()
        df_results['Status'] = df_results['Zona_PCS'].apply(
            lambda x: "✅" if zone_counts[x] >= min_nodes else "⚠️ Isolado"
        )
        
        st.metric("Total de Zonas Válidas", len(zone_counts))
        st.dataframe(df_results, use_container_width=True, hide_index=True)