import streamlit as st
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import squareform

# Configuração da página
st.set_page_config(page_title="Simulador de Zoneamento PCS", layout="wide")
st.title("Simulador de Agrupamento de Nós (Gás)")

# 1. Dados Base (Os 3 Cenários)
s1 = np.array([[98, 1.5, 0.5], [95, 3, 2], [92.5, 5, 2.5], [2, 96, 2], [10, 85, 5], [1, 1, 98], [5, 15, 80], [48, 47, 5], [5, 55, 40], [33, 33, 34]])
s2 = np.array([[90, 8, 2], [85, 10, 5], [80, 15, 5], [1, 97, 2], [5, 90, 5], [1, 1, 98], [2, 10, 88], [30, 65, 5], [2, 60, 38], [20, 50, 30]])
s3 = np.array([[97, 2, 1], [94, 3, 3], [90, 5, 5], [5, 90, 5], [8, 80, 12], [1, 1, 98], [2, 5, 93], [45, 40, 15], [2, 40, 58], [25, 25, 50]])

# Padronização
scaler = StandardScaler()
z1, z2, z3 = scaler.fit_transform(s1), scaler.fit_transform(s2), scaler.fit_transform(s3)

# 2. Controles na Barra Lateral (Interface)
st.sidebar.header("Parâmetros da Simulação")
threshold = st.sidebar.slider("Limiar de Corte (Threshold)", min_value=0.1, max_value=3.5, value=2.0, step=0.1)
min_nodes = st.sidebar.number_input("Mínimo de Nós por Zona", min_value=1, max_value=5, value=1)

st.sidebar.subheader("Cenários Ativos")
use_s1 = st.sidebar.checkbox("Cenário 1: Padrão", value=True)
use_s2 = st.sidebar.checkbox("Cenário 2: Queda de Pressão", value=True)
use_s3 = st.sidebar.checkbox("Cenário 3: Válvula de Anel", value=True)

# Validação: Pelo menos um cenário deve estar ativo
if not (use_s1 or use_s2 or use_s3):
    st.warning("Por favor, selecione pelo menos um cenário operacional para calcular as distâncias.")
else:
    # 3. Lógica de Cálculo Dinâmico
    num_nodes = 10
    dist_matrix = np.zeros((num_nodes, num_nodes))

    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            diffs = []
            if use_s1: diffs.append(np.max(np.abs(z1[i] - z1[j])))
            if use_s2: diffs.append(np.max(np.abs(z2[i] - z2[j])))
            if use_s3: diffs.append(np.max(np.abs(z3[i] - z3[j])))
            
            # Pega o pior caso apenas dos cenários ativos
            dist_matrix[i, j] = max(diffs)
            dist_matrix[j, i] = dist_matrix[i, j]

    # Clusterização
    condensed_dist = squareform(dist_matrix)
    Z = linkage(condensed_dist, method='complete')
    clusters = fcluster(Z, threshold, criterion='distance')

    # 4. Plotagem Visual
    col1, col2 = st.columns([2, 1]) # Divide a tela: Gráfico à esquerda, Tabela à direita

    with col1:
        st.subheader("Dendrograma Multicenário")
        fig, ax = plt.subplots(figsize=(10, 6))
        node_labels = [f'Nó_{i+1:02d}' for i in range(num_nodes)]
        dendrogram(Z, labels=node_labels, color_threshold=threshold, ax=ax)
        ax.axhline(y=threshold, c='r', ls='--', label=f'Threshold = {threshold}')
        ax.set_ylabel("Distância de Chebyshev")
        ax.legend()
        st.pyplot(fig) # Renderiza o gráfico no Streamlit

    with col2:
        st.subheader("Resultado das Zonas")
        # Monta a tabela de resultados
        df_results = pd.DataFrame({'Nó': node_labels, 'Zona_PCS': clusters})
        
        # Verifica a regra de negócio do Mínimo de Nós
        zone_counts = df_results['Zona_PCS'].value_counts()
        df_results['Status'] = df_results['Zona_PCS'].apply(
            lambda x: "✅ OK" if zone_counts[x] >= min_nodes else "⚠️ Isolado"
        )
        
        st.dataframe(df_results, use_container_width=True)
        
        # Métricas rápidas
        st.metric("Total de Zonas Formadas", len(zone_counts))
        st.metric("Nós Isolados (Abaixo do Mínimo)", len(df_results[df_results['Status'] == "⚠️ Isolado"]))