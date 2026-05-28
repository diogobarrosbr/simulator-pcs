import streamlit as st
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import squareform

st.set_page_config(page_title="Zoneamento PCS", layout="wide")
st.title("Simulador de Agrupamento de Nós (Upload de Arquivo)")

# 1. Campo para o usuário fazer o upload da planilha
uploaded_file = st.file_uploader("Faça o upload da sua planilha de simulação (CSV ou Excel)", type=["csv", "xlsx"])

if uploaded_file is not None:
    # Lê o arquivo automaticamente identificando o separador (, ou ;)
    if uploaded_file.name.endswith('.csv'):
        df_dados = pd.read_csv(uploaded_file, sep=None, engine='python', decimal=',')
    else:
        df_dados = pd.read_excel(uploaded_file)

    try:
        # Identifica as colunas de gases dinamicamente (Tudo que começar com 'Porcentagem_')
        gas_cols = [col for col in df_dados.columns if col.startswith('Porcentagem_')]

        # Extrai os nós únicos garantindo a ordem alfabética (Nó_01, Nó_02...)
        node_labels = sorted(df_dados['ID_No'].astype(str).unique())
        num_nodes = len(node_labels)

        # Extrai os cenários únicos presentes na base (Sim_01, Sim_02...)
        cenarios_unicos = sorted(df_dados['Cenario'].astype(str).unique())

        # 2. Preparação e Padronização dos Dados por Cenário
        scaler = StandardScaler()
        z_dict = {}

        for cenario in cenarios_unicos:
            # Filtra os dados apenas para o cenário atual
            df_cenario = df_dados[df_dados['Cenario'] == cenario]
            
            # Usa o ID_No como índice e reordena para garantir o alinhamento correto dos nós
            df_cenario = df_cenario.set_index('ID_No').reindex(node_labels)
            
            # Extrai os valores das porcentagens e aplica o Z-score
            s_values = df_cenario[gas_cols].values
            z_dict[cenario] = scaler.fit_transform(s_values)

        # 3. Interface (Barra Lateral Dinâmica)
        st.sidebar.header("Parâmetros do Algoritmo")
        threshold = st.sidebar.slider("Limiar de Corte (Threshold)", 0.1, 10.0, 2.0, 0.1)
        min_nodes = st.sidebar.number_input("Mínimo de Nós por Zona", 1, 15, 1)

        st.sidebar.subheader("Cenários Ativos")
        cenarios_ativos = []
        
        # Cria os checkboxes automaticamente baseado em quantos cenários existirem na planilha
        for cenario in cenarios_unicos:
            ativo = st.sidebar.checkbox(f"Considerar {cenario}", True)
            if ativo:
                cenarios_ativos.append(cenario)

        # Validações de segurança
        if not cenarios_ativos:
            st.warning("Selecione pelo menos um cenário no menu lateral para realizar o cálculo.")
        elif num_nodes < 2:
            st.warning("A planilha precisa ter pelo menos 2 nós preenchidos para rodar o agrupamento.")
        else:
            # 4. Matemática (Chebyshev Multi-Cenário Dinâmico)
            dist_matrix = np.zeros((num_nodes, num_nodes))
            
            for i in range(num_nodes):
                for j in range(i + 1, num_nodes):
                    diffs = []
                    # Calcula a diferença apenas nos cenários que o usuário manteve marcados
                    for cenario in cenarios_ativos:
                        z_atual = z_dict[cenario]
                        diffs.append(np.max(np.abs(z_atual[i] - z_atual[j])))
                    
                    # Salva o pior caso
                    dist_matrix[i, j] = max(diffs)
                    dist_matrix[j, i] = dist_matrix[i, j]

            # Clusterização
            condensed_dist = squareform(dist_matrix)
            Z = linkage(condensed_dist, method='complete')
            clusters = fcluster(Z, threshold, criterion='distance')

            # --- NOVO: CÁLCULO DO SILHOUETTE SCORE ---
            num_zonas_unicas = len(set(clusters))
            
            # A silhueta só pode ser calculada se houver entre 2 e (N-1) clusters
            if 1 < num_zonas_unicas < num_nodes:
                # Usamos metric='precomputed' porque já construímos a matriz de distância perfeita
                score_silhueta = silhouette_score(dist_matrix, clusters, metric='precomputed')
            else:
                score_silhueta = None
            # -----------------------------------------

            # 5. Visualização
            col1, col2 = st.columns([2, 1])

            with col1:
                st.subheader("Dendrograma de Distância")
                fig, ax = plt.subplots(figsize=(10, 6))
                dendrogram(Z, labels=node_labels, color_threshold=threshold, ax=ax)
                ax.axhline(y=threshold, c='r', ls='--', label=f'Corte = {threshold}')
                ax.set_ylabel("Distância Máxima (Chebyshev)")
                plt.xticks(rotation=90, ha='center', fontsize=8) 
                ax.legend()
                st.pyplot(fig)

            with col2:
                st.subheader("Resultado das Zonas")
                df_results = pd.DataFrame({'Nó': node_labels, 'Zona_PCS': clusters})
                zone_counts = df_results['Zona_PCS'].value_counts()
                
                df_results['Status'] = df_results['Zona_PCS'].apply(
                    lambda x: "✅ OK" if zone_counts[x] >= min_nodes else "⚠️ Isolado"
                )
                
                # --- NOVO: EXIBIÇÃO DA MÉTRICA ---
                col_m1, col_m2 = st.columns(2)
                with col_m1:
                    st.metric("Total de Zonas Válidas", len(zone_counts))
                with col_m2:
                    if score_silhueta is not None:
                        # Exibe com 3 casas decimais e usa a cor do Streamlit para indicar positividade
                        st.metric("Silhouette Score", f"{score_silhueta:.3f}", 
                                  help="Varia de -1 a 1. Mais próximo de 1 = Grupos coesos e bem separados.")
                    else:
                        st.metric("Silhouette Score", "N/A", 
                                  help="O score requer pelo menos 2 zonas formadas para ser calculado.")
                # ---------------------------------
                
                st.dataframe(df_results, use_container_width=True, hide_index=True)

    except Exception as e:
        st.error(f"Erro ao processar os dados. Verifique o formato da planilha. Detalhes técnicos: {e}")
else:
    st.info("Aguardando o upload do arquivo base (csv ou xlsx)...")

