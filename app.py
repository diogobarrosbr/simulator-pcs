import streamlit as st
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import squareform
from sklearn.metrics import silhouette_score
import plotly.express as px # <-- Adicione esta linha no topo
import plotly.graph_objects as go # Importação do motor de baixo nível
from scipy.spatial import ConvexHull # Importação da ferramenta geográfica
import numpy as np # Necessário para manipulação de matrizes


st.set_page_config(page_title="Zoneamento PCS", layout="wide")
st.title("Simulador de Agrupamento de Nós")

# 1. Campo para o usuário fazer o upload de MÚLTIPLOS arquivos
uploaded_files = st.file_uploader(
    "Faça o upload dos arquivos de simulação (CSV ou Excel)", 
    type=["csv", "xlsx"], 
    accept_multiple_files=True # <-- A MÁGICA ACONTECE AQUI
)

# Verifica se a lista de arquivos não está vazia
if uploaded_files:
    lista_dfs = []
    
    # Processa cada arquivo enviado
    for file in uploaded_files:
        if file.name.endswith('.csv'):
            df_temp = pd.read_csv(file, sep=None, engine='python', decimal=',')
        else:
            df_temp = pd.read_excel(file)
            
        # DICA DE ENGENHARIA: Se a planilha individual não tiver a coluna 'Cenario',
        # o código usa o próprio nome do arquivo (ex: "Sim_01.xlsx" vira "Sim_01")
        if 'Cenario' not in df_temp.columns:
            # Extrai o nome do arquivo sem a extensão
            nome_cenario = file.name.rsplit('.', 1)[0]
            df_temp['Cenario'] = nome_cenario
            
        lista_dfs.append(df_temp)

    # Empilha todas as planilhas em um único DataFrame mestre
    df_dados = pd.concat(lista_dfs, ignore_index=True)

    try:
        # A partir daqui, o código não muda! Ele continua lendo o df_dados normalmente.
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
            #s_values = df_cenario[gas_cols].values
            #z_dict[cenario] = scaler.fit_transform(s_values)

            # Extrai os valores das porcentagens sem aplicar o z-score
            z_dict[cenario] = df_cenario[gas_cols].values

        # 3. Interface (Barra Lateral Dinâmica)
        st.sidebar.header("Parâmetros do Algoritmo")
        #threshold = st.sidebar.slider("Limiar de Corte (Threshold)", 0.1, 5.0, 2.0, 0.1)
        threshold = st.sidebar.slider("Limiar de Corte (Threshold)", 0.1, 100.0, 10.0, 0.1)
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
            st.markdown("---")
            st.subheader("Análise Automática do Melhor Corte (Otimização)")
            
            # 1. Varredura automática
            #range_thresholds = np.arange(0.1, 10.0, 0.1)
            range_thresholds = np.arange(1.0, 100.0, 0.5)
            lista_scores = []
            lista_thresholds = []
            
            for t in range_thresholds:
                # Simula o corte para cada limiar possível
                clusters_simulados = fcluster(Z, t, criterion='distance')
                n_zonas = len(set(clusters_simulados))
                
                # Só calcula a silhueta se gerar um agrupamento válido (entre 2 e N-1 zonas)
                if 1 < n_zonas < num_nodes:
                    score = silhouette_score(dist_matrix, clusters_simulados, metric='precomputed')
                    lista_scores.append(score)
                    lista_thresholds.append(t)
            
            # 2. Plota o gráfico para o usuário
            if len(lista_scores) > 0:
                melhor_score = max(lista_scores)
                melhor_t = lista_thresholds[lista_scores.index(melhor_score)]
                
                fig2, ax2 = plt.subplots(figsize=(10, 4))
                ax2.plot(lista_thresholds, lista_scores, marker='.', linestyle='-', color='#1f77b4')
                ax2.axvline(x=melhor_t, color='green', linestyle='--', label=f'Melhor Corte: {melhor_t:.1f}')
                ax2.axvline(x=threshold, color='red', linestyle='-', alpha=0.5, label='Seu Corte Atual')
                
                ax2.set_xlabel("Limiar de Corte (Threshold)")
                ax2.set_ylabel("Silhouette Score")
                ax2.set_title("Busca pelo Ponto Doce da Clusterização")
                ax2.legend()
                st.pyplot(fig2)
                
                st.success(f"💡 Dica do Algoritmo: Para maximizar a coesão matemática baseada nos cenários selecionados, mova o slider superior para **{melhor_t:.1f}** (Score previsto: {melhor_score:.3f}).")
            else:
                st.info("Não foi possível calcular uma curva de otimização com os parâmetros atuais.")
                
            # ==========================================
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


            # =========================================================
            # NOVO: MAPA TERRITORIAL COM NUVENS (CONVEX HULLS)
            # Motor: plotly.graph_objects + scipy.spatial
            # =========================================================
            st.markdown("---")
            st.subheader("Mapa Territorial de Zonas de PCS")

            df_results = pd.DataFrame({'Nó': node_labels, 'Zona_PCS': clusters})
            zone_counts = df_results['Zona_PCS'].value_counts()
            
            # 1. Preparação dos Dados (Mesma lógica anterior)
            df_coords = df_dados[df_dados['Cenario'] == cenarios_unicos[0]][['ID_No', 'Coordenada_X', 'Coordenada_Y']].copy()
            df_mapa = pd.merge(df_coords, df_results, left_on='ID_No', right_on='Nó')
            
            # Define uma paleta de cores forte e distinta para as zonas (Expanda se tiver muitas zonas)
            paleta_cores = px.colors.qualitative.Bold 

            # Inicializa a Figura vazia do motor de Baixo Nível
            fig_territorial = go.Figure()

            # 2. LOOP MATEMÁTICO: Calcular e desenhar as "Nuvens" para cada Zona
            zonas_unicas = sorted(df_mapa['Zona_PCS'].unique())

            for i, zona in enumerate(zonas_unicas):
                # Filtra os nós pertencentes apenas a esta zona
                df_zona = df_mapa[df_mapa['Zona_PCS'] == zona]
                
                # Pega as coordenadas X e Y como uma matriz NumPy
                pontos = df_zona[['Coordenada_X', 'Coordenada_Y']].values
                
                # DICA TÉCNICA: Uma envoltória convexa precisa de pelo menos 3 pontos
                # Se a zona for muito pequena (nó isolado), desenhamos apenas os pontos, sem nuvem.
                if len(pontos) >= 3:
                    try:
                        # O SciPy calcula quais são os pontos mais externos do grupo
                        hull = ConvexHull(pontos)
                        # Fecha o polígono voltando ao primeiro ponto
                        vertices = np.append(hull.vertices, hull.vertices[0])
                        
                        # Extrai as coordenadas X e Y dos vértices da fronteira
                        hull_x = pontos[vertices, 0]
                        hull_y = pontos[vertices, 1]
                        
                        # Define a cor desta zona baseada na paleta
                        cor_zona = paleta_cores[i % len(paleta_cores)]
                        
                        # ADICIONA A CAMADA DA NUVEM (Scatter com preenchimento)
                        fig_territorial.add_trace(go.Scatter(
                            x=hull_x, 
                            y=hull_y,
                            mode='lines',        # Desenha a linha de borda
                            name=f'Território Zona {zona}',
                            fill='toself',       # Preenche o interior do polígono
                            fillcolor=cor_zona, # Cor do preenchimento
                            opacity=0.2,         # Torna a nuvem translúcida (efeito cloud)
                            line=dict(color=cor_zona, width=1, dash='dot'), # Linha de borda pontilhada
                            hoverinfo='skip',    # Não mostra balão ao passar o mouse na nuvem
                            showlegend=True      # Mostra esta zona na legenda
                        ))
                    except:
                        # Em casos raros onde pontos são colineares, o SciPy falha. Ignoramos.
                        pass

            # 3. ADICIONA A CAMADA DOS PONTOS (Nós da Rede) por cima das nuvens
            for i, zona in enumerate(zonas_unicas):
                df_zona = df_mapa[df_mapa['Zona_PCS'] == zona]
                cor_zona = paleta_cores[i % len(paleta_cores)]
                
                # Pontos normais (Nós de consumo)
                fig_territorial.add_trace(go.Scatter(
                    x=df_zona['Coordenada_X'], 
                    y=df_zona['Coordenada_Y'],
                    mode='markers+text', # Mostra o ponto e o nome
                    name=f'Nós Zona {zona}',
                    text=df_zona['Nó'],  # Texto a exibir
                    textposition="top center",
                    marker=dict(
                        size=12, 
                        color=cor_zona, # Cor sólida da zona
                        line=dict(width=1, color='DarkSlateGrey')
                    ),
                    hoverinfo='text',    # Mostra apenas o nome no mouse
                    showlegend=False      # Não duplicar na legenda (já temos a nuvem)
                ))

            # 4. Configuração Final do Layout (Visual)
            fig_territorial.update_layout(
                title=f"Zoneamento Territorial de PCS (Ponto Doce: Corte {threshold}%)",
                xaxis=dict(title="Eixo X (Mestros)", showgrid=False, zeroline=False),
                yaxis=dict(title="Eixo Y (Metros)", showgrid=False, zeroline=False),
                plot_bgcolor='rgba(248, 249, 250, 1)', # Fundo cinza muito claro
                height=700,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            st.plotly_chart(fig_territorial, use_container_width=True)
            # ==========================================
    
    except Exception as e:
        st.error(f"Erro ao processar os dados. Verifique o formato da planilha. Detalhes técnicos: {e}")
else:
    st.info("Aguardando o upload do arquivo base (csv ou xlsx)...")

