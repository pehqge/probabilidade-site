import streamlit as st
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import os
import gzip
import shutil
import atexit
import json
import re
import io
from google import genai
from google.genai import types

# Configurar o layout da página para wide mode
st.set_page_config(
    page_title="MetObjects Explorer",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título do aplicativo
st.title("🏛️ Metropolitan Museum of Art - Explorador de Dados")
st.markdown("### Análise interativa da coleção do Metropolitan Museum of Art")

# Adicionar explicação sobre o processo de banco de dados
with st.expander("ℹ️ Informações sobre o banco de dados", expanded=False):
    st.markdown("""
    Este aplicativo utiliza um banco de dados SQLite para armazenar e consultar 
    a coleção do Metropolitan Museum of Art.
    
    O banco de dados será automaticamente excluído quando você 
    fechar o aplicativo para economizar espaço em disco.
    
    Se você encontrar problemas com o banco de dados, tente reiniciar o aplicativo.
    """)

# Caminho para o banco de dados
DB_PATH = "metobjects.db"
GZIP_PATH = "database.gz"

# Função para descompactar o arquivo database.gz
def descompactar_database():
    try:
        # Verificar se o arquivo compactado existe
        if os.path.exists(GZIP_PATH):
            # Verificar se o banco já está descompactado
            if not os.path.exists(DB_PATH):
                st.info("Preparando o banco de dados... Por favor, aguarde...")
                status = st.status("Preparando...", expanded=True)
                with gzip.open(GZIP_PATH, 'rb') as f_in:
                    with open(DB_PATH, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                status.update(label="Banco de dados pronto!", state="complete", expanded=False)
            return True
        else:
            st.error(f"Arquivo de banco de dados não encontrado")
            st.error("O aplicativo não pode funcionar sem o arquivo de banco de dados.")
            st.markdown("""
            ### Solução:
            1. Verifique se o arquivo necessário está presente no diretório do aplicativo.
            2. Reinicie o aplicativo após resolver o problema.
            """)
            return False
    except Exception as e:
        st.error(f"Erro ao preparar o banco de dados: {e}")
        return False

# Função para excluir o banco de dados quando o aplicativo for encerrado
def excluir_database():
    if os.path.exists(DB_PATH):
        try:
            # Tentar fechar todas as conexões com o banco de dados
            try:
                conn = sqlite3.connect(DB_PATH)
                conn.close()
            except:
                pass
            
            # Excluir o arquivo do banco de dados
            os.remove(DB_PATH)
            print(f"Banco de dados {DB_PATH} excluído com sucesso!")
        except Exception as e:
            print(f"Erro ao excluir o banco de dados: {e}")

# Função para consultar a API do Gemini para gerar consultas SQL ou analisar dados
def consultar_ia(pergunta, schema_info):
    try:
        # Inicializar o cliente Gemini com a chave da API dos secrets do Streamlit
        API_KEY = st.secrets["API_KEY"]
        client = genai.Client(api_key=API_KEY)
        
        # Preparar o prompt com informações sobre o esquema do banco
        prompt = f"""
        Você é um assistente especializado em SQL para o banco de dados do Metropolitan Museum of Art.
        
        Informações sobre o esquema do banco de dados:
        {schema_info}
        
        A consulta do usuário é: "{pergunta}"
        
        Se o usuário estiver pedindo uma consulta SQL:
        1. Gere APENAS o código SQL que atenda à solicitação
        2. Use a sintaxe SQLite
        3. Coloque aspas duplas em nomes de colunas com espaços
        4. Retorne apenas o código SQL sem explicações
        5. As categorias devem ser exibidas pelo nome completo entre aspas duplas exemplo: "Object Name"
        6. A resposta não deve ser em markdown, APENAS o código SQL
        
        Se o usuário estiver fazendo uma pergunta geral sobre o banco de dados:
        1. Forneça uma resposta clara e direta
        2. Mencione também uma consulta SQL que pode ser usada para obter esses dados
        
        Resposta:
        """
        
        # Configurar o modelo e a solicitação
        model = "gemini-2.0-flash-lite"
        contents = [
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=prompt)],
            ),
        ]
        
        generate_content_config = types.GenerateContentConfig(
            temperature=0.2,
            top_p=0.95,
            top_k=40,
            max_output_tokens=2048,
            response_mime_type="text/plain",
        )
        
        # Fazer a chamada da API
        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=generate_content_config,
        )
        
        return response.text
    except Exception as e:
        return f"Erro ao consultar a IA: {e}"

# Registrar a função para ser executada ao encerrar o aplicativo
atexit.register(excluir_database)

# Descompactar o banco de dados
if not descompactar_database():
    st.stop()

# Verificar se o banco de dados existe após descompactar
if not os.path.exists(DB_PATH):
    st.error(f"Banco de dados não encontrado: {DB_PATH}")
    st.info("Verifique se os arquivos necessários estão presentes no diretório.")
    st.stop()

# Função para executar consultas SQL
@st.cache_data(ttl=3600)
def executar_consulta(query):
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    except Exception as e:
        st.error(f"Erro ao executar consulta: {e}")
        return pd.DataFrame()

# Função para obter as colunas da tabela
@st.cache_data(ttl=3600)
def obter_colunas():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(metobjects)")
    colunas = [col[1] for col in cursor.fetchall()]
    conn.close()
    return colunas

# Função para obter valores únicos de uma coluna
@st.cache_data(ttl=3600)
def obter_valores_unicos(coluna):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(f'SELECT DISTINCT "{coluna}" FROM metobjects WHERE "{coluna}" != "" ORDER BY "{coluna}"')
    valores = [val[0] for val in cursor.fetchall()]
    conn.close()
    return valores

# Função para obter o esquema do banco de dados
@st.cache_data(ttl=7200)
def obter_schema_info():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Obter informações sobre a tabela
    cursor.execute("PRAGMA table_info(metobjects)")
    colunas = cursor.fetchall()
    
    # Obter alguns exemplos de dados para cada coluna
    schema_info = "Tabela: metobjects\n\nColunas:\n"
    
    for col in colunas:
        col_id, nome, tipo, notnull, default_val, pk = col
        schema_info += f"- {nome} ({tipo})\n"
        
        # Obter alguns valores distintos para esta coluna (se não for muito grande)
        try:
            cursor.execute(f'SELECT DISTINCT "{nome}" FROM metobjects WHERE "{nome}" IS NOT NULL AND "{nome}" != "" LIMIT 5')
            exemplos = cursor.fetchall()
            if exemplos:
                schema_info += f"  Exemplos: {', '.join([str(ex[0]) for ex in exemplos])}\n"
        except:
            pass
    
    # Obter contagens para algumas colunas importantes
    cursor.execute("SELECT COUNT(*) FROM metobjects")
    total_rows = cursor.fetchone()[0]
    schema_info += f"\nTotal de registros: {total_rows}\n"
    
    # Obter estatísticas para algumas colunas categóricas importantes
    for coluna in ["Department", "Culture", "Object Name", "Classification"]:
        try:
            cursor.execute(f'SELECT COUNT(DISTINCT "{coluna}") FROM metobjects WHERE "{coluna}" != ""')
            distinct_count = cursor.fetchone()[0]
            schema_info += f"Total de {coluna} distintos: {distinct_count}\n"
        except:
            pass
    
    conn.close()
    return schema_info

# Função para obter estatísticas básicas
@st.cache_data(ttl=3600)
def obter_estatisticas():
    stats = {}
    
    # Total de objetos
    query = "SELECT COUNT(*) FROM metobjects"
    stats['total_objetos'] = executar_consulta(query).iloc[0, 0]
    
    # Total de departamentos
    query = "SELECT COUNT(DISTINCT Department) FROM metobjects WHERE Department != ''"
    stats['total_departamentos'] = executar_consulta(query).iloc[0, 0]
    
    # Total de culturas
    query = "SELECT COUNT(DISTINCT Culture) FROM metobjects WHERE Culture != ''"
    stats['total_culturas'] = executar_consulta(query).iloc[0, 0]
    
    # Total de artistas
    query = 'SELECT COUNT(DISTINCT "Artist Display Name") FROM metobjects WHERE "Artist Display Name" != ""'
    stats['total_artistas'] = executar_consulta(query).iloc[0, 0]
    
    # Total de tipos de objetos
    query = 'SELECT COUNT(DISTINCT "Object Name") FROM metobjects WHERE "Object Name" != ""'
    stats['total_tipos_objetos'] = executar_consulta(query).iloc[0, 0]
    
    return stats

# Função para criar visualização de departamentos
def visualizar_departamentos():
    query = """
    SELECT Department, COUNT(*) as Count 
    FROM metobjects 
    WHERE Department != '' 
    GROUP BY Department 
    ORDER BY Count DESC
    """
    
    df = executar_consulta(query)
    
    # Calcular a porcentagem
    total = df['Count'].sum()
    df['Porcentagem'] = (df['Count'] / total * 100).round(2)
    
    # Criar gráfico interativo com Plotly
    fig = px.bar(
        df, 
        y='Department', 
        x='Count', 
        orientation='h',
        text=df['Porcentagem'].apply(lambda x: f'{x:.2f}%'),
        color='Count',
        color_continuous_scale='Blues',
        title='Distribuição de Objetos por Departamento'
    )
    
    fig.update_layout(
        xaxis_title='Número de Objetos',
        yaxis_title='Departamento',
        height=600
    )
    
    return fig, df

# Função para criar visualização de objetos por tipo
def visualizar_objetos_por_tipo():
    query = """
    SELECT "Object Name", COUNT(*) as Count 
    FROM metobjects 
    WHERE "Object Name" != '' 
    GROUP BY "Object Name" 
    ORDER BY Count DESC
    """
    
    df = executar_consulta(query)
    
    # Criar gráfico interativo com Plotly
    fig = px.bar(
        df.head(50), 
        y='Object Name', 
        x='Count', 
        orientation='h',
        color='Count',
        color_continuous_scale='Viridis',
        title='Top 50 Tipos de Objetos (mostrando os 50 mais comuns de um total de ' + str(len(df)) + ')'
    )
    
    fig.update_layout(
        xaxis_title='Número de Objetos',
        yaxis_title='Tipo de Objeto',
        height=700
    )
    
    return fig, df

# Função para criar visualização de culturas
def visualizar_culturas():
    query = """
    SELECT Culture, COUNT(*) as Count 
    FROM metobjects 
    WHERE Culture != '' 
    GROUP BY Culture 
    ORDER BY Count DESC
    """
    
    df = executar_consulta(query)
    
    # Criar gráfico interativo com Plotly
    fig = px.pie(
        df.head(30), 
        values='Count', 
        names='Culture',
        title='Distribuição de Objetos por Cultura (mostrando as 30 principais de um total de ' + str(len(df)) + ')'
    )
    
    fig.update_layout(height=600)
    
    return fig, df

# Função para filtrar objetos 
def filtrar_objetos():
    # Obter as colunas disponíveis
    colunas = obter_colunas()
    
    # Criar opções para filtros nas colunas mais comuns
    st.subheader("Filtros")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        departamento = st.selectbox(
            "Departamento", 
            ["Todos"] + obter_valores_unicos("Department"),
            index=0
        )
    
    with col2:
        cultura = st.selectbox(
            "Cultura", 
            ["Todas"] + obter_valores_unicos("Culture"),
            index=0
        )
    
    with col3:
        tipo_objeto = st.text_input("Tipo de Objeto (contém):", "")
    
    col4, col5, col6 = st.columns(3)
    
    with col4:
        artista = st.text_input("Artista (contém):", "")
    
    with col5:
        data_objeto = st.text_input("Data (contém):", "")
    
    with col6:
        is_domain_publico = st.selectbox(
            "Domínio Público", 
            ["Qualquer", "Sim", "Não"],
            index=0
        )
    
    # Construir a consulta SQL
    query = 'SELECT * FROM metobjects WHERE 1=1'
    
    if departamento != "Todos":
        query += f' AND Department = "{departamento}"'
    
    if cultura != "Todas":
        query += f' AND Culture = "{cultura}"'
    
    if tipo_objeto:
        query += f' AND "Object Name" LIKE "%{tipo_objeto}%"'
    
    if artista:
        query += f' AND "Artist Display Name" LIKE "%{artista}%"'
    
    if data_objeto:
        query += f' AND "Object Date" LIKE "%{data_objeto}%"'
    
    if is_domain_publico != "Qualquer":
        value = "True" if is_domain_publico == "Sim" else "False"
        query += f' AND "Is Public Domain" = "{value}"'
    
    # Executar a consulta
    df = executar_consulta(query)
    
    return df

# Função para visualizar dados de um objeto específico
def visualizar_objeto(objeto_id):
    query = f'SELECT * FROM metobjects WHERE "Object ID" = "{objeto_id}"'
    df = executar_consulta(query)
    
    if len(df) == 0:
        st.error("Objeto não encontrado")
        return
    
    # Obter os dados do objeto
    obj = df.iloc[0]
    
    col1, col2 = st.columns([2, 3])
    
    with col1:
        st.subheader(obj["Title"] if obj["Title"] else "Sem título")
        
        if obj["Artist Display Name"]:
            st.write(f"**Artista:** {obj['Artist Display Name']}")
        
        if obj["Object Date"]:
            st.write(f"**Data:** {obj['Object Date']}")
        
        if obj["Culture"]:
            st.write(f"**Cultura:** {obj['Culture']}")
        
        if obj["Medium"]:
            st.write(f"**Meio:** {obj['Medium']}")
        
        if obj["Dimensions"]:
            st.write(f"**Dimensões:** {obj['Dimensions']}")
        
        if obj["Credit Line"]:
            st.write(f"**Crédito:** {obj['Credit Line']}")
        
        if obj["Department"]:
            st.write(f"**Departamento:** {obj['Department']}")
        
        # Link para o objeto no site do museu
        if obj["Object ID"]:
            object_url = f"https://www.metmuseum.org/art/collection/search/{obj['Object ID']}"
            st.markdown(f"[Ver no site do Metropolitan Museum 🔗]({object_url})")
    
    with col2:
        # Link Resource pode conter a URL da imagem
        if obj["Link Resource"] and obj["Is Public Domain"] == "True":
            st.image(obj["Link Resource"], caption=obj["Title"], use_column_width=True)
        else:
            st.info("Imagem não disponível ou não está em domínio público")
            
            # Se tiver URL do Wikidata, mostrar link
            if obj["Object Wikidata URL"]:
                st.markdown(f"[Ver no Wikidata 🔗]({obj['Object Wikidata URL']})")

# Função para criar visualização personalizada
def criar_visualizacao_personalizada():
    st.subheader("Criar Visualização Personalizada")
    
    # Obter as colunas disponíveis
    colunas = obter_colunas()
    colunas_categoricas = [col for col in colunas if col not in ["Object ID", "Dimensions"]]
    
    # Seleção do tipo de gráfico
    tipo_grafico = st.selectbox(
        "Tipo de Gráfico",
        ["Barras", "Pizza", "Dispersão", "Linha"],
        index=0
    )
    
    # Configuração do gráfico
    col1, col2 = st.columns(2)
    
    with col1:
        coluna_x = st.selectbox("Selecione a coluna para agrupar", colunas_categoricas)
        
        limite = st.slider("Limite de dados", 5, 50, 15)
        
    with col2:
        # Agregação para contagem
        if tipo_grafico in ["Barras", "Pizza"]:
            agregacao = "COUNT(*)"
            legenda_y = "Contagem"
        else:
            # Para gráficos de dispersão/linha, precisa de uma segunda variável
            colunas_numericas = ["Object ID"]  # Poderia incluir outras se tivéssemos colunas numéricas
            coluna_y = st.selectbox("Selecione a coluna para o eixo Y", colunas_numericas)
            agregacao = f'AVG("{coluna_y}")'
            legenda_y = f"Média de {coluna_y}"
    
    # Construir consulta base
    query = f"""
    SELECT "{coluna_x}", {agregacao} as Y
    FROM metobjects
    WHERE "{coluna_x}" != ""
    GROUP BY "{coluna_x}"
    ORDER BY Y DESC
    LIMIT {limite}
    """
    
    # Executar a consulta
    df = executar_consulta(query)
    
    # Criar visualização
    if len(df) > 0:
        if tipo_grafico == "Barras":
            fig = px.bar(
                df, 
                x=coluna_x, 
                y="Y",
                color=coluna_x,
                title=f'Distribuição por {coluna_x}',
                labels={coluna_x: coluna_x, "Y": legenda_y}
            )
        
        elif tipo_grafico == "Pizza":
            fig = px.pie(
                df, 
                values="Y", 
                names=coluna_x,
                title=f'Distribuição por {coluna_x}',
                hole=0.3
            )
        
        elif tipo_grafico == "Dispersão":
            fig = px.scatter(
                df, 
                x=coluna_x, 
                y="Y",
                color=coluna_x,
                title=f'{legenda_y} por {coluna_x}',
                labels={coluna_x: coluna_x, "Y": legenda_y},
                size="Y",
                size_max=60
            )
        
        elif tipo_grafico == "Linha":
            # Ordenar por nome da coluna para linha
            df = df.sort_values(by=coluna_x)
            
            fig = px.line(
                df, 
                x=coluna_x, 
                y="Y",
                markers=True,
                title=f'{legenda_y} por {coluna_x}',
                labels={coluna_x: coluna_x, "Y": legenda_y}
            )
        
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
        
        # Exibir os dados
        with st.expander("Ver dados"):
            st.dataframe(df)
    else:
        st.warning("Não há dados suficientes para criar o gráfico selecionado")

# Função para executar SQL personalizado com ajuda da IA
def executar_sql_personalizado():
    st.subheader("⚙️ Consulta SQL Personalizada com Assistente IA")

    # Inicializar o estado da sessão para a consulta, se necessário
    if 'sql_query' not in st.session_state:
        st.session_state.sql_query = """
-- Conte o número de objetos por departamento
SELECT Department, COUNT(*) as Count 
FROM metobjects 
WHERE Department != '' 
GROUP BY Department 
ORDER BY Count DESC;"""

    st.markdown("### Gere SQL com IA")
    pergunta_ia = st.text_area(
        "Descreva em linguagem natural o que você quer consultar:",
        height=100,
        placeholder="Ex: Mostre todas as pinturas de Vincent van Gogh ordenadas por data"
    )

    if st.button("🤖 Gerar SQL com IA"):
        if pergunta_ia:
            with st.spinner("A IA está processando sua consulta..."):
                try:
                    # Obter informações do esquema
                    schema_info = obter_schema_info()
                    # Consultar a IA
                    resposta = consultar_ia(pergunta_ia, schema_info)
                    
                    # Verificar se a resposta parece ser SQL
                    is_sql_query = "SELECT" in resposta.upper() and "FROM" in resposta.upper()
                    
                    if is_sql_query:
                        # Extrair apenas a consulta SQL
                        sql_match = re.search(r'(SELECT.+?;)', resposta, re.DOTALL | re.IGNORECASE)
                        if sql_match:
                            consulta_sql_gerada = sql_match.group(1)
                        else:
                            # Tentar pegar tudo se não encontrar o padrão exato
                            consulta_sql_gerada = resposta.strip()
                        
                        st.session_state.sql_query = consulta_sql_gerada
                        st.success("Consulta SQL gerada e inserida abaixo!")
                        # Força o rerender para atualizar a text_area
                        st.rerun() 
                    else:
                        st.warning(f"A IA retornou uma resposta que não parece SQL: {resposta}")
                except Exception as e:
                    st.error(f"Erro ao consultar a IA: {e}")
        else:
            st.error("Por favor, digite uma descrição para a IA gerar o SQL.")

    st.markdown("### Editor SQL")
    # Usar a chave explícita ligada ao session_state para atualização programática
    st.session_state.sql_query = st.text_area(
        "Edite ou digite sua consulta SQL aqui:",
        value=st.session_state.sql_query, 
        height=250,
        key='sql_editor' # Chave para controle
    )
    
    if st.button("Executar Consulta"):
        query_para_executar = st.session_state.sql_query
        if query_para_executar:
            with st.spinner("Executando consulta..."):
                try:
                    # Executar a consulta
                    df = executar_consulta(query_para_executar)
                    
                    # Exibir os resultados
                    if not df.empty:
                        st.subheader("Resultados da Consulta")
                        st.dataframe(df)
                        
                        # Opção para baixar como CSV
                        csv = df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="Baixar como CSV",
                            data=csv,
                            file_name="resultado_consulta.csv",
                            mime="text/csv"
                        )
                    else:
                        st.info("A consulta não retornou resultados.")
                except Exception as e:
                    st.error(f"Erro ao executar a consulta SQL: {e}")
        else:
            st.error("A caixa de consulta SQL está vazia.")

# Função para consultar a API do Gemini para analisar um DataFrame
def consultar_ia_dataframe(pergunta, df):
    try:
        API_KEY = st.secrets["API_KEY"]
        client = genai.Client(api_key=API_KEY)
        
        # Preparar informações sobre o DataFrame para o prompt
        # Usar io.StringIO para capturar a saída de df.info()
        buffer = io.StringIO()
        df.info(buf=buffer)
        df_info_str = buffer.getvalue()
        
        prompt = f"""
        Você é um assistente de análise de dados IA. Você está analisando um DataFrame pandas carregado pelo usuário.

        Informações sobre o DataFrame ('df'):
        Número de linhas: {len(df)}
        Número de colunas: {len(df.columns)}

        Informações das colunas (df.info()):
        {df_info_str}

        Primeiras 5 linhas (df.head()):
        {df.head().to_string()}

        A pergunta do usuário é: "{pergunta}"

        Instruções:
        1. Responda à pergunta do usuário da forma mais clara e concisa possível.
        2. Se a pergunta exigir uma análise que gere um resultado tabular (como uma contagem de valores, descrição estatística, correlação, etc.), gere APENAS o código Python necessário usando o DataFrame 'df' (que já está disponível para você). Use pandas para manipulação e coloque o resultado final em uma variável chamada 'result_df'. Exemplo: result_df = df['Coluna'].value_counts().reset_index()
        3. Se a pergunta exigir uma visualização (gráfico de barras, dispersão, pizza, etc.), gere APENAS o código Python necessário usando plotly.express (importado como 'px') ou plotly.graph_objects (importado como 'go'). Use o DataFrame 'df'. Coloque a figura gerada em uma variável chamada 'fig'. Exemplo: fig = px.histogram(df, x='NomeDaColuna')
        4. Se for gerar código Python, NÃO adicione explicações, apenas o bloco de código Python formatado entre ```python e ```.
        5. NÃO use st.write, st.dataframe, st.plotly_chart ou qualquer outra função do Streamlit ('st') dentro do código gerado. Apenas calcule 'result_df' ou 'fig'.
        6. Se a pergunta for geral ou não puder ser respondida com o código, forneça uma resposta textual direta.
        7. Para análises estatísticas comuns, você pode usar df.describe().
        8. Se precisar de colunas específicas, use os nomes exatos das colunas fornecidos em df.info() e df.head(). Lembre-se que nomes de colunas podem ter espaços ou caracteres especiais, use df['Nome Coluna Com Espaço'].

        Resposta:
        """
        
        model = "gemini-1.5-flash-latest" # Usar um modelo mais capaz para análise e código
        contents = [types.Content(role="user", parts=[types.Part.from_text(text=prompt)])]
        
        generate_content_config = types.GenerateContentConfig(
            temperature=0.3,
            top_p=0.95,
            top_k=40,
            max_output_tokens=4096, # Aumentar tokens para código + explicação potencial
            response_mime_type="text/plain",
        )
        
        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=generate_content_config,
        )
        
        return response.text
    except Exception as e:
        st.error(f"Erro ao consultar a IA: {e}")
        return f"Erro: {e}"

# Função para a página do Chatbot CSV Analyzer
def chatbot_csv_analyzer():
    st.subheader("🤖 Chatbot Analisador de CSV")

    # Inicializar estado da sessão se necessário
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    if "uploaded_df" not in st.session_state:
        st.session_state.uploaded_df = None
    if "file_uploader_key" not in st.session_state:
         st.session_state.file_uploader_key = 0 # Para resetar o uploader

    uploaded_file = st.file_uploader(
        "Carregue seu arquivo CSV aqui", 
        type="csv", 
        key=f"uploader_{st.session_state.file_uploader_key}"
    )

    if uploaded_file is not None:
        # Tentar ler o CSV e armazenar no estado da sessão
        if st.session_state.uploaded_df is None: # Só carrega se ainda não tiver carregado
            try:
                # Ler o arquivo carregado em um DataFrame
                df = pd.read_csv(uploaded_file)
                st.session_state.uploaded_df = df
                st.session_state.chat_messages = [] # Limpar chat ao carregar novo arquivo
                st.success("Arquivo CSV carregado com sucesso!")
                 # Limpar o uploader incrementando a chave
                st.session_state.file_uploader_key += 1
                st.rerun() # Força o rerender para mostrar o estado inicial do chat e info
            except Exception as e:
                st.error(f"Erro ao ler o arquivo CSV: {e}")
                st.session_state.uploaded_df = None # Reset em caso de erro
    
    # Se um DataFrame estiver carregado
    if st.session_state.uploaded_df is not None:
        df = st.session_state.uploaded_df
        
        # Mostrar informações básicas sobre o DF
        st.markdown("### Informações do CSV Carregado")
        st.markdown(f"**Nome do arquivo:** {uploaded_file.name if uploaded_file else 'N/A'}")
        st.markdown(f"**Número de Linhas:** {len(df)}")
        st.markdown(f"**Número de Colunas:** {len(df.columns)}")
        
        with st.expander("Ver Nomes das Colunas"):
             st.write(df.columns.tolist())
        with st.expander("Ver Primeiras Linhas (Head)"):
            st.dataframe(df.head())
        
        st.markdown("---")
        st.markdown("### Chat com IA")

        # Exibir histórico do chat
        for message in st.session_state.chat_messages:
            with st.chat_message(message["role"]):
                # Se for código Python na mensagem da IA, formatar como código
                if message["role"] == "assistant" and "```python" in message["content"]:
                     code_match = re.search(r"```python\n(.*)\n```", message["content"], re.DOTALL)
                     if code_match:
                         st.code(code_match.group(1), language="python")
                     else: # Caso não consiga extrair, mostrar como texto normal
                         st.markdown(message["content"])
                # Se for resultado (df ou fig), exibir diretamente
                elif "result_df" in message:
                     st.dataframe(message["result_df"])
                elif "fig" in message:
                     st.plotly_chart(message["fig"], use_container_width=True)
                else:
                    st.markdown(message["content"])

        # Input do usuário
        if prompt := st.chat_input("Faça uma pergunta sobre seus dados..."):
            # Adicionar mensagem do usuário ao histórico e exibir
            st.session_state.chat_messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Gerar resposta da IA
            with st.chat_message("assistant"):
                with st.spinner("Pensando..."):
                    response_text = consultar_ia_dataframe(prompt, df)
                    
                    # Verificar se a IA gerou código Python
                    code_to_execute = None
                    if "```python" in response_text:
                        code_match = re.search(r"```python\n(.*)\n```", response_text, re.DOTALL)
                        if code_match:
                            code_to_execute = code_match.group(1).strip()
                            # Adicionar mensagem da IA (código) ao histórico
                            st.session_state.chat_messages.append({"role": "assistant", "content": response_text}) 
                            st.code(code_to_execute, language="python") # Mostra o código gerado
                        else:
                             # Se não conseguiu extrair, tratar como texto normal
                             st.session_state.chat_messages.append({"role": "assistant", "content": response_text})
                             st.markdown(response_text)
                    else:
                        # Resposta textual normal da IA
                        st.session_state.chat_messages.append({"role": "assistant", "content": response_text})
                        st.markdown(response_text)

                    # Se houver código para executar
                    if code_to_execute:
                        try:
                            # Preparar o ambiente de execução
                            # Passamos o DataFrame 'df', pandas 'pd', plotly 'px'/'go', streamlit 'st', numpy 'np'
                            # e locals() para capturar as variáveis resultantes ('result_df', 'fig')
                            local_vars = {}
                            global_vars = {'pd': pd, 'px': px, 'go': go, 'np': np, 'st': st, 'df': df}
                            exec(code_to_execute, global_vars, local_vars)
                            
                            # Verificar se 'result_df' ou 'fig' foram criados pela execução
                            if 'result_df' in local_vars:
                                result_df_output = local_vars['result_df']
                                st.dataframe(result_df_output)
                                # Adicionar o resultado ao histórico para persistência
                                st.session_state.chat_messages.append({"role": "assistant", "result_df": result_df_output})
                            elif 'fig' in local_vars:
                                fig_output = local_vars['fig']
                                st.plotly_chart(fig_output, use_container_width=True)
                                # Adicionar o resultado ao histórico para persistência
                                st.session_state.chat_messages.append({"role": "assistant", "fig": fig_output})

                        except Exception as exec_e:
                            st.error(f"Erro ao executar o código gerado pela IA: {exec_e}")
                            st.session_state.chat_messages.append({"role": "assistant", "content": f"Erro ao executar código: {exec_e}"})
            # st.rerun() # Rerun pode ser opcional aqui, pois o chat atualiza

    else:
        # Mensagem inicial se nenhum arquivo foi carregado ainda
        st.info("Por favor, carregue um arquivo CSV para começar a análise.")
        # Limpar estado se o usuário remover o arquivo (ou para garantir limpeza inicial)
        st.session_state.uploaded_df = None
        st.session_state.chat_messages = []

# Interface principal (modificada para adicionar a nova página)
def main():
    # Barra lateral
    st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/7/7a/The_MET_logo.png", width=200)
    st.sidebar.title("Navegação")
    
    # Menu principal - ADICIONAR "Chatbot CSV"
    pagina = st.sidebar.radio(
        "Escolha uma seção:",
        ["Visão Geral", "Filtrar Objetos", "Análise por Departamento", 
         "Análise por Tipo de Objeto", "Análise por Cultura", 
         "Busca por ID", "Visualização Personalizada", "Consulta SQL", 
         "Chatbot CSV"] # Adicionado aqui
    )
    
    # Obter estatísticas gerais (só para a parte do MetObjects)
    stats = {}
    if os.path.exists(DB_PATH): # Só calcula se o DB original existir
         try:
              stats = obter_estatisticas()
         except Exception as e:
              print(f"Não foi possível calcular estatísticas iniciais: {e}")
              stats = {'total_objetos': 'N/A', 'total_departamentos': 'N/A', 
                       'total_culturas': 'N/A', 'total_artistas': 'N/A', 'total_tipos_objetos': 'N/A'}
    else:
         stats = {'total_objetos': 'N/A', 'total_departamentos': 'N/A', 
                  'total_culturas': 'N/A', 'total_artistas': 'N/A', 'total_tipos_objetos': 'N/A'}

    # Mostrar informações do banco de dados na barra lateral
    st.sidebar.subheader("Informações do Banco (MetObjects)")
    if os.path.exists(DB_PATH):
        tamanho_db = os.path.getsize(DB_PATH) / (1024 * 1024)  # Tamanho em MB
        st.sidebar.info(f"""
        💾 **Banco de Dados:**
        - Arquivo: {DB_PATH}
        - Tamanho: {tamanho_db:.2f} MB
        - Status: Temporário (será excluído ao fechar)
        """)
    else:
        st.sidebar.warning("Banco de dados MetObjects não encontrado.")

    st.sidebar.info(f"""
    📊 **Estatísticas (MetObjects):**
    - Total de objetos: {stats.get('total_objetos', 'N/A')}
    - Departamentos: {stats.get('total_departamentos', 'N/A')}
    - Culturas: {stats.get('total_culturas', 'N/A')}
    - Artistas: {stats.get('total_artistas', 'N/A')}
    - Tipos de objetos: {stats.get('total_tipos_objetos', 'N/A')}
    """)
    
    # Footer na barra lateral
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "Desenvolvido para análise da coleção do [Metropolitan Museum of Art](https://www.metmuseum.org/)"
    )
    st.sidebar.markdown("Funcionalidade de Chatbot CSV adicionada.")
    
    # Conteúdo principal
    if pagina == "Visão Geral":
        st.subheader("📊 Visão Geral da Coleção")
        if stats.get('total_objetos') != 'N/A':
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total de Objetos", f"{stats['total_objetos']:,}")
            col2.metric("Departamentos", stats['total_departamentos'])
            col3.metric("Culturas", stats['total_culturas'])
            col4.metric("Artistas", f"{stats['total_artistas']:,}")
            
            st.markdown("---")
            
            st.subheader("Distribuição por Departamento")
            fig_dept, df_dept = visualizar_departamentos()
            st.plotly_chart(fig_dept, use_container_width=True)
            
            st.markdown("---")
            
            st.subheader("Top 50 Tipos de Objetos") 
            fig_tipo, df_tipo = visualizar_objetos_por_tipo()
            st.plotly_chart(fig_tipo, use_container_width=True)
            
            st.markdown("---")
            
            st.subheader("Principais Culturas")
            fig_cult, df_cult = visualizar_culturas()
            st.plotly_chart(fig_cult, use_container_width=True)
        else:
            st.warning("Banco de dados MetObjects não disponível para exibir a visão geral.")

    elif pagina == "Filtrar Objetos":
        st.subheader("Filtrar Objetos")
        if os.path.exists(DB_PATH):
             df_filtrado = filtrar_objetos()
             if not df_filtrado.empty:
                 st.subheader(f"Resultados: {len(df_filtrado)} objetos encontrados")
                 colunas_padrao = ["Object Number", "Object Name", "Title", "Artist Display Name", "Object Date", "Culture", "Department"]
                 colunas_disponiveis = df_filtrado.columns.tolist()
                 colunas_padrao_existentes = [col for col in colunas_padrao if col in colunas_disponiveis]
                 colunas_selecionadas = st.multiselect("Selecione as colunas para exibir:", colunas_disponiveis, default=colunas_padrao_existentes)
                 if colunas_selecionadas:
                     st.dataframe(df_filtrado[colunas_selecionadas])
                     csv = df_filtrado[colunas_selecionadas].to_csv(index=False).encode('utf-8')
                     st.download_button(label="Baixar como CSV", data=csv, file_name="objetos_filtrados.csv", mime="text/csv")
                 elif not colunas_disponiveis:
                     st.warning("O resultado da filtragem não contém colunas.")
                 else:
                     st.warning("Selecione pelo menos uma coluna para exibir")
             else:
                 st.info("Nenhum objeto encontrado com os filtros selecionados")
        else:
             st.warning("Banco de dados MetObjects não disponível para filtrar objetos.")

    elif pagina == "Análise por Departamento":
        st.subheader("🏛️ Análise por Departamento")
        if os.path.exists(DB_PATH):
             fig_dept, df_dept = visualizar_departamentos()
             st.plotly_chart(fig_dept, use_container_width=True)
             st.subheader("Dados por Departamento")
             st.dataframe(df_dept)
             st.subheader("Objetos mais comuns por Departamento")
             departamentos_validos = obter_valores_unicos("Department")
             if departamentos_validos:
                 departamento = st.selectbox("Selecione um departamento:", departamentos_validos)
                 query = f'''SELECT "Object Name", COUNT(*) as Count FROM metobjects WHERE Department = '{departamento}' AND "Object Name" != '' GROUP BY "Object Name" ORDER BY Count DESC LIMIT 10'''
                 df_objetos = executar_consulta(query)
                 if not df_objetos.empty:
                     fig = px.bar(df_objetos, x='Object Name', y='Count', color='Count', color_continuous_scale='Teal', title=f'Top 10 Tipos de Objetos no Departamento: {departamento}')
                     st.plotly_chart(fig, use_container_width=True)
                 else:
                     st.info(f"Não foram encontrados tipos de objeto para o departamento '{departamento}'.")
             else:
                 st.warning("Não há departamentos disponíveis para análise.")
        else:
             st.warning("Banco de dados MetObjects não disponível para análise por departamento.")
        
    elif pagina == "Análise por Tipo de Objeto":
        st.subheader("🖼️ Análise por Tipo de Objeto")
        if os.path.exists(DB_PATH):
             fig_tipo, df_tipo = visualizar_objetos_por_tipo()
             st.plotly_chart(fig_tipo, use_container_width=True)
             st.subheader("Dados por Tipo de Objeto")
             st.dataframe(df_tipo)
             st.subheader("Departamentos por Tipo de Objeto")
             tipos_comuns = obter_valores_unicos("Object Name")
             if tipos_comuns:
                 tipo_objeto = st.selectbox("Selecione um tipo de objeto:", tipos_comuns)
                 query = f'''SELECT Department, COUNT(*) as Count FROM metobjects WHERE "Object Name" = '{tipo_objeto}' AND Department != '' GROUP BY Department ORDER BY Count DESC LIMIT 10'''
                 df_depts = executar_consulta(query)
                 if not df_depts.empty:
                     fig = px.pie(df_depts, values='Count', names='Department', title=f'Distribuição de {tipo_objeto} por Departamento (Top 10)', hole=0.3)
                     st.plotly_chart(fig, use_container_width=True)
                 else:
                     st.info(f"Não foram encontrados departamentos para o tipo de objeto '{tipo_objeto}'.")
             else:
                 st.warning("Não há tipos de objeto disponíveis para análise.")
        else:
             st.warning("Banco de dados MetObjects não disponível para análise por tipo de objeto.")

    elif pagina == "Análise por Cultura":
        st.subheader("🌎 Análise por Cultura")
        if os.path.exists(DB_PATH):
             fig_cult, df_cult = visualizar_culturas()
             st.plotly_chart(fig_cult, use_container_width=True)
             st.subheader("Dados por Cultura")
             st.dataframe(df_cult)
             st.subheader("Objetos mais comuns por Cultura")
             culturas_validas = obter_valores_unicos("Culture")
             if culturas_validas:
                 cultura = st.selectbox("Selecione uma cultura:", culturas_validas)
                 query = f'''SELECT "Object Name", COUNT(*) as Count FROM metobjects WHERE Culture = '{cultura}' AND "Object Name" != '' GROUP BY "Object Name" ORDER BY Count DESC LIMIT 10'''
                 df_objetos = executar_consulta(query)
                 if not df_objetos.empty:
                     fig = px.bar(df_objetos, x='Object Name', y='Count', color='Count', color_continuous_scale='Viridis', title=f'Top 10 Tipos de Objetos na Cultura: {cultura}')
                     st.plotly_chart(fig, use_container_width=True)
                 else:
                     st.info(f"Não foram encontrados tipos de objeto para a cultura '{cultura}'.")
             else:
                  st.warning("Não há culturas disponíveis para análise.")
        else:
             st.warning("Banco de dados MetObjects não disponível para análise por cultura.")

    elif pagina == "Busca por ID":
        st.subheader("🔎 Busca por ID")
        if os.path.exists(DB_PATH):
            col1, col2 = st.columns([1, 2])
            with col1:
                object_id = st.text_input("Digite o ID do objeto:")
                if st.button("Buscar"):
                    if object_id:
                        object_id_clean = str(object_id).strip()
                        if object_id_clean:
                            visualizar_objeto(object_id_clean)
                        else:
                             st.error("Por favor, digite um ID de objeto válido.")
                    else:
                        st.error("Por favor, digite um ID de objeto")
                if st.button("Objeto Aleatório"):
                    query = '''SELECT "Object ID" FROM metobjects WHERE "Is Public Domain" = 'True' AND "Link Resource" != '' ORDER BY RANDOM() LIMIT 1'''
                    result = executar_consulta(query)
                    if not result.empty:
                        random_id = result.iloc[0, 0]
                        st.info(f"Exibindo objeto aleatório com ID: {random_id}")
                        visualizar_objeto(random_id)
                    else:
                        st.warning("Não foi possível encontrar um objeto aleatório com imagem em domínio público.")
            with col2:
                st.subheader("Exemplos de IDs (com imagem)")
                query = '''SELECT "Object ID", "Object Name", "Title", "Artist Display Name" FROM metobjects WHERE "Is Public Domain" = 'True' AND "Link Resource" != '' ORDER BY RANDOM() LIMIT 30'''
                df_examples = executar_consulta(query)
                if not df_examples.empty:
                    st.dataframe(df_examples)
                else:
                    st.info("Não foi possível carregar exemplos.")
        else:
             st.warning("Banco de dados MetObjects não disponível para busca por ID.")

    elif pagina == "Visualização Personalizada":
        st.subheader("📊 Criar Visualização Personalizada")
        if os.path.exists(DB_PATH):
            criar_visualizacao_personalizada()
        else:
            st.warning("Banco de dados MetObjects não disponível para visualização personalizada.")
        
    elif pagina == "Consulta SQL":
        if os.path.exists(DB_PATH):
            executar_sql_personalizado()
        else:
            st.warning("Banco de dados MetObjects não disponível para consulta SQL.")

    # ADICIONADO: Bloco para a nova página Chatbot CSV
    elif pagina == "Chatbot CSV":
        chatbot_csv_analyzer()

# Executar o aplicativo
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Ocorreu um erro durante a execução do aplicativo: {e}")
        # Adicionar traceback para debug em ambiente de desenvolvimento
        import traceback
        st.error("Detalhes do erro:")
        st.code(traceback.format_exc())
    finally:
        # O banco de dados será excluído pela função registrada no atexit
        pass 