# met_art_analyzer.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import io
from scipy import stats
import re # Adicionado para extrair código
import google.generativeai as genai # Garantir que está importado corretamente
import traceback 

# --- Configuração da Página ---
st.set_page_config(
    page_title="Met Art Analyzer PRO - Pinturas & Esculturas",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Constantes e Funções Auxiliares ---
CSV_PATH = "met_data_explorada.csv"
# Assumindo que 'Object Name' contém 'Painting' ou 'Sculpture'
# Ajuste se for outra coluna como 'Classification'
FILTER_COLUMN = "Object Name"
FILTER_VALUES = ["Painting", "Sculpture"]

# Cachear o carregamento e pré-processamento dos dados
@st.cache_data(ttl=3600) # Cache por 1 hora
def load_and_filter_data(csv_path, filter_column, filter_values):
    """Carrega o CSV e faz limpeza básica sem filtrar por tipo de objeto."""
    try:
        df = pd.read_csv(csv_path, low_memory=False) # low_memory=False pode ajudar com tipos mistos

        # Remover a filtragem inicial
        # df_filtered = df[df[filter_column].isin(filter_values)].copy()

        # Limpeza básica (exemplos - ajuste conforme necessário)
        # Converter anos para numérico, tratando erros
        df['AccessionYear'] = pd.to_numeric(df['AccessionYear'], errors='coerce')
        # Tentar extrair um ano de 'Object Date' (pode ser complexo)
        # Exemplo simples: pegar 4 dígitos seguidos (melhorar se necessário)
        df['ObjectYear'] = df['Object Date'].str.extract(r'(\d{4})', expand=False)
        df['ObjectYear'] = pd.to_numeric(df['ObjectYear'], errors='coerce')

        # Tratar colunas categóricas importantes como string e preencher NaNs
        for col in ['Department', 'Culture', 'Artist Display Name', 'Artist Nationality', 'Period', 'Medium', 'Country', 'Classification', 'City', 'Region']:
            if col in df.columns:
                df[col] = df[col].astype(str).fillna('Desconhecido').replace('', 'Desconhecido')
            else:
                st.warning(f"Coluna '{col}' esperada não encontrada no CSV.")

        return df

    except FileNotFoundError:
        st.error(f"Erro: Arquivo CSV não encontrado em '{csv_path}'. Certifique-se que ele está no mesmo diretório.")
        return None
    except Exception as e:
        st.error(f"Erro ao carregar ou processar o CSV: {e}")
        return None

def generate_insights(df):
    """Gera sugestões de temas e insights baseados nos dados."""
    insights = []
    if df is None or df.empty:
        return ["Não foi possível gerar insights: DataFrame vazio ou não carregado."]

    total_obras = len(df)
    insights.append(f"**Total de Obras Analisadas (Pinturas/Esculturas):** {total_obras}")

    # Contagem por tipo
    # tipo_counts = df[FILTER_COLUMN].value_counts()
    # insights.append("**Distribuição por Tipo:**")
    # for tipo, count in tipo_counts.items():
    #     insights.append(f"- {tipo}: {count} ({count/total_obras:.1%})")

    # Top Departamentos
    if 'Department' in df.columns:
        top_deptos = df['Department'].value_counts().nlargest(5)
        insights.append("\n**Top 5 Departamentos:**")
        insights.append(f"`{', '.join(top_deptos.index.tolist())}`")
        insights.append(f"*Sugestão:* Investigar as características das obras (tipos, culturas, períodos) predominantes nesses departamentos.")
        prob_top_depto = top_deptos.iloc[0] / total_obras
        insights.append(f"*Probabilidade:* A chance de uma obra ser do departamento '{top_deptos.index[0]}' é de aproximadamente {prob_top_depto:.1%}.")


    # Top Culturas
    if 'Culture' in df.columns:
        top_culturas = df[df['Culture'] != 'Desconhecido']['Culture'].value_counts().nlargest(5)
        if not top_culturas.empty:
            insights.append("\n**Top 5 Culturas (excluindo 'Desconhecido'):**")
            insights.append(f"`{', '.join(top_culturas.index.tolist())}`")
            insights.append(f"*Sugestão:* Analisar a relação entre essas culturas e os períodos/departamentos das obras.")

    # Top Artistas
    if 'Artist Display Name' in df.columns:
        top_artistas = df[df['Artist Display Name'] != 'Desconhecido']['Artist Display Name'].value_counts().nlargest(5)
        if not top_artistas.empty:
            insights.append("\n**Top 5 Artistas (excluindo 'Desconhecido'):**")
            insights.append(f"`{', '.join(top_artistas.index.tolist())}`")
            insights.append(f"*Sugestão:* Estudar o período de atividade e os tipos de obras desses artistas.")

    # Análise Temporal Básica
    if 'ObjectYear' in df.columns and df['ObjectYear'].notna().any():
        median_year = df['ObjectYear'].median()
        insights.append(f"\n**Análise Temporal (Baseada no Ano Extraído de 'Object Date'):**")
        insights.append(f"- Ano Mediano das Obras: {int(median_year)}")
        insights.append(f"*Sugestão:* Explorar a distribuição de obras por século ou período específico na aba 'Análise Temporal'.")

    # Domínio Público
    if 'Is Public Domain' in df.columns:
        public_domain_counts = df['Is Public Domain'].value_counts(normalize=True)
        prob_public = public_domain_counts.get(True, 0) # Usar True booleano se for o caso, ou 'True' string
        insights.append(f"\n**Domínio Público:**")
        insights.append(f"- Aproximadamente {prob_public:.1%} das obras estão em domínio público.")
        insights.append(f"*Sugestão:* Comparar características (departamento, período) entre obras em domínio público e as que não estão.")

    # Combinações Frequentes (Exemplo: Departamento x Tipo)
    if 'Department' in df.columns:
        try:
            common_combo = df.groupby(['Department', FILTER_COLUMN]).size().nlargest(3)
            insights.append(f"\n**Combinações Comuns (Departamento x Tipo):**")
            for index, count in common_combo.items():
                 insights.append(f"- {index[0]} / {index[1]}: {count} obras")
            insights.append(f"*Sugestão:* Investigar por que essas combinações são tão frequentes. Há especialização dos departamentos?")
        except Exception as e:
            insights.append(f"\n*Não foi possível analisar combinações Departamento x Tipo: {e}*")


    insights.append("\n---")
    insights.append("**Próximos Passos Sugeridos:**")
    insights.append("1. Explore as abas de **Análise Exploratória** para visualizar as distribuições de cada variável.")
    insights.append("2. Use a aba **Probabilidades** para calcular chances específicas.")
    insights.append("3. Investigue **Artistas e Culturas** em detalhe.")
    insights.append("4. Verifique a **Análise Temporal** para tendências ao longo dos anos.")
    insights.append("5. Utilize a aba **Explorar Dados** para filtrar e visualizar subconjuntos específicos.")

    return insights

# --- Funções de Consulta à IA ---

def safe_exec(code_string, global_vars=None, local_vars=None):
    """Executa código Python de forma um pouco mais segura (sem acesso direto a builtins perigosos)."""
    if global_vars is None:
        global_vars = {}
    # Limitar acesso a builtins perigosos
    safe_builtins = {
        'print': print, 'len': len, 'range': range, 'list': list, 'dict': dict, 'set': set,
        'str': str, 'int': int, 'float': float, 'bool': bool, 'True': True, 'False': False, 'None': None,
        'abs': abs, 'max': max, 'min': min, 'round': round, 'sum': sum, 'zip': zip, 'enumerate': enumerate,
        'isinstance': isinstance, 'Exception': Exception,
    }
    global_vars['__builtins__'] = safe_builtins
    exec(code_string, global_vars, local_vars)


@st.cache_data(ttl=600) # Cache curto para chamadas repetidas
def consultar_ia_geral(prompt, model="gemini-1.5-flash-latest", temperature=0.5, max_tokens=1024):
    """Função genérica para consultar a IA."""
    try:
        if "API_KEY" not in st.secrets:
             st.error("Chave da API do Google não configurada em st.secrets.")
             return "Erro: Chave da API não configurada."
        API_KEY = st.secrets["API_KEY"]

        # ATIVAR CHAMADA REAL DA API
        genai.configure(api_key=API_KEY)
        model_instance = genai.GenerativeModel(model)
        response = model_instance.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens
            )
        )
        # Verificar se a resposta contém texto antes de acessá-lo
        # A estrutura exata pode variar um pouco dependendo da API/versão
        if response and response.parts:
             return response.text
        elif response and hasattr(response, 'text'): # Outra forma comum
             return response.text
        else:
            # Tentar logar a resposta bruta para depuração se não encontrar texto
            print(f"Resposta inesperada da API: {response}") 
            return "Erro: Resposta da IA em formato inesperado ou vazia."

        # REMOVER CÓDIGO DE SIMULAÇÃO
        # st.warning("Funcionalidade de IA desabilitada (requer configuração de API Key).")
        # return "Funcionalidade de IA desabilitada."

    except Exception as e:
        # Incluir detalhes do erro no log/output para diagnóstico
        st.error(f"Erro ao consultar a IA: {e}")
        print(f"Detalhes do erro na consulta IA: {traceback.format_exc()}") # Adicionar traceback ao log
        return f"Erro ao consultar a IA: {e}"


def get_df_summary(df):
    """Gera um resumo conciso do DataFrame para incluir no prompt da IA."""
    buffer = io.StringIO()
    df.info(buf=buffer)
    df_info_str = buffer.getvalue()
    summary = f"""
Resumo do DataFrame ('df'):
- Linhas: {len(df)}, Colunas: {len(df.columns)}
- Colunas e Tipos (df.info()):
{df_info_str}
- Estatísticas Descritivas (Numéricas):
{df.describe().to_string()}
- Primeiras 3 Linhas (df.head(3)):
{df.head(3).to_string()}
"""
    return summary


def consultar_ia_dataframe_expert(pergunta, df):
    """Consulta a IA configurada como especialista em análise de dados/arte."""
    df_summary = get_df_summary(df)
    prompt = f"""
    Você é um assistente IA especialista em análise de dados e história da arte, analisando um DataFrame pandas ('df') com dados de pinturas e esculturas do Metropolitan Museum of Art.

    {df_summary}

    A pergunta/pedido do usuário é: "{pergunta}"

    Instruções DETALHADAS:
    1.  **Compreensão e Resposta:** Analise a pergunta. Responda de forma clara, concisa e informativa, como um especialista faria. Se possível, forneça interpretação dos resultados no contexto da história da arte.
    2.  **Geração de Código Python (SE NECESSÁRIO):**
        *   Se a pergunta exigir cálculo, agregação, filtragem ou análise estatística, gere o código Python necessário usando o DataFrame 'df'.
        *   **IMPORTANTE:** As seguintes bibliotecas JÁ ESTÃO DISPONÍVEIS no escopo e prontas para uso: `pandas` (como `pd`), `numpy` (como `np`), `scipy.stats` (como `stats`), `plotly.express` (como `px`), `plotly.graph_objects` (como `go`). **NÃO inclua NENHUMA declaração `import` para essas bibliotecas no código gerado.**
        *   Para visualizações, use `px` ou `go`. Gere a figura na variável 'fig'.
        *   Para resultados tabulares, coloque o DataFrame resultante em 'result_df'.
        *   **Formato do Código:** Envolva TODO o código Python gerado estritamente entre ```python e ```. NÃO inclua NENHUM texto explicativo antes ou depois do bloco de código.
        *   **Simplicidade:** Gere o código mais direto possível para a tarefa. NÃO use funções do Streamlit (st.) dentro do bloco de código.
    3.  **Saída:**
        *   Se gerar código, a resposta DEVE conter apenas o bloco ```python ... ```.
        *   Se a resposta for puramente textual (interpretação, explicação), forneça apenas o texto.
        *   Se a resposta contiver tanto uma análise (código) quanto uma interpretação, forneça PRIMEIRO o bloco de código ```python ... ``` e DEPOIS a interpretação textual, separada por uma linha em branco.
    4.  **Contexto:** Lembre-se que os dados são sobre Pinturas e Esculturas. Use os nomes exatos das colunas do resumo. Considere as limitações dos dados (ex: 'ObjectYear' extraído pode não ser perfeito).

    Resposta:
    """
    # Usando a função genérica (ajuste o modelo/parâmetros se necessário)
    return consultar_ia_geral(prompt, model="gemini-1.5-pro-latest", temperature=0.4, max_tokens=3000) # Modelo mais capaz


def consultar_ia_sugestao_teste(df, col1_name, col2_name=None):
    """Pede à IA para sugerir um teste estatístico apropriado."""
    df_summary = get_df_summary(df[[col1_name] + ([col2_name] if col2_name else [])])
    prompt = f"""
    Você é um assistente estatístico. O usuário selecionou a(s) seguinte(s) coluna(s) de um DataFrame para análise:
    - Coluna 1: '{col1_name}' (Tipo: {df[col1_name].dtype})
    {f"- Coluna 2: '{col2_name}' (Tipo: {df[col2_name].dtype})" if col2_name else ""}

    {df_summary}

    Tarefa: Sugira um teste estatístico apropriado para investigar a relação ou diferença entre essas colunas.
    Se for uma coluna numérica, considere testes de média. Se forem categóricas, considere testes de associação.
    Forneça o nome do teste (ex: 'Teste t de amostras independentes', 'Qui-quadrado de independência') e uma breve justificativa (1-2 frases) do porquê ele é adequado, mencionando as suposições básicas.
    Responda apenas com o nome do teste e a justificativa.

    Exemplo de resposta para 1 coluna numérica:
    'Teste t de uma amostra. Adequado para testar se a média da coluna '{col1_name}' é significativamente diferente de um valor específico. Assume distribuição aproximadamente normal ou amostra grande.'

    Exemplo de resposta para 1 numérica e 1 categórica (binária):
    'Teste t de amostras independentes. Adequado para comparar as médias da coluna '{col1_name}' entre os dois grupos definidos pela coluna '{col2_name}'. Assume normalidade (ou N grande) e variâncias homogêneas (pode ser relaxado).'

    Exemplo de resposta para 2 categóricas:
    'Teste Qui-quadrado de independência (Chi-Square). Adequado para verificar se há associação estatisticamente significativa entre as categorias de '{col1_name}' e '{col2_name}'. Assume frequências esperadas adequadas.'

    Sugestão:
    """
    return consultar_ia_geral(prompt, temperature=0.2, max_tokens=256)


def consultar_ia_gerar_hipotese(df):
    """Pede à IA para gerar hipóteses testáveis a partir dos dados."""
    df_summary = get_df_summary(df)
    prompt = f"""
    Você é um pesquisador assistente de IA analisando dados de arte (pinturas e esculturas).

    {df_summary}

    Tarefa: Baseado no resumo dos dados, gere 3 hipóteses interessantes e ESTATISTICAMENTE TESTÁVEIS sobre relações, diferenças ou tendências nos dados. Formule cada hipótese claramente (H0 e H1 se possível, ou apenas a questão de pesquisa) e sugira brevemente como poderia ser testada (quais colunas e qual tipo de teste).

    Exemplo de Hipótese:
    'Hipótese 1: Existe uma diferença significativa no ano mediano de criação ('ObjectYear') entre Pinturas e Esculturas ('{FILTER_COLUMN}').
     Teste Sugerido: Teste de Mann-Whitney U (não paramétrico para medianas) ou Teste t (se as médias e suposições forem apropriadas).'

    Gere 3 hipóteses:
    """
    return consultar_ia_geral(prompt, temperature=0.7, max_tokens=512)


def consultar_ia_gerar_filtro_pandas(pergunta, df):
    """Pede à IA para gerar uma string de EXPRESSÃO BOOLEANA Pandas."""
    df_summary = get_df_summary(df)
    prompt = f"""
    Você é um assistente de IA especialista em Pandas. Sua tarefa é traduzir uma pergunta do usuário em linguagem natural para uma EXPRESSÃO BOOLEANA Python/Pandas completa e válida que possa ser avaliada para gerar uma máscara de filtro (uma Série booleana).

    DataFrame Resumo (disponível como variável 'df'):
    {df_summary}

    Pergunta do Usuário: "{pergunta}"

    Instruções:
    1.  Analise a pergunta e identifique as condições de filtragem.
    2.  Gere APENAS a expressão booleana Python/Pandas completa. Esta expressão, quando avaliada, deve retornar uma Série booleana (True/False para cada linha do DataFrame 'df').
    3.  Use a sintaxe padrão de acesso a colunas do Pandas: `df['Nome da Coluna']`. Use os nomes exatos das colunas conforme aparecem no resumo.
    4.  Para strings, use aspas duplas (ex: `df['Department'] == "European Paintings"`).
    5.  Combine múltiplas condições com `&` (AND), `|` (OR), `~` (NOT). Use parênteses `()` para garantir a ordem correta das operações.
    6.  Para verificações de "contém", use `.str.contains()` (ex: `df['Artist Display Name'].str.contains(\"van Gogh\", case=False, na=False)`). Use `na=False` para tratar valores ausentes.
    7.  Para verificações de nulos/não nulos, use `.isna()` ou `.notna()` (ex: `df['Culture'].notna()`).
    8.  Para verificações de pertencimento a uma lista, use `.isin()` (ex: `df['Country'].isin(["Italy", "France"])`).
    9.  NÃO inclua `import`, `df = ...`, ou qualquer outra coisa além da expressão booleana única.
    10. NÃO adicione explicações, comentários ou qualquer texto fora da expressão.
    11. Se a pergunta for ambígua ou não puder ser traduzida em um filtro claro, retorne a string 'AMBIGUOUS_QUERY'1
    12. MAIS IMPORTANTE: SEMPRE RETORNAR A EXPRESSÃO BOOLEANA PANDAS, NÃO RETORNAR NADA ALÉM DA EXPRESSÃO.
    13. NAO COLOCAR COMO MARKDOWN MOSTRANDO QUE É PYTHON, APENAS RETORNAR A EXPRESSÃO.

    Exemplo 1:
    Pergunta: "Mostrar obras do departamento de pinturas europeias feitas depois de 1800"
    Resposta: `(df['Department'] == "European Paintings") & (df['ObjectYear'] > 1800)`

    Exemplo 2:
    Pergunta: "Esculturas que não são do Egito"
    Resposta: `(df['Object Name'] == "Sculpture") & (df['Culture'] != "Egyptian")`

    Exemplo 3:
    Pergunta: "Obras do artista Van Gogh ou Monet"
    Resposta: `df['Artist Display Name'].str.contains(\"Van Gogh|Monet\", case=False, regex=True, na=False)`

    Exemplo 4:
    Pergunta: "Peças em domínio público cujo país é Itália ou França"
    Resposta: `(df['Is Public Domain'] == True) & (df['Country'].isin(["Italy", "France"]))`

    Exemplo 5:
    Pergunta: "Obras cujo nome do artista está preenchido"
    Resposta: `df['Artist Display Name'].notna() & (df['Artist Display Name'] != 'Desconhecido')`

    Expressão Booleana Pandas:
    """
    return consultar_ia_geral(prompt, model="gemini-1.5-flash-latest", temperature=0.1, max_tokens=512) # Mais preciso


# --- Funções das Páginas ---

def render_visao_geral(df):
    st.title("🏠 Visão Geral e Insights Aprimorados")
    st.markdown("Análise inicial e geração de ideias para seu trabalho de estatística e probabilidade.")
    st.markdown("---")

    if df is None or df.empty:
        st.warning("Dados não carregados ou vazios.")
        return

    # Métricas Principais
    col1, col2, col3 = st.columns(3)
    col1.metric("Total de Obras (Filtradas)", f"{len(df):,}")
    if 'Department' in df.columns:
        col2.metric("Departamentos Únicos", f"{df['Department'].nunique():,}")
    if 'Artist Display Name' in df.columns:
         col3.metric("Artistas Únicos (Conhecidos)", f"{df[df['Artist Display Name'] != 'Desconhecido']['Artist Display Name'].nunique():,}")

    st.markdown("---")
    st.subheader("💡 Insights e Sugestões de Temas")
    st.info("Pontos iniciais e sugestões aprofundadas:")

    insights_list = generate_insights(df)
    for insight in insights_list:
        st.markdown(insight)

    # Botão para gerar hipóteses com IA
    st.markdown("---")
    st.subheader("🧠 Gerador de Hipóteses (IA)")
    if st.button("Sugerir Hipóteses Testáveis"):
        with st.spinner("IA pensando em hipóteses..."):
             hipoteses = consultar_ia_gerar_hipotese(df)
             st.markdown(hipoteses)


def render_analise_exploratoria(df):
    st.title("📊 Análise Exploratória Detalhada")
    st.markdown("Explore as distribuições e relações entre as variáveis.")
    st.markdown("---")

    if df is None or df.empty:
        st.warning("Dados não carregados ou vazios.")
        return

    tab1, tab2, tab3 = st.tabs(["📈 Análise Univariada", "🔗 Análise Bivariada", "📅 Análise Temporal (Geral)"])

    # --- Análise Univariada ---
    with tab1:
        st.subheader("Análise de Coluna Individual")
        col_options = [col for col in df.columns if df[col].nunique() < 100 or pd.api.types.is_numeric_dtype(df[col])] # Selecionar cols relevantes
        column_to_analyze = st.selectbox("Selecione a coluna para analisar:", col_options, index=col_options.index('Department') if 'Department' in col_options else 0)

        if column_to_analyze:
            st.markdown(f"**Analisando:** `{column_to_analyze}`")

            # Tentar mostrar tipo de dado
            try:
                 st.write(f"**Tipo de Dado:** {df[column_to_analyze].dtype}")
            except: pass

            # Se for numérica
            if pd.api.types.is_numeric_dtype(df[column_to_analyze]):
                st.markdown("**Distribuição Numérica:**")
                fig = px.histogram(df, x=column_to_analyze, title=f"Distribuição de {column_to_analyze}", marginal="box")
                st.plotly_chart(fig, use_container_width=True)
                st.markdown("**Estatísticas Descritivas:**")
                st.dataframe(df[column_to_analyze].describe())

            # Se for categórica (ou tratada como string/object)
            else:
                st.markdown("**Distribuição Categórica:**")
                counts = df[column_to_analyze].value_counts()
                df_counts = counts.reset_index()
                df_counts.columns = [column_to_analyze, 'Contagem']

                # Mostrar top N categorias
                top_n = st.slider(f"Mostrar Top N categorias para '{column_to_analyze}':", min_value=5, max_value=50, value=15, key=f'slider_{column_to_analyze}')
                df_plot = df_counts.head(top_n)

                fig = px.bar(df_plot, y=column_to_analyze, x='Contagem', orientation='h',
                             title=f"Top {top_n} Valores Mais Comuns para {column_to_analyze}",
                             text='Contagem', height=max(400, top_n*25)) # Altura dinâmica
                fig.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig, use_container_width=True)

                with st.expander(f"Ver todas as contagens para '{column_to_analyze}' ({len(df_counts)} únicas)"):
                    st.dataframe(df_counts)

    # --- Análise Bivariada ---
    with tab2:
        st.subheader("Relação Entre Duas Colunas")
        cat_cols = df.select_dtypes(include=['object', 'category', 'boolean']).columns.tolist()
        # Remover colunas com excesso de categorias para clareza
        cat_cols_filtered = [col for col in cat_cols if df[col].nunique() < 50 and col != 'Desconhecido']

        col_x = st.selectbox("Selecione a Coluna X (Variável de Agrupamento/Eixo X):", cat_cols_filtered, index=cat_cols_filtered.index('Department') if 'Department' in cat_cols_filtered else 0, key='sel_x')
        col_y = st.selectbox("Selecione a Coluna Y (Variável para Comparar/Cor):", cat_cols_filtered, index=cat_cols_filtered.index(FILTER_COLUMN) if FILTER_COLUMN in cat_cols_filtered else 0, key='sel_y')

        if col_x and col_y and col_x != col_y:
            st.markdown(f"**Analisando:** Relação entre `{col_x}` e `{col_y}`")

            # Tabela de Contingência (Crosstab)
            try:
                contingency_table = pd.crosstab(df[col_x], df[col_y])
                with st.expander("Ver Tabela de Contingência (Contagens)"):
                     st.dataframe(contingency_table)

                # Gráfico de Barras Empilhadas/Agrupadas (Normalizado)
                contingency_norm = pd.crosstab(df[col_x], df[col_y], normalize='index') * 100 # Normaliza por linha (índice X)
                fig = px.bar(contingency_norm, barmode='stack', # ou 'group'
                             title=f"Distribuição Percentual de '{col_y}' dentro de cada '{col_x}'",
                             labels={'value': 'Percentual (%)', 'index': col_x})
                st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"Não foi possível gerar análise bivariada para '{col_x}' e '{col_y}': {e}")
        else:
             st.info("Selecione duas colunas diferentes para análise.")


    # --- Análise Temporal Geral ---
    with tab3:
        st.subheader("Tendências Temporais Gerais")
        # Usar 'AccessionYear' ou 'ObjectYear' se disponível
        date_col_options = []
        if 'AccessionYear' in df.columns and df['AccessionYear'].notna().any():
            date_col_options.append('AccessionYear')
        if 'ObjectYear' in df.columns and df['ObjectYear'].notna().any():
             date_col_options.append('ObjectYear') # Ano extraído

        if not date_col_options:
             st.warning("Nenhuma coluna de ano adequada ('AccessionYear', 'ObjectYear') encontrada.")
        else:
             date_col_to_use = st.selectbox("Selecione a coluna de ano para análise:", date_col_options)

             if date_col_to_use:
                  st.markdown(f"**Analisando:** Contagem de Obras por `{date_col_to_use}`")
                  # Remover NaNs e converter para inteiro para agregação
                  year_counts = df.dropna(subset=[date_col_to_use])[date_col_to_use].astype(int).value_counts().sort_index()

                  # Agrupar por década para melhor visualização se houver muitos anos
                  if year_counts.index.nunique() > 100:
                      df_temp = df.dropna(subset=[date_col_to_use]).copy()
                      df_temp[date_col_to_use] = df_temp[date_col_to_use].astype(int)
                      df_temp['Decade'] = (df_temp[date_col_to_use] // 10) * 10
                      decade_counts = df_temp['Decade'].value_counts().sort_index()
                      # Filtrar décadas muito antigas/irrelevantes se necessário
                      decade_counts = decade_counts[decade_counts.index > 1500] # Exemplo
                      fig = px.line(decade_counts, x=decade_counts.index, y=decade_counts.values,
                                    title=f"Contagem de Obras por Década ({date_col_to_use})",
                                    labels={'index': 'Década', 'y': 'Número de Obras'}, markers=True)
                      st.plotly_chart(fig, use_container_width=True)

                  else:
                      # Plotar por ano se não forem tantos assim
                      year_counts = year_counts[year_counts.index > 1500] # Exemplo de filtro
                      fig = px.line(year_counts, x=year_counts.index, y=year_counts.values,
                                    title=f"Contagem de Obras por Ano ({date_col_to_use})",
                                    labels={'index': 'Ano', 'y': 'Número de Obras'}, markers=True)
                      st.plotly_chart(fig, use_container_width=True)


def render_probabilidades(df):
    st.title("🎲 Probabilidades e Estatísticas")
    st.markdown("Calcule probabilidades e veja estatísticas sobre os dados.")
    st.markdown("---")

    if df is None or df.empty:
        st.warning("Dados não carregados ou vazios.")
        return

    cat_cols = df.select_dtypes(include=['object', 'category', 'boolean']).columns.tolist()
    cat_cols_filtered = [col for col in cat_cols if df[col].nunique() < 100 and col != 'Desconhecido'] # Relevantes

    st.subheader("Probabilidades Marginais")
    st.markdown("Qual a probabilidade de uma obra pertencer a uma categoria específica?")
    col_prob = st.selectbox("Selecione a coluna para calcular a probabilidade:", cat_cols_filtered, key='prob_col')

    if col_prob:
        prob_dist = df[col_prob].value_counts(normalize=True)
        df_prob = prob_dist.reset_index()
        df_prob.columns = [col_prob, 'Probabilidade']
        st.dataframe(df_prob)

        # Visualizar as probabilidades mais altas
        top_n_prob = min(15, len(df_prob))
        fig = px.bar(df_prob.head(top_n_prob), x=col_prob, y='Probabilidade',
                     title=f"Probabilidade das Top {top_n_prob} Categorias em '{col_prob}'",
                     labels={col_prob: col_prob, 'Probabilidade': 'Probabilidade (0 a 1)'},
                     text=df_prob['Probabilidade'].head(top_n_prob).apply(lambda x: f'{x:.2%}'))
        st.plotly_chart(fig, use_container_width=True)


    st.subheader("Probabilidades Condicionais")
    st.markdown("Qual a probabilidade de uma categoria ocorrer, DADO que outra categoria ocorreu?")
    st.markdown("*Exemplo: Qual a probabilidade de uma obra ser uma 'Pintura' dado que ela é do departamento 'European Paintings'?*")

    col_condition = st.selectbox("Selecione a coluna da CONDIÇÃO (Dado que...):", cat_cols_filtered, key='cond_col')
    if col_condition:
        value_condition_options = df[col_condition].unique().tolist()
        # Limitar número de opções se forem muitas
        if len(value_condition_options) > 100:
            value_condition_options = df[col_condition].value_counts().nlargest(100).index.tolist()

        value_condition = st.selectbox(f"Selecione o valor específico para a condição '{col_condition}':", value_condition_options, key='cond_val')

        col_target = st.selectbox("Selecione a coluna ALVO (Probabilidade de...):",
                                  [c for c in cat_cols_filtered if c != col_condition], key='target_col')

        if value_condition and col_target:
            # Filtrar o DataFrame pela condição
            df_conditional = df[df[col_condition] == value_condition]

            if df_conditional.empty:
                st.warning(f"Nenhuma obra encontrada com a condição: '{col_condition}' = '{value_condition}'")
            else:
                st.markdown(f"**Calculando P({col_target} | {col_condition} = '{value_condition}')**")
                cond_prob_dist = df_conditional[col_target].value_counts(normalize=True)
                df_cond_prob = cond_prob_dist.reset_index()
                df_cond_prob.columns = [col_target, 'Probabilidade Condicional']
                st.dataframe(df_cond_prob)

                # Visualizar
                top_n_cond_prob = min(15, len(df_cond_prob))
                fig_cond = px.bar(df_cond_prob.head(top_n_cond_prob), x=col_target, y='Probabilidade Condicional',
                             title=f"Probabilidade de '{col_target}' dado '{col_condition}'='{value_condition}'",
                             labels={col_target: col_target, 'Probabilidade Condicional': 'Probabilidade (0 a 1)'},
                             text=df_cond_prob['Probabilidade Condicional'].head(top_n_cond_prob).apply(lambda x: f'{x:.2%}'))
                st.plotly_chart(fig_cond, use_container_width=True)


def render_artistas_culturas(df):
    st.title("🎨 Artistas e Culturas")
    st.markdown("Análise focada nos criadores e origens culturais das obras.")
    st.markdown("---")

    if df is None or df.empty:
        st.warning("Dados não carregados ou vazios.")
        return

    tab1, tab2 = st.tabs(["👨‍🎨 Análise de Artistas", "🌍 Análise de Culturas"])

    with tab1:
        st.subheader("Análise de Artistas")
        if 'Artist Display Name' not in df.columns:
            st.warning("Coluna 'Artist Display Name' não encontrada.")
            return

        # Top Artistas
        artistas = df[df['Artist Display Name'] != 'Desconhecido']['Artist Display Name'].value_counts()
        df_artistas = artistas.reset_index()
        df_artistas.columns = ['Artista', 'Contagem']
        top_n_artistas = st.slider("Mostrar Top N Artistas:", min_value=5, max_value=100, value=20, key='slider_artistas')
        fig_art = px.bar(df_artistas.head(top_n_artistas), y='Artista', x='Contagem', orientation='h',
                         title=f"Top {top_n_artistas} Artistas com Mais Obras", text='Contagem', height=max(400, top_n_artistas*25))
        fig_art.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_art, use_container_width=True)

        # Nacionalidades (se disponível)
        if 'Artist Nationality' in df.columns:
            st.subheader("Nacionalidade dos Artistas")
            nacionalidades = df[(df['Artist Display Name'] != 'Desconhecido') & (df['Artist Nationality'] != 'Desconhecido')]['Artist Nationality'].value_counts()
            df_nac = nacionalidades.reset_index()
            df_nac.columns = ['Nacionalidade', 'Contagem']
            top_n_nac = st.slider("Mostrar Top N Nacionalidades:", min_value=5, max_value=30, value=10, key='slider_nac')
            fig_nac = px.pie(df_nac.head(top_n_nac), names='Nacionalidade', values='Contagem',
                             title=f"Top {top_n_nac} Nacionalidades de Artistas (por nº de obras)", hole=0.3)
            st.plotly_chart(fig_nac, use_container_width=True)
            with st.expander("Ver contagem por nacionalidade"):
                st.dataframe(df_nac)

        # Relação Artista x Tipo de Obra (para o top artista)
        if not df_artistas.empty:
             top_artista_nome = df_artistas.iloc[0]['Artista']
             st.subheader(f"Tipos de Obras de '{top_artista_nome}'")
             obras_top_artista = df[df['Artist Display Name'] == top_artista_nome][FILTER_COLUMN].value_counts().reset_index()
             obras_top_artista.columns = ['Tipo de Obra', 'Contagem']
             fig_top_art_tipo = px.pie(obras_top_artista, names='Tipo de Obra', values='Contagem', title=f"Distribuição de Tipos de Obra para {top_artista_nome}")
             st.plotly_chart(fig_top_art_tipo, use_container_width=True)


    with tab2:
        st.subheader("Análise de Culturas")
        if 'Culture' not in df.columns:
            st.warning("Coluna 'Culture' não encontrada.")
            return

        # Top Culturas
        culturas = df[df['Culture'] != 'Desconhecido']['Culture'].value_counts()
        df_culturas = culturas.reset_index()
        df_culturas.columns = ['Cultura', 'Contagem']
        top_n_culturas = st.slider("Mostrar Top N Culturas:", min_value=5, max_value=50, value=20, key='slider_culturas')
        fig_cult = px.bar(df_culturas.head(top_n_culturas), y='Cultura', x='Contagem', orientation='h',
                         title=f"Top {top_n_culturas} Culturas (por nº de obras)", text='Contagem', height=max(400, top_n_culturas*25))
        fig_cult.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_cult, use_container_width=True)
        with st.expander("Ver contagem por cultura"):
             st.dataframe(df_culturas)

        # Relação Cultura x Departamento
        if 'Department' in df.columns:
             st.subheader("Relação Cultura x Departamento")
             cult_dept_table = pd.crosstab(df[df['Culture'] != 'Desconhecido']['Culture'],
                                           df[df['Department'] != 'Desconhecido']['Department'])
             # Filtrar para top N culturas para visualização
             top_culturas_list = df_culturas.head(top_n_culturas)['Cultura'].tolist()
             cult_dept_filtered = cult_dept_table[cult_dept_table.index.isin(top_culturas_list)]

             # Somar contagens por departamento para ordenação do heatmap
             dept_totals = cult_dept_filtered.sum().sort_values(ascending=False)
             # Selecionar top N departamentos para visualização
             top_depts_list = dept_totals.head(15).index.tolist()
             cult_dept_filtered = cult_dept_filtered[top_depts_list]


             if not cult_dept_filtered.empty:
                 fig_cult_dept = px.imshow(cult_dept_filtered.T, # Transpor para ter deptos no eixo Y
                                         title=f"Concentração de Obras: Top {top_n_culturas} Culturas vs Top 15 Departamentos",
                                         labels=dict(x="Cultura", y="Departamento", color="Contagem"),
                                         aspect="auto", height=600)
                 st.plotly_chart(fig_cult_dept, use_container_width=True)
                 with st.expander("Ver tabela de cruzamento Cultura x Departamento (filtrada)"):
                     st.dataframe(cult_dept_filtered)
             else:
                 st.info("Não foi possível gerar o heatmap Cultura x Departamento (dados insuficientes após filtragem).")


def render_analise_temporal(df):
    st.title("📅 Análise Temporal Aprofundada")
    st.markdown("Investigue tendências e padrões ao longo do tempo.")
    st.markdown("---")

    if df is None or df.empty:
        st.warning("Dados não carregados ou vazios.")
        return

    # Usar 'AccessionYear' ou 'ObjectYear'
    date_col_options = []
    if 'AccessionYear' in df.columns and df['AccessionYear'].notna().any():
        date_col_options.append('AccessionYear')
    if 'ObjectYear' in df.columns and df['ObjectYear'].notna().any():
         date_col_options.append('ObjectYear')

    if not date_col_options:
         st.warning("Nenhuma coluna de ano adequada ('AccessionYear', 'ObjectYear') encontrada.")
         return

    date_col_to_use = st.selectbox("Selecione a coluna de ano para análise:", date_col_options, key='temporal_date_col')

    if date_col_to_use:
        # Preparar DataFrame para análise temporal
        df_time = df.dropna(subset=[date_col_to_use]).copy()
        df_time[date_col_to_use] = df_time[date_col_to_use].astype(int)
        # Filtrar anos extremos/inválidos
        df_time = df_time[(df_time[date_col_to_use] > 1000) & (df_time[date_col_to_use] < 2025)]

        if df_time.empty:
            st.warning(f"Nenhum dado válido encontrado para '{date_col_to_use}' no range de anos selecionado.")
            return

        min_year, max_year = int(df_time[date_col_to_use].min()), int(df_time[date_col_to_use].max())
        year_range = st.slider("Selecione o intervalo de anos:", min_year, max_year, (min_year, max_year), key='temporal_slider')

        df_time_filtered = df_time[(df_time[date_col_to_use] >= year_range[0]) & (df_time[date_col_to_use] <= year_range[1])]

        if df_time_filtered.empty:
             st.warning(f"Nenhum dado encontrado no intervalo de anos selecionado: {year_range}")
             return

        # Contagem total ao longo do tempo
        st.subheader(f"Contagem de Obras por Ano ({year_range[0]}-{year_range[1]})")
        time_counts = df_time_filtered[date_col_to_use].value_counts().sort_index()
        fig_time = px.line(time_counts, x=time_counts.index, y=time_counts.values,
                           title=f"Número de Obras por Ano ({date_col_to_use})",
                           labels={'index': 'Ano', 'y': 'Número de Obras'})
        st.plotly_chart(fig_time, use_container_width=True)

        # Contagem por Tipo de Objeto ao longo do tempo
        st.subheader(f"Contagem por Tipo de Obra ao Longo do Tempo ({year_range[0]}-{year_range[1]})")
        time_type_counts = df_time_filtered.groupby([date_col_to_use, FILTER_COLUMN]).size().reset_index(name='Contagem')
        fig_time_type = px.line(time_type_counts, x=date_col_to_use, y='Contagem', color=FILTER_COLUMN,
                                title=f"Número de Pinturas vs. Esculturas por Ano ({date_col_to_use})",
                                labels={date_col_to_use: 'Ano', 'Contagem': 'Número de Obras'})
        st.plotly_chart(fig_time_type, use_container_width=True)

        # Opcional: Agrupar por década e ver por departamento
        if st.checkbox("Mostrar contagem por década e departamento (pode ser lento)"):
             df_time_filtered['Decade'] = (df_time_filtered[date_col_to_use] // 10) * 10
             if 'Department' in df.columns:
                 time_dept_counts = df_time_filtered.groupby(['Decade', 'Department']).size().reset_index(name='Contagem')
                 # Filtrar para top N departamentos para clareza
                 top_deptos_list = df['Department'].value_counts().nlargest(10).index.tolist()
                 time_dept_counts_filtered = time_dept_counts[time_dept_counts['Department'].isin(top_deptos_list)]

                 fig_time_dept = px.area(time_dept_counts_filtered, x='Decade', y='Contagem', color='Department',
                                         title=f"Número de Obras por Década e Top 10 Departamentos ({date_col_to_use})",
                                         labels={'Decade': 'Década', 'Contagem': 'Número de Obras'})
                 st.plotly_chart(fig_time_dept, use_container_width=True)
             else:
                 st.info("Coluna 'Department' não disponível para esta análise.")


def render_explorar_dados(df_original):
    st.title("💾 Explorar Dados com Filtro Inteligente")
    st.markdown("Visualize, filtre a tabela de dados (Pinturas/Esculturas) e use linguagem natural para refinar sua busca.")
    st.markdown("---")

    if df_original is None or df_original.empty:
        st.warning("Dados originais não carregados ou vazios.")
        return

    # --- Gerenciamento de Estado para Filtros --- 
    if 'df_explorar_display' not in st.session_state:
        st.session_state.df_explorar_display = df_original.copy()
    if 'ai_filter_query' not in st.session_state:
        st.session_state.ai_filter_query = ""
    if 'sidebar_filters_explorar' not in st.session_state:
        st.session_state.sidebar_filters_explorar = {}
    if 'reset_key_explorar' not in st.session_state:
        st.session_state.reset_key_explorar = 0 # Para forçar reset dos widgets

    # --- Filtros da Barra Lateral --- 
    st.sidebar.header("Filtros para Exploração")
    filter_cols = ['Department', 'Culture', 'Artist Display Name', 'Period', 'Is Public Domain', 'Country', 'Classification']
    available_filters = [col for col in filter_cols if col in df_original.columns]
    sidebar_filter_values = {}
    df_after_sidebar_filter = df_original.copy() # Começa com o original a cada rerun ANTES de aplicar sidebar

    for col in available_filters:
        options = sorted(df_original[col].unique().tolist())
        # Gerar chave única para cada widget baseado no reset_key
        widget_key = f'filter_{col}_{st.session_state.reset_key_explorar}'

        if len(options) < 50 and len(options) > 1 :
            # Usar valor do estado da sessão se existir, senão 'Todos'
            default_value = st.session_state.sidebar_filters_explorar.get(col, 'Todos')
            selected_val = st.sidebar.selectbox(f"Filtrar por {col}:", ['Todos'] + options, index=(['Todos'] + options).index(default_value) if default_value in ['Todos'] + options else 0, key=widget_key)
            sidebar_filter_values[col] = selected_val # Armazena seleção atual
            if selected_val != 'Todos':
                df_after_sidebar_filter = df_after_sidebar_filter[df_after_sidebar_filter[col] == selected_val]
        elif len(options) >= 50:
            default_value = st.session_state.sidebar_filters_explorar.get(col, "")
            text_search = st.sidebar.text_input(f"Filtrar {col} (contém):", value=default_value, key=widget_key)
            sidebar_filter_values[col] = text_search # Armazena seleção atual
            if text_search:
                df_after_sidebar_filter = df_after_sidebar_filter[df_after_sidebar_filter[col].astype(str).str.contains(text_search, case=False, na=False)]

    # Filtro de Ano (barra lateral)
    date_col_filter_options = [col for col in ['ObjectYear', 'AccessionYear'] if col in df_original.columns and df_original[col].notna().any()]
    date_filter_applied = False
    if date_col_filter_options:
         default_date_col = st.session_state.sidebar_filters_explorar.get('date_col', "Nenhum")
         date_col_filter = st.sidebar.selectbox("Filtrar por coluna de ano:", ["Nenhum"] + date_col_filter_options, index=(["Nenhum"] + date_col_filter_options).index(default_date_col) if default_date_col in ["Nenhum"] + date_col_filter_options else 0, key=f'filter_date_col_select_{st.session_state.reset_key_explorar}')
         sidebar_filter_values['date_col'] = date_col_filter

         if date_col_filter != "Nenhum":
             df_temp_filter = df_after_sidebar_filter.dropna(subset=[date_col_filter]).copy()
             df_temp_filter[date_col_filter] = df_temp_filter[date_col_filter].astype(int)
             if not df_temp_filter.empty:
                 min_yr_f, max_yr_f = int(df_temp_filter[date_col_filter].min()), int(df_temp_filter[date_col_filter].max())
                 if min_yr_f <= max_yr_f:
                    default_range = st.session_state.sidebar_filters_explorar.get('date_range', (min_yr_f, max_yr_f))
                    # Garantir que o default_range está dentro dos limites atuais
                    default_range = (max(min_yr_f, default_range[0]), min(max_yr_f, default_range[1]))

                    year_range_f = st.sidebar.slider(f"Intervalo para {date_col_filter}:", min_yr_f, max_yr_f, default_range, key=f'filter_slider_year_{st.session_state.reset_key_explorar}')
                    df_after_sidebar_filter = df_after_sidebar_filter[(df_after_sidebar_filter[date_col_filter].isna()) |
                                                                      ((df_after_sidebar_filter[date_col_filter] >= year_range_f[0]) & (df_after_sidebar_filter[date_col_filter] <= year_range_f[1]))]
                    sidebar_filter_values['date_range'] = year_range_f
                    date_filter_applied = True # Marca que um filtro de data foi aplicado

    # Atualiza o estado dos filtros da sidebar para persistência
    st.session_state.sidebar_filters_explorar = sidebar_filter_values

    # Atualiza o dataframe base para a IA SE filtros da sidebar mudaram
    # Compara com o estado anterior antes de aplicar filtros da IA
    # Isso é complexo de fazer perfeitamente sem comparar dataframes, então vamos simplificar:
    # Sempre reaplicamos o filtro AI sobre o resultado do filtro da sidebar atual.
    df_current_display = df_after_sidebar_filter.copy()

    # --- Filtro por Linguagem Natural (IA) --- 
    st.subheader("🔍 Filtrar com Linguagem Natural (IA)")
    nl_query = st.text_area("Descreva o filtro que você quer aplicar aos dados JÁ FILTRADOS acima:", 
                           placeholder="Ex: artistas franceses OU italianos", key="nl_filter_query")

    col_buttons1, col_buttons2, col_buttons3 = st.columns(3)
    with col_buttons1:
        if st.button("Adicionar Filtro", key="btn_add_filter"):
            if nl_query:
                with st.spinner("IA gerando filtro Pandas..."):
                    pandas_boolean_expression = consultar_ia_gerar_filtro_pandas(nl_query, st.session_state.df_explorar_display)
                    
                    if pandas_boolean_expression and pandas_boolean_expression != 'AMBIGUOUS_QUERY':
                        st.session_state.ai_filter_query = pandas_boolean_expression # Armazena a EXPRESSÃO da IA
                        st.info(f"Filtro IA gerado (Expressão): `{pandas_boolean_expression}`. Aplicando...")
                        
                        # Aplica a EXPRESSÃO usando safe_exec para obter a máscara booleana
                        try:
                            local_vars = {}
                            # Passar o dataframe atual (df_current_display) para o escopo da execução
                            global_vars = {'df': st.session_state.df_explorar_display, 'pd': pd, 'np': np} 
                            safe_exec(f"boolean_mask = {pandas_boolean_expression}", global_vars=global_vars, local_vars=local_vars)
                            
                            if 'boolean_mask' in local_vars:
                                # Aplicar a máscara gerada
                                st.session_state.df_explorar_display = st.session_state.df_explorar_display[local_vars['boolean_mask']]
                                st.success("Filtro IA aplicado!")
                                st.rerun() # Atualiza a exibição
                            else:
                                 st.error("Erro: A IA gerou uma expressão que não resultou em uma máscara booleana.")
                                 st.session_state.ai_filter_query = "" # Limpa query inválida

                        except Exception as e:
                             st.error(f"Erro ao AVALIAR a expressão da IA: `{pandas_boolean_expression}`. Erro: {e}")
                             # Se der erro, mantém o dataframe como estava antes de tentar aplicar o filtro AI
                             st.session_state.df_explorar_display = st.session_state.df_explorar_display 
                             st.session_state.ai_filter_query = "" # Limpa query inválida
                    elif pandas_boolean_expression == 'AMBIGUOUS_QUERY':
                         st.warning("IA considerou a pergunta ambígua. Tente reformular.")
                    else:
                         st.error("IA não conseguiu gerar um filtro. Tente novamente ou reformule a pergunta.")
            else:
                 st.error("Por favor, digite uma descrição para o filtro.")

    with col_buttons2:
        if st.button("Novo Filtro", key="btn_new_filter"):
            if nl_query:
                with st.spinner("IA gerando filtro Pandas..."):
                    pandas_boolean_expression = consultar_ia_gerar_filtro_pandas(nl_query, df_original)
                    
                    if pandas_boolean_expression and pandas_boolean_expression != 'AMBIGUOUS_QUERY':
                        st.session_state.ai_filter_query = pandas_boolean_expression # Armazena a EXPRESSÃO da IA
                        st.info(f"Filtro IA gerado (Expressão): `{pandas_boolean_expression}`. Aplicando...")
                        
                        # Aplica a EXPRESSÃO usando safe_exec para obter a máscara booleana
                        try:
                            local_vars = {}
                            # Passar o dataframe original (df_original) para o escopo da execução
                            global_vars = {'df': df_original, 'pd': pd, 'np': np} 
                            safe_exec(f"boolean_mask = {pandas_boolean_expression}", global_vars=global_vars, local_vars=local_vars)
                            
                            if 'boolean_mask' in local_vars:
                                # Aplicar a máscara gerada
                                st.session_state.df_explorar_display = df_original[local_vars['boolean_mask']]
                                st.success("Novo filtro IA aplicado!")
                                st.rerun() # Atualiza a exibição
                            else:
                                 st.error("Erro: A IA gerou uma expressão que não resultou em uma máscara booleana.")
                                 st.session_state.ai_filter_query = "" # Limpa query inválida

                        except Exception as e:
                             st.error(f"Erro ao AVALIAR a expressão da IA: `{pandas_boolean_expression}`. Erro: {e}")
                             # Se der erro, mantém o dataframe como estava antes de tentar aplicar o filtro AI
                             st.session_state.df_explorar_display = df_original 
                             st.session_state.ai_filter_query = "" # Limpa query inválida
                    elif pandas_boolean_expression == 'AMBIGUOUS_QUERY':
                         st.warning("IA considerou a pergunta ambígua. Tente reformular.")
                    else:
                         st.error("IA não conseguiu gerar um filtro. Tente novamente ou reformule a pergunta.")
            else:
                 st.error("Por favor, digite uma descrição para o filtro.")

    with col_buttons3:
         if st.button("Resetar TODOS os Filtros", key="btn_reset_filters"):
            st.session_state.df_explorar_display = df_original.copy()
            st.session_state.ai_filter_query = ""
            st.session_state.sidebar_filters_explorar = {} # Limpa estado dos filtros sidebar
            st.session_state.reset_key_explorar += 1 # Muda a chave para resetar widgets
            st.rerun()

    # --- Exibição da Tabela --- 
    st.subheader(f"Tabela de Dados ({len(st.session_state.df_explorar_display):,} linhas)")
    
    # Mostra filtros aplicados
    applied_filters_text = []
    for col, val in st.session_state.sidebar_filters_explorar.items():
        if val not in ['Todos', "", None, "Nenhum"]:
             # Tratamento especial para range de data
             if col == 'date_range':
                 date_col = st.session_state.sidebar_filters_explorar.get('date_col', 'Ano Desconhecido')
                 applied_filters_text.append(f"**{date_col}:** entre {val[0]}-{val[1]}")
             elif col != 'date_col': # Não mostrar a seleção da coluna de data em si
                 applied_filters_text.append(f"**{col}:** {val}")

    # Adicionar filtro da IA se existir
    if st.session_state.ai_filter_query:
         applied_filters_text.append(f"**Filtro IA:** `{st.session_state.ai_filter_query}`")

    if applied_filters_text:
         st.markdown("**Filtros Ativos:** " + " | ".join(applied_filters_text))
    else:
         st.info("Nenhum filtro ativo.")


    st.dataframe(st.session_state.df_explorar_display)

    # --- Download --- 
    @st.cache_data
    def convert_df_to_csv(df):
        return df.to_csv(index=False).encode('utf-8')

    csv_download = convert_df_to_csv(st.session_state.df_explorar_display)
    st.download_button(
        label="Baixar Dados Atuais como CSV",
        data=csv_download,
        file_name='met_data_explorada.csv',
        mime='text/csv',
    )


# --- Novas Páginas ---

def render_chatbot_expert(df):
    st.title("🤖 Chatbot Analista Especialista")
    st.markdown("Converse com a IA para obter análises e interpretações sobre os dados.")
    st.info("A IA tem acesso ao resumo dos seus dados (Pinturas/Esculturas). Peça análises, gráficos ou interpretações.")

    if df is None or df.empty:
        st.warning("Dados não carregados ou vazios. Carregue os dados na Visão Geral primeiro.")
        return

    # Inicializar estado do chat se necessário
    if "expert_chat_messages" not in st.session_state:
        st.session_state.expert_chat_messages = [{"role": "assistant", "content": "Olá! Como posso ajudar a analisar estes dados de arte hoje?"}]

    # Exibir histórico
    for message in st.session_state.expert_chat_messages:
        with st.chat_message(message["role"]):
            # Checar se é um dicionário com resultado (df ou fig)
            if isinstance(message["content"], dict):
                if "result_df" in message["content"]:
                    st.dataframe(message["content"]["result_df"])
                elif "fig" in message["content"]:
                    st.plotly_chart(message["content"]["fig"], use_container_width=True)
                elif "text" in message["content"]: # Texto interpretativo junto com output
                     st.markdown(message["content"]["text"])

            # Se for string (texto ou código)
            elif isinstance(message["content"], str):
                code_match = re.search(r"```python\n(.*)\n```", message["content"], re.DOTALL)
                text_after_code = re.sub(r"```python\n.*?\n```", "", message["content"], flags=re.DOTALL).strip()

                if code_match:
                    st.code(code_match.group(1).strip(), language="python")
                if text_after_code:
                    st.markdown(text_after_code)
                elif not code_match: # Se não for código nem tiver texto depois, mostrar conteúdo original
                     st.markdown(message["content"])


    # Input do usuário
    if prompt := st.chat_input("Sua pergunta ou pedido de análise..."):
        st.session_state.expert_chat_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Gerar resposta da IA e processar
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            with st.spinner("IA está analisando e respondendo..."):
                response_text = consultar_ia_dataframe_expert(prompt, df)

                code_to_execute = None
                interpretation_text = response_text
                output_content = {} # Para armazenar df/fig e texto

                code_match = re.search(r"```python\n(.*)\n```", response_text, re.DOTALL)
                if code_match:
                    code_to_execute = code_match.group(1).strip()
                    interpretation_text = re.sub(r"```python\n.*?\n```", "", response_text, flags=re.DOTALL).strip()
                    # Exibir código imediatamente
                    message_placeholder.code(code_to_execute, language="python")
                    output_content["code"] = code_to_execute # Guardar para histórico

                # Executar código se houver
                if code_to_execute:
                    try:
                        local_vars = {}
                        global_vars = {'pd': pd, 'px': px, 'go': go, 'np': np, 'stats': stats, 'df': df.copy()} # Passar cópia
                        safe_exec(code_to_execute, global_vars=global_vars, local_vars=local_vars)

                        if 'result_df' in local_vars:
                            result_df_output = local_vars['result_df']
                            st.dataframe(result_df_output)
                            output_content["result_df"] = result_df_output # Guardar para histórico
                        elif 'fig' in local_vars:
                            fig_output = local_vars['fig']
                            st.plotly_chart(fig_output, use_container_width=True)
                            output_content["fig"] = fig_output # Guardar para histórico

                        # Exibir interpretação após execução bem sucedida (se houver)
                        if interpretation_text:
                             st.markdown(interpretation_text)
                             output_content["text"] = interpretation_text # Guardar para histórico

                    except Exception as exec_e:
                        st.error(f"Erro ao executar o código da IA: {exec_e}")
                        interpretation_text = f"Erro ao executar código: {exec_e}" # Sobrescrever texto com erro
                        output_content["text"] = interpretation_text # Guardar erro no histórico

                else:
                    # Se não houve código, apenas exibir o texto
                    message_placeholder.markdown(interpretation_text)
                    output_content["text"] = interpretation_text # Guardar texto no histórico


                # Adicionar resposta completa (código, resultado, texto) ao histórico
                # Evitar adicionar mensagens vazias
                if output_content or interpretation_text:
                     # Se só tiver código no dict, usar o response_text original
                     if "code" in output_content and len(output_content) == 1 and not interpretation_text:
                          st.session_state.expert_chat_messages.append({"role": "assistant", "content": response_text})
                     # Se tiver resultado (df/fig) ou texto, usar o dict
                     elif "result_df" in output_content or "fig" in output_content or "text" in output_content:
                          st.session_state.expert_chat_messages.append({"role": "assistant", "content": output_content})
                     # Fallback para texto puro se nada mais foi capturado
                     elif interpretation_text:
                          st.session_state.expert_chat_messages.append({"role": "assistant", "content": interpretation_text})


def render_testes_estatisticos(df):
    st.title("📈 Testes Estatísticos (Beta)")
    st.warning("**Atenção:** Esta seção é experimental. A escolha e interpretação dos testes requer conhecimento estatístico.")
    st.markdown("Realize testes estatísticos básicos para explorar relações nos dados.")
    st.markdown("---")

    if df is None or df.empty:
        st.warning("Dados não carregados ou vazios.")
        return

    # Seleção de Variáveis
    st.subheader("1. Selecione as Variáveis")
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category', 'boolean']).columns.tolist()
    # Filtrar categóricas com poucos níveis para testes comuns
    cat_cols_test = [col for col in cat_cols if df[col].nunique() >= 2 and df[col].nunique() < 50]

    col1 = st.selectbox("Selecione a Coluna 1:", num_cols + cat_cols_test, key='test_col1')
    col2_options = ["Nenhuma (Teste de 1 Amostra)"] + [c for c in (num_cols + cat_cols_test) if c != col1]
    col2 = st.selectbox("Selecione a Coluna 2 (Opcional):", col2_options, key='test_col2')

    # Sugestão de Teste com IA
    st.subheader("2. Sugestão de Teste (IA)")
    if col1:
        col2_name = col2 if col2 != "Nenhuma (Teste de 1 Amostra)" else None
        if st.button("Sugerir Teste Estatístico com IA"):
            with st.spinner("IA sugerindo teste..."):
                 sugestao = consultar_ia_sugestao_teste(df, col1, col2_name)
                 st.markdown(sugestao)

    # Seleção Manual e Execução
    st.subheader("3. Selecione e Execute o Teste")
    test_options = ["Nenhum"]
    df_test = df.copy() # Trabalhar com cópia

    # Lógica para habilitar testes baseados nos tipos de colunas selecionadas
    type1 = df[col1].dtype if col1 else None
    type2 = df[col2].dtype if col2 and col2 != "Nenhuma (Teste de 1 Amostra)" else None

    is_col1_numeric = pd.api.types.is_numeric_dtype(type1)
    is_col1_categorical = pd.api.types.is_categorical_dtype(type1) or pd.api.types.is_object_dtype(type1) or pd.api.types.is_bool_dtype(type1)
    is_col2_numeric = pd.api.types.is_numeric_dtype(type2)
    is_col2_categorical = pd.api.types.is_categorical_dtype(type2) or pd.api.types.is_object_dtype(type2) or pd.api.types.is_bool_dtype(type2)

    col1_nunique = df[col1].nunique() if col1 and is_col1_categorical else 0
    col2_nunique = df[col2].nunique() if col2 and col2 != "Nenhuma (Teste de 1 Amostra)" and is_col2_categorical else 0


    # Teste t para 1 amostra (1 Numérica)
    if is_col1_numeric and col2 == "Nenhuma (Teste de 1 Amostra)":
        test_options.append("Teste t (1 Amostra)")

    # Teste t para amostras independentes (1 Numérica, 1 Categórica Binária)
    if is_col1_numeric and is_col2_categorical and col2_nunique == 2:
        test_options.append("Teste t (Amostras Independentes)")

    # ANOVA (1 Numérica, 1 Categórica > 2 níveis) - Requer mais validação
    # if is_col1_numeric and is_col2_categorical and col2_nunique > 2:
    #     test_options.append("ANOVA (One-Way)") # Adicionar com cuidado

    # Qui-Quadrado (2 Categóricas)
    if is_col1_categorical and is_col2_categorical:
        test_options.append("Qui-Quadrado (Associação)")


    selected_test = st.selectbox("Selecione o teste a realizar:", test_options)

    # Executar o teste selecionado
    if selected_test != "Nenhum":
        st.markdown(f"**Executando:** {selected_test}")
        alpha = st.slider("Nível de Significância (alpha):", 0.01, 0.10, 0.05, 0.01)

        try:
            if selected_test == "Teste t (1 Amostra)":
                popmean = st.number_input(f"Valor de referência (Média Populacional H0) para '{col1}':", value=df[col1].mean())
                data_test = df_test[col1].dropna()
                if len(data_test) < 3: raise ValueError("Amostra muito pequena para Teste t.")
                t_stat, p_value = stats.ttest_1samp(data_test, popmean=popmean)
                st.write(f"**Resultado do Teste t (1 Amostra) para '{col1}' contra {popmean}:**")
                st.write(f"- Estatística t: {t_stat:.4f}")
                st.write(f"- Valor-p: {p_value:.4f}")
                if p_value < alpha:
                    st.success(f"Resultado significativo (p < {alpha}): Rejeitamos H0. A média de '{col1}' é significativamente diferente de {popmean}.")
                else:
                    st.info(f"Resultado não significativo (p >= {alpha}): Não podemos rejeitar H0. Não há evidência suficiente para dizer que a média é diferente de {popmean}.")

            elif selected_test == "Teste t (Amostras Independentes)":
                groups = df_test[col2].unique()
                group1_data = df_test[df_test[col2] == groups[0]][col1].dropna()
                group2_data = df_test[df_test[col2] == groups[1]][col1].dropna()
                if len(group1_data) < 3 or len(group2_data) < 3: raise ValueError("Um ou ambos os grupos têm amostra muito pequena para Teste t.")
                # Checar variâncias (opcional, ttest_ind lida com equal_var=False)
                # levene_stat, levene_p = stats.levene(group1_data, group2_data)
                # equal_var = levene_p >= alpha
                t_stat, p_value = stats.ttest_ind(group1_data, group2_data, equal_var=False) # Welch's t-test por padrão
                st.write(f"**Resultado do Teste t (Independente) para '{col1}' entre os grupos de '{col2}':**")
                st.write(f"- Grupo 1 ('{groups[0]}'): Média={group1_data.mean():.2f}, N={len(group1_data)}")
                st.write(f"- Grupo 2 ('{groups[1]}'): Média={group2_data.mean():.2f}, N={len(group2_data)}")
                st.write(f"- Estatística t: {t_stat:.4f}")
                st.write(f"- Valor-p: {p_value:.4f}")
                if p_value < alpha:
                    st.success(f"Resultado significativo (p < {alpha}): Rejeitamos H0. Existe uma diferença significativa na média de '{col1}' entre os grupos '{groups[0]}' e '{groups[1]}'.")
                else:
                    st.info(f"Resultado não significativo (p >= {alpha}): Não podemos rejeitar H0. Não há evidência suficiente de diferença nas médias.")

            elif selected_test == "Qui-Quadrado (Associação)":
                contingency = pd.crosstab(df_test[col1], df_test[col2])
                if contingency.empty or contingency.shape[0] < 2 or contingency.shape[1] < 2:
                     raise ValueError("Tabela de contingência inválida ou com poucas categorias para o teste.")
                chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
                st.write(f"**Resultado do Teste Qui-Quadrado para Associação entre '{col1}' e '{col2}':**")
                with st.expander("Ver Tabela de Contingência Observada"):
                    st.dataframe(contingency)
                with st.expander("Ver Frequências Esperadas (H0)"):
                    st.dataframe(pd.DataFrame(expected, index=contingency.index, columns=contingency.columns))
                st.write(f"- Estatística Qui-Quadrado (χ²): {chi2:.4f}")
                st.write(f"- Graus de Liberdade (dof): {dof}")
                st.write(f"- Valor-p: {p_value:.4f}")
                if p_value < alpha:
                    st.success(f"Resultado significativo (p < {alpha}): Rejeitamos H0. Existe uma associação estatisticamente significativa entre '{col1}' e '{col2}'.")
                else:
                    st.info(f"Resultado não significativo (p >= {alpha}): Não podemos rejeitar H0. Não há evidência suficiente de associação entre as variáveis.")
                # Aviso sobre frequências esperadas baixas
                if (expected < 5).any().any():
                    st.warning("Aviso: Algumas frequências esperadas são menores que 5. O resultado do Qui-Quadrado pode ser menos confiável.")


        except ValueError as ve:
             st.error(f"Erro ao preparar dados para o teste: {ve}")
        except Exception as e:
            st.error(f"Erro ao executar o teste '{selected_test}': {e}")


def render_analise_geoespacial(df):
    st.title("🌍 Análise Geoespacial (Experimental)")
    st.markdown("Visualize a distribuição geográfica das obras.")
    st.warning("Funcionalidade experimental. A qualidade depende dos dados de 'Country' ou 'City'.")
    st.markdown("---")

    if df is None or df.empty:
        st.warning("Dados não carregados ou vazios.")
        return

    geo_col = None
    if 'Country' in df.columns and df['Country'].nunique() > 1 and df['Country'].nunique() < 200: # Coluna 'Country' parece mais promissora
         geo_col = 'Country'
    elif 'Region' in df.columns and df['Region'].nunique() > 1 and df['Region'].nunique() < 100:
         geo_col = 'Region'
    elif 'City' in df.columns and df['City'].nunique() > 1 and df['City'].nunique() < 500: # Cidade pode ser demais
         geo_col = 'City'


    if not geo_col:
        st.info("Não foi encontrada uma coluna geográfica adequada ('Country', 'Region', 'City') com dados suficientes para mapeamento.")
        return

    st.subheader(f"Distribuição de Obras por '{geo_col}'")
    geo_counts = df[df[geo_col] != 'Desconhecido'][geo_col].value_counts().reset_index()
    geo_counts.columns = [geo_col, 'Contagem']

    # Tentar criar um mapa de coroplético (funciona melhor para países/regiões)
    if geo_col in ['Country', 'Region']:
        try:
            fig = px.choropleth(geo_counts,
                                locations=geo_col,
                                locationmode='country names' if geo_col == 'Country' else None, # Tenta usar nomes de países
                                color='Contagem',
                                hover_name=geo_col,
                                color_continuous_scale=px.colors.sequential.Plasma,
                                title=f"Distribuição Geográfica das Obras por {geo_col}")
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Erro ao gerar mapa coroplético: {e}. Tentando gráfico de barras.")
            # Fallback para gráfico de barras se o mapa falhar
            top_n_geo = min(30, len(geo_counts))
            fig_bar = px.bar(geo_counts.head(top_n_geo), x=geo_col, y='Contagem',
                           title=f"Top {top_n_geo} Localizações ({geo_col}) por Número de Obras")
            st.plotly_chart(fig_bar, use_container_width=True)

    else: # Para 'City', mapa de pontos seria melhor, mas barras são mais simples
        st.info("Mapeamento de cidades não implementado, exibindo gráfico de barras.")
        top_n_geo = min(50, len(geo_counts))
        fig_bar = px.bar(geo_counts.head(top_n_geo), x=geo_col, y='Contagem',
                       title=f"Top {top_n_geo} Localizações ({geo_col}) por Número de Obras")
        st.plotly_chart(fig_bar, use_container_width=True)

    with st.expander(f"Ver contagem por {geo_col}"):
        st.dataframe(geo_counts)


# --- Aplicação Principal ---
def run_app():
    st.sidebar.title("Met Art Analyzer ✨PRO✨")
    st.sidebar.markdown("Análise Aprofundada de **Pinturas e Esculturas**.")
    st.sidebar.markdown(f"Fonte: `{CSV_PATH}`")
    st.sidebar.markdown("---")

    # Carregar dados
    data = load_and_filter_data(CSV_PATH, FILTER_COLUMN, FILTER_VALUES)

    if data is None:
        st.sidebar.error("Falha ao carregar dados. O aplicativo não pode continuar.")
        # Adicionado st.stop() para interromper completamente
        st.stop()

    st.sidebar.success(f"{len(data):,} obras carregadas.")
    st.sidebar.markdown("---")

    # Menu de Navegação - Adicionando Novas Páginas
    page = st.sidebar.radio(
        "Escolha uma seção de análise:",
        ["🏠 Visão Geral & Insights",
         "📊 Análise Exploratória",
         "🎲 Probabilidades",
         "🎨 Artistas & Culturas",
         "📅 Análise Temporal",
         "🌍 Análise Geoespacial (Beta)", # Nova
         "📈 Testes Estatísticos (Beta)", # Nova
         "🤖 Chatbot Especialista",       # Nova
         "💾 Explorar Dados"]
    )
    st.sidebar.markdown("---")
    st.sidebar.info("Seções marcadas com (Beta) são experimentais.")

    # Renderizar a página selecionada
    if page == "🏠 Visão Geral & Insights":
        render_visao_geral(data)
    elif page == "📊 Análise Exploratória":
        render_analise_exploratoria(data)
    elif page == "🎲 Probabilidades":
        render_probabilidades(data)
    elif page == "🎨 Artistas & Culturas":
        render_artistas_culturas(data)
    elif page == "📅 Análise Temporal":
        render_analise_temporal(data)
    elif page == "🌍 Análise Geoespacial (Beta)": # Nova
        render_analise_geoespacial(data)
    elif page == "📈 Testes Estatísticos (Beta)": # Nova
        render_testes_estatisticos(data)
    elif page == "🤖 Chatbot Especialista":       # Nova
        render_chatbot_expert(data)
    elif page == "💾 Explorar Dados":
        render_explorar_dados(data)

if __name__ == "__main__":
    run_app()