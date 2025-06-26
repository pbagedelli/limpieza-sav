import streamlit as st
import openai
import json
import pandas as pd
import numpy as np # Para dtypes numéricos
import re # Para un fallback si el LLM falla
import os
import io # Para manejar bytes en memoria para la descarga
import pyreadstat # Para guardar archivos .sav
import tempfile 

# --- Configuración de la Página de Streamlit ---
st.set_page_config(layout="wide", page_title="Procesador de Datos para SPSS", initial_sidebar_state="expanded")

st.title("Procesador de Datos de Encuestas con IA para SPSS (.sav)")
st.markdown("""
Sube un archivo CSV o Excel, elige las operaciones y descarga los resultados en formato `.sav`.
- **Simplificar Nombres de Columnas**: Usa un LLM para acortar los nombres largos de las columnas (serán los nombres de variable en SPSS).
- **Generar Etiquetas de Variables**: Usa un LLM para crear etiquetas descriptivas para cada variable (serán las etiquetas de variable en SPSS).
- **Codificar Variables Ordinales**: Identifica columnas **NO NUMÉRICAS** con categorías ordinales (ej. escalas Likert) y las convierte a números, usando los textos originales como etiquetas de valor en SPSS.
- **Manejo de Missing para Strings**: Las variables de texto tendrán 'nan' (como string) definido como valor perdido en SPSS.
""")

# --- Constantes ---
MAX_CATEGORIAS_PARA_LLM = 10 # Para la codificación ordinal
SPSS_VAR_NAME_MAX_LEN = 64 # Límite de longitud para nombres de variable en SPSS
# MODELO_LLM = "gpt-4-turbo-preview" # Modelo por defecto anterior
MODELO_LLM_PRINCIPAL = "gpt-4.1-mini" # Modelo especificado por el usuario

# --- NUEVA FUNCIÓN DE SANITIZACIÓN ---
def sanitize_spss_varname(name_str):
    name_str = str(name_str)
    name_str = re.sub(r'[.:\-/]', '_', name_str) 
    name_str = re.sub(r'\s+', '_', name_str)     
    name_str = re.sub(r'[^\w_]', '', name_str)  
    if not name_str or not name_str[0].isalpha():
        name_str = "V_" + name_str 
    name_str = name_str[:SPSS_VAR_NAME_MAX_LEN]
    if not name_str: 
        return "UnnamedVar"
    return name_str

# --- Funciones LLM (Originales y Nuevas) ---
def simplify_survey_column_names_llm(column_names_list, client, model=MODELO_LLM_PRINCIPAL):
    if not client:
        st.error("Error: El cliente de OpenAI no está inicializado.")
        return None
    if not column_names_list:
        st.warning("La lista de nombres de columnas está vacía.")
        return {}
    column_names_for_prompt = "\n".join([f"- \"{name}\"" for name in column_names_list])
    prompt = f"""
Eres un asistente experto en preparar datos de encuestas para análisis estadístico, especialmente para SPSS.
Te proporcionaré una lista de nombres de columnas que a menudo son preguntas completas de una encuesta.
Tu tarea es, para CADA nombre de columna original, generar un NOMBRE DE VARIABLE CORTO, CLARO y VÁLIDO para SPSS.

Consideraciones para los nuevos NOMBRES DE VARIABLE:
1.  Deben ser muy concisos (idealmente 1-3 palabras, máximo {SPSS_VAR_NAME_MAX_LEN} caracteres según SPSS, pero apunta a mucho menos).
2.  Deben ser descriptivos del contenido o la pregunta de la columna.
3.  Utiliza PascalCase o snake_case (ej. NivelEducacion o nivel_educacion). Evita espacios y caracteres especiales. Comienza con una letra.
4.  Si la pregunta original es muy larga, enfócate en el concepto clave.
5.  Mantiene el mismo idioma que el original.
6.  No agregue conceptos o entidades que no este en la pregunta original

Devuelve tu respuesta ÚNICAMENTE como un objeto JSON que mapee cada nombre de columna ORIGINAL a su nuevo NOMBRE DE VARIABLE SIMPLIFICADO.
Asegúrate de que CADA nombre de columna original de la lista de entrada tenga su correspondiente nombre de variable simplificado en el JSON de salida.

Aquí tienes algunos ejemplos del tipo de transformación deseada:
- Original: "Corporations and the wealthy have too much influence over government in this country.:Please say whether you AGREE or DISAGREE with the following statements."
  Simplificado: "CorpInfluence"
- Original: "Some people vote in elections, while others choose NOT to vote. In the current CANADIAN FEDERAL election, how likely are you to vote?  "
  Simplificado: "LikelyToVote"
- Original: "Plear enter your email below if you would like to participate for a chance to win cash! (optional)  "
  Simplificado: "EmailOptIn"
- Original: "Overall, how satisfied or dissatisfied are you with our customer service?"
  Simplificado: "CustServSat"
- Original: "What is your highest level of education completed?"
  Simplificado: "EducationLevel"

Lista de nombres de columnas originales a simplificar:
{column_names_for_prompt}

Formato JSON de salida esperado:
{{
  "nombre_columna_original_1": "NombreVariable1",
  "nombre_columna_original_2": "NombreVariable2",
  ...
}}
"""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": f"Eres un experto en crear nombres de variable cortos y válidos (máx {SPSS_VAR_NAME_MAX_LEN} caracteres) para SPSS a partir de preguntas de encuestas. Devuelves solo JSON."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )
        llm_response_json = response.choices[0].message.content
    except Exception as e:
        st.error(f"Error al llamar a la API de OpenAI para simplificar nombres: {e}")
        return None
    try:
        parsed_response = json.loads(llm_response_json)
        if not isinstance(parsed_response, dict):
            st.error(f"Error: La respuesta del LLM para simplificación no es un diccionario: {parsed_response}")
            return None
        validated_response = {}
        for original, simplified in parsed_response.items():
            # Usar sanitize_spss_varname para una limpieza más robusta
            validated_response[original] = sanitize_spss_varname(simplified)
        
        missing_keys = [name for name in column_names_list if name not in validated_response]
        if missing_keys:
            st.warning(f"Advertencia: El LLM no devolvió nombres para las siguientes columnas originales: {missing_keys}")
            for key in missing_keys:
                validated_response[key] = basic_column_simplifier(key) 
        return validated_response
    except json.JSONDecodeError as e:
        st.error(f"Error al decodificar JSON de la respuesta del LLM (simplificación): {e}")
        st.text_area("Respuesta recibida del LLM (simplificación):", llm_response_json, height=150)
        return None

def basic_column_simplifier(column_name, max_words=3): 
    name = re.sub(r':Please say whether you AGREE or DISAGREE with the following statements\.', '', column_name, flags=re.IGNORECASE)
    name = re.sub(r'Please tell us.*', '', name, flags=re.IGNORECASE)
    name = re.sub(r'\(optional\)', '', name, flags=re.IGNORECASE)
    name = re.sub(r'[¿?.:!]', '', name)
    words = name.strip().split()
    simplified = ''.join([word.capitalize() for word in words[:max_words]]) if words else column_name
    return sanitize_spss_varname(simplified)


def generate_variable_labels_llm(column_name_map, client, model=MODELO_LLM_PRINCIPAL):
    if not client:
        st.error("Error: El cliente de OpenAI no está inicializado.")
        return None
    if not column_name_map:
        st.warning("La lista/mapa de nombres de columnas para generar etiquetas está vacía.")
        return {}
    if isinstance(column_name_map, list):
        column_name_map_for_prompt = {name: name for name in column_name_map}
    else:
        column_name_map_for_prompt = column_name_map
    items_for_prompt = "\n".join([f"- Nombre de Variable: \"{spss_name}\", Descripción/Pregunta Original: \"{original_desc}\"" for spss_name, original_desc in column_name_map_for_prompt.items()])
    prompt = f"""
Eres un asistente experto en preparar datos de encuestas para análisis estadístico en SPSS.
Te proporcionaré una lista de NOMBRES DE VARIABLE (cortos, para SPSS) junto con su pregunta original o descripción.
Tu tarea es, para CADA nombre de variable, generar una ETIQUETA DE VARIABLE descriptiva y clara para SPSS.

Consideraciones para las ETIQUETAS DE VARIABLE:
1.  Deben ser cortas, descriptivas y explicar lo que la variable representa.
2.  Deben estar en el mismo idioma que la pregunta original.
3.  Si la pregunta original es muy larga, resúmela sin perder el significado esencial.
4.  La longitud máxima de una etiqueta de variable en SPSS es 256 caracteres. Intenta no superarla.


Devuelve tu respuesta ÚNICAMENTE como un objeto JSON que mapee cada NOMBRE DE VARIABLE (el corto que te di) a su nueva ETIQUETA DE VARIABLE (la descriptiva).
Asegúrate de que CADA nombre de variable de la lista de entrada tenga su correspondiente etiqueta en el JSON de salida.

Ejemplos:
- Nombre de Variable: "CorpInfluence", Pregunta Original: "Corporations and the wealthy have too much influence over government in this country.:Please say whether you AGREE or DISAGREE with the following statements."
  Etiqueta de Variable: "Influence of Corporations and Wealthy on Government"
- Nombre de Variable: "LikelyToVote", Pregunta Original: "Some people vote in elections, while others choose NOT to vote. In the current CANADIAN FEDERAL election, how likely are you to vote?"
  Etiqueta de Variable: "Likelihood of Voting in Current Canadian Federal Election"
- Nombre de Variable: "EducationLevel", Pregunta Original: "What is your highest level of education completed?"
  Etiqueta de Variable: "Education Level"
- Nombre de Variable: "ImagenDinaBoluarte", Pregunta Original: "Imagen de Diana Boluarte"
  Etiqueta de Variable: "Imagen Diana Boluarte"
- Nombre de Variable: "ImagenPresidente", Pregunta Original: "imagen_presidente"
  Etiqueta de Variable: "Imagen Presidente"

Lista de nombres de variable y sus descripciones originales:
{items_for_prompt}

Formato JSON de salida esperado:
{{
  "NombreVariable1": "Etiqueta Descriptiva para Variable 1",
  "NombreVariable2": "Etiqueta Descriptiva para Variable 2",
  ...
}}
"""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Eres un experto en crear etiquetas de variable descriptivas para SPSS (max 256 caracteres). Devuelves solo JSON."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )
        llm_response_json = response.choices[0].message.content
    except Exception as e:
        st.error(f"Error al llamar a la API de OpenAI para generar etiquetas de variable: {e}")
        return None
    try:
        parsed_response = json.loads(llm_response_json)
        if not isinstance(parsed_response, dict):
            st.error(f"Error: La respuesta del LLM para etiquetas de variable no es un diccionario: {parsed_response}")
            return None
        final_labels = {key: str(value)[:256] for key, value in parsed_response.items()} # Truncar a 256
        missing_keys = [name for name in column_name_map_for_prompt.keys() if name not in final_labels]
        if missing_keys:
            st.warning(f"Advertencia: El LLM no devolvió etiquetas para las siguientes variables: {missing_keys}")
            for key in missing_keys:
                final_labels[key] = str(column_name_map_for_prompt.get(key, key))[:256] # Usar la original o la clave como fallback
        return final_labels
    except json.JSONDecodeError as e:
        st.error(f"Error al decodificar JSON de la respuesta del LLM (etiquetas de variable): {e}")
        st.text_area("Respuesta recibida del LLM (etiquetas de variable):", llm_response_json, height=150)
        return None

def get_llm_mapping_suggestion(categories_list, client, model=MODELO_LLM_PRINCIPAL): # Cambio de modelo aquí
    if not client:
        st.error("Error: El cliente de OpenAI no está inicializado. No se puede hacer la llamada API para codificación.")
        return None
    categories_list_string = json.dumps(categories_list)
    prompt = f"""
Eres un asistente experto en análisis de datos y encuestas, preparando datos para SPSS.
Te proporcionaré una lista de Python que contiene las categorías únicas (ya ordenadas alfabéticamente/numéricamente si es posible) de una columna de una encuesta.
Tu tarea es:
1. Determinar si estas categorías representan una escala ordinal significativa.
2. Si es ordinal, crea un diccionario de Python que mapee cada categoría de texto a un valor numérico entero. El mapeo debe respetar el orden lógico de la escala. Los valores numéricos deben ser enteros y preferiblemente comenzar desde 1 (el valor más positivo de la escala).
3. Si es ordinal pero tiene alguna categoría que no se puede ordenar o representa un "no sabe/no contesta" (ej. "No sé", "Prefiero no responder", "N/A", "nan"), mapea las categorías ordenables y asigna un valor numérico alto (ej. 99, 98) a estas categorías no ordenables/missing. Estos se usarán luego como valores perdidos definidos por el usuario en SPSS. Si no hay tales categorías, no incluyas valores altos.
4. Devuelve tu respuesta ÚNICAMENTE como un objeto JSON con la estructura:
   {{
     "is_ordinal": true_or_false,
     "mapping_dict": {{ "categoria_texto_1": numero_1, "categoria_texto_2": numero_2, "No sé": 99, ... }}
   }}
   Si `is_ordinal` es false, `mapping_dict` debe ser `null`.
Usa las categorías de texto exactas proporcionadas como claves en `mapping_dict`. Asegúrate que los valores en el diccionario sean NUMÉRICOS, no strings.

Ejemplos de mapeo:
- ["Totalmente en desacuerdo", "En desacuerdo", "Neutral", "De acuerdo", "Totalmente de acuerdo"] -> {{"is_ordinal": true, "mapping_dict": {{"Totalmente de acuerdo": 1, "De acuerdo": 2, "Neutral": 3, "En desacuerdo": 4, "Totalmente en desacuerdo": 5}}}}
- ["Muy Malo", "Malo", "Regular", "Bueno", "Excelente", "No aplica"] -> {{"is_ordinal": true, "mapping_dict": {{"Excelente": 1, "Bueno": 2, "Regular": 3, "Malo": 4, "Muy Malo": 5, "No aplica": 99}}}}
- ["Bajo", "Medio", "Alto"] -> {{"is_ordinal": true, "mapping_dict": {{"Alto": 1, "Medio": 2, "Bajo": 3}}}}
- ["Sí", "No", "Quizás"] -> {{"is_ordinal": true, "mapping_dict": {{"Sí": 1, "Quizás": 2, "No": 3}}}} (orden lógico, no alfabético) o bien {{"Sí": 1, "No": 0, "Quizás": 2}} si es más intuitivo para sí/no. Sé consistente.
- ["Manzana", "Banana", "Cereza"] -> {{"is_ordinal": false, "mapping_dict": null}} (nominal)
- ["1", "2", "3", "4", "5", "nan"] -> {{"is_ordinal": true, "mapping_dict": {{"1":1, "2":2, "3":3, "4":4, "5":5, "nan":99}}}} (si el contexto sugiere ordinalidad y 'nan' es un missing)

Lista de categorías:
{categories_list_string}
"""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Eres un asistente experto en análisis de datos y encuestas, especializado en devolver respuestas en formato JSON para escalas ordinales. Asegúrate que los valores del mapping_dict sean números, no strings."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )
        llm_response_json = response.choices[0].message.content
    except Exception as e:
        st.error(f"Error al llamar a la API de OpenAI para codificación: {e}")
        return None
    try:
        parsed_response = json.loads(llm_response_json)
        # Validar que los valores del mapping_dict sean numéricos si existe
        if parsed_response and isinstance(parsed_response.get("mapping_dict"), dict):
            for k, v in parsed_response["mapping_dict"].items():
                if not isinstance(v, (int, float)):
                    st.warning(f"LLM devolvió un valor no numérico ('{v}') para la categoría '{k}' en el mapeo. Intentando convertir a int o marcando como error si falla.")
                    try:
                        parsed_response["mapping_dict"][k] = int(v)
                    except ValueError:
                        st.error(f"No se pudo convertir '{v}' a int para la categoría '{k}'. Este mapeo podría ser inválido.")
                        # Podríamos eliminar esta entrada o invalidar todo el mapeo
                        # Por ahora, se deja para que el usuario lo vea y la lógica posterior lo maneje
        return parsed_response
    except json.JSONDecodeError as e:
        st.error(f"Error al decodificar JSON de la respuesta del LLM (codificación): {e}")
        st.text_area("Respuesta recibida del LLM (codificación):", llm_response_json, height=150)
        return None

# --- Interfaz de Streamlit ---
st.sidebar.header("🔑 Configuración de OpenAI")
api_key_input = st.sidebar.text_input("Ingresa tu OpenAI API Key", type="password", help="Tu API key no se almacena.")
openai_client = None
if api_key_input:
    try:
        openai_client = openai.OpenAI(api_key=api_key_input)
        st.sidebar.success("Cliente de OpenAI inicializado.")
    except Exception as e:
        st.sidebar.error(f"Error al inicializar OpenAI: {e}")
        openai_client = None
else:
    st.sidebar.info("Ingresa una API Key de OpenAI para usar las funciones basadas en LLM.")

st.sidebar.header("📂 Cargar Archivo")
uploaded_file = st.sidebar.file_uploader("Sube tu archivo CSV o Excel", type=["csv", "xlsx"])

st.sidebar.header("⚙️ Opciones de Procesamiento")
do_simplify_cols = st.sidebar.checkbox("Simplificar nombres de columnas (Nombres de Variable SPSS)", value=False)
do_generate_var_labels = st.sidebar.checkbox("Generar etiquetas de variables (Etiquetas de Variable SPSS)", value=True)
do_encode_ordinal = st.sidebar.checkbox("Codificar variables ordinales (con Etiquetas de Valor SPSS)", value=True)

ordinal_encoding_mode = "Crear nuevas columnas (ej. VarName_num)"
if do_encode_ordinal:
    ordinal_encoding_mode = st.sidebar.radio(
        "Modo de codificación ordinal:",
        ("Crear nuevas columnas (ej. VarName_num)", "Reemplazar valores en columnas existentes"),
        index=0,
        help="Elige si las columnas codificadas reemplazan a las originales o se crean como nuevas."
    )

if 'df_processed' not in st.session_state: st.session_state.df_processed = None
if 'spss_variable_labels' not in st.session_state: st.session_state.spss_variable_labels = {}
if 'spss_value_labels' not in st.session_state: st.session_state.spss_value_labels = {}
if 'spss_missing_ranges' not in st.session_state: st.session_state.spss_missing_ranges = {}
if 'codificaciones_ordinales_cache' not in st.session_state: st.session_state.codificaciones_ordinales_cache = {}
if 'log_messages' not in st.session_state: st.session_state.log_messages = []


if uploaded_file is not None:
    st.subheader("Vista Previa del Archivo Original")
    try:
        # MODIFICACIÓN 1: Cargar sin dtype=str para preservar tipos originales
        if uploaded_file.name.endswith('.csv'):
            df_original = pd.read_csv(uploaded_file)
        else:
            df_original = pd.read_excel(uploaded_file)
        st.dataframe(df_original.head())

        if st.sidebar.button("🚀 Procesar Datos para .sav"):
            with st.spinner("Procesando datos... Por favor espera."):
                df_processed = df_original.copy()
                st.session_state.log_messages = [] # Resetear logs
                
                spss_variable_labels_dict = {} 
                spss_value_labels_dict = {}  
                original_to_simplified_map_for_labels = {col: col for col in df_processed.columns}

                # 1. Simplificar Nombres de Columnas (Nombres de Variable SPSS) - Opcional
                if do_simplify_cols:
                    st.session_state.log_messages.append("--- Iniciando Simplificación de Nombres de Columnas ---")
                    if openai_client:
                        original_column_names_list = df_processed.columns.tolist()
                        with st.spinner("Simplificando nombres de columnas (Nombres de Variable SPSS) con LLM..."):
                            simplified_names_map_llm = simplify_survey_column_names_llm(original_column_names_list, client=openai_client)

                        if simplified_names_map_llm:
                            st.session_state.log_messages.append("Mapa de nombres de variable simplificados (LLM) obtenido.")
                            final_rename_map_llm = {}
                            seen_llm_names = set()
                            for original_name in original_column_names_list:
                                proposed_name = simplified_names_map_llm.get(original_name, basic_column_simplifier(original_name))
                                unique_name = proposed_name
                                count = 1
                                while unique_name in seen_llm_names:
                                    unique_name = f"{proposed_name[:SPSS_VAR_NAME_MAX_LEN - (len(str(count))+1)]}_{count}"
                                    count += 1
                                seen_llm_names.add(unique_name)
                                final_rename_map_llm[original_name] = unique_name
                            
                            df_processed.rename(columns=final_rename_map_llm, inplace=True)
                            original_to_simplified_map_for_labels = {orig: final_rename_map_llm.get(orig, orig) for orig in original_column_names_list}
                            st.session_state.log_messages.append("Nombres de columna (variables SPSS) simplificados y aplicados.")
                        else: 
                            st.session_state.log_messages.append("No se pudieron simplificar los nombres con LLM. Usando fallback básico para todos.")
                            fb_rename_map = {name: basic_column_simplifier(name) for name in df_processed.columns}
                            unique_fb_map = {}
                            seen_fb_names = set()
                            for orig, simpl in fb_rename_map.items():
                                uname = simpl
                                c = 1
                                while uname in seen_fb_names:
                                    uname = f"{simpl[:SPSS_VAR_NAME_MAX_LEN-(len(str(c))+1)]}_{c}"
                                    c+=1
                                seen_fb_names.add(uname)
                                unique_fb_map[orig] = uname
                            df_processed.rename(columns=unique_fb_map, inplace=True)
                            original_to_simplified_map_for_labels = {orig: unique_fb_map.get(orig, orig) for orig in df_original.columns}
                    else: 
                        st.session_state.log_messages.append("Cliente OpenAI no configurado. Aplicando simplificación básica de renombrado.")
                        fb_rename_map = {name: basic_column_simplifier(name) for name in df_processed.columns}
                        unique_fb_map = {}
                        seen_fb_names = set()
                        for orig, simpl in fb_rename_map.items():
                            uname = simpl
                            c = 1
                            while uname in seen_fb_names:
                                uname = f"{simpl[:SPSS_VAR_NAME_MAX_LEN-(len(str(c))+1)]}_{c}"
                                c+=1
                            seen_fb_names.add(uname)
                            unique_fb_map[orig] = uname
                        df_processed.rename(columns=unique_fb_map, inplace=True)
                        original_to_simplified_map_for_labels = {orig: unique_fb_map.get(orig, orig) for orig in df_original.columns}
                    st.session_state.log_messages.append("--- Fin de Simplificación de Nombres ---")
                

                if do_generate_var_labels:
                    st.session_state.log_messages.append("\n--- Iniciando Generación de Etiquetas de Variable ---")
                    if openai_client:
                        map_current_name_to_original_question = {}
                        for original_q_name, current_df_proc_name in original_to_simplified_map_for_labels.items():
                            if current_df_proc_name in df_processed.columns:
                                map_current_name_to_original_question[current_df_proc_name] = original_q_name
                            else: 
                                if not do_simplify_cols and original_q_name in df_processed.columns:
                                     map_current_name_to_original_question[original_q_name] = original_q_name
                        if not map_current_name_to_original_question: 
                            map_current_name_to_original_question = {col: col for col in df_processed.columns}
                        
                        with st.spinner("Generando etiquetas de variable (Etiquetas SPSS) con LLM..."):
                            generated_labels = generate_variable_labels_llm(map_current_name_to_original_question, client=openai_client)
                        
                        if generated_labels:
                            spss_variable_labels_dict = generated_labels 
                            st.session_state.log_messages.append("Etiquetas de variable generadas por LLM.")
                        else:
                            st.session_state.log_messages.append("No se pudieron generar etiquetas de variable con LLM. Usando pregunta original/nombre de columna como fallback.")
                            spss_variable_labels_dict = {col: str(map_current_name_to_original_question.get(col, col))[:256] for col in df_processed.columns} 
                    else: 
                        st.session_state.log_messages.append("Cliente OpenAI no configurado para etiquetas de variable. Usando pregunta original/nombre de columna como fallback.")
                        # Construir map_current_name_to_original_question incluso si no hay cliente, para que las etiquetas sean las originales si es posible
                        map_current_name_to_original_question_fallback = {}
                        for original_q_name, current_df_proc_name in original_to_simplified_map_for_labels.items():
                            if current_df_proc_name in df_processed.columns:
                                map_current_name_to_original_question_fallback[current_df_proc_name] = original_q_name
                            elif not do_simplify_cols and original_q_name in df_processed.columns:
                                map_current_name_to_original_question_fallback[original_q_name] = original_q_name
                        if not map_current_name_to_original_question_fallback:
                             map_current_name_to_original_question_fallback = {col: col for col in df_processed.columns}

                        spss_variable_labels_dict = {col: str(map_current_name_to_original_question_fallback.get(col,col))[:256] for col in df_processed.columns}
                    st.session_state.log_messages.append("--- Fin de Generación de Etiquetas de Variable ---")
                else: 
                    st.session_state.log_messages.append("\nGeneración de etiquetas de variable omitida por el usuario.")
                    for current_col_name_in_df_proc in df_processed.columns:
                        original_question = current_col_name_in_df_proc 
                        for orig_q, simpl_name in original_to_simplified_map_for_labels.items():
                            if simpl_name == current_col_name_in_df_proc:
                                original_question = orig_q
                                break
                        spss_variable_labels_dict[current_col_name_in_df_proc] = str(original_question)[:256]


                if do_encode_ordinal:
                    st.session_state.log_messages.append("\n--- Iniciando Codificación de Variables Ordinales ---")
                    if openai_client:
                        columnas_a_evaluar_para_ordinal = list(df_processed.columns) 
                        cols_to_encode_spinner = st.empty()

                        for i, col_actual_en_df_proc in enumerate(columnas_a_evaluar_para_ordinal):
                            cols_to_encode_spinner.info(f"Evaluando columna para codificación ordinal: '{col_actual_en_df_proc}' ({i+1}/{len(columnas_a_evaluar_para_ordinal)})")
                            st.session_state.log_messages.append(f"\nProcesando columna para codificación: '{col_actual_en_df_proc}'")

                            nombre_col_df_original = None
                            for orig_name, current_name in original_to_simplified_map_for_labels.items():
                                if current_name == col_actual_en_df_proc:
                                    nombre_col_df_original = orig_name
                                    break
                            if not nombre_col_df_original and col_actual_en_df_proc in df_original.columns: 
                                nombre_col_df_original = col_actual_en_df_proc
                            
                            if not nombre_col_df_original or nombre_col_df_original not in df_original.columns:
                                st.session_state.log_messages.append(f"  Advertencia: No se pudo encontrar la columna original para '{col_actual_en_df_proc}'. Omitiendo.")
                                continue
                            
                            # MODIFICACIÓN 1: Comprobación de tipo de datos numéricos
                            if pd.api.types.is_numeric_dtype(df_original[nombre_col_df_original].dtype):
                                st.session_state.log_messages.append(f"  Omitiendo '{col_actual_en_df_proc}' (original: '{nombre_col_df_original}'): Es de tipo numérico ({df_original[nombre_col_df_original].dtype}).")
                                continue
                            
                            try:
                                # Convertir a string explícitamente para obtener categorías
                                categorias_unicas_series = df_original[nombre_col_df_original].dropna().astype(str)
                                # Reemplazar strings vacíos con 'nan' antes de unique() para que LLM lo vea
                                categorias_unicas_series = categorias_unicas_series.replace('', 'nan')
                                categorias_unicas = sorted(list(categorias_unicas_series.unique()))
                            except Exception as e:
                                st.session_state.log_messages.append(f"  Error al obtener categorías únicas de '{nombre_col_df_original}' para '{col_actual_en_df_proc}': {e}. Omitiendo.")
                                continue
                            
                            if not categorias_unicas or not (1 < len(categorias_unicas) <= MAX_CATEGORIAS_PARA_LLM):
                                if not categorias_unicas: st.session_state.log_messages.append(f"  Omitiendo '{col_actual_en_df_proc}': Sin categorías procesables.")
                                else: st.session_state.log_messages.append(f"  Omitiendo '{col_actual_en_df_proc}': {len(categorias_unicas)} categorías (límite 1-{MAX_CATEGORIAS_PARA_LLM}).")
                                continue

                            clave_cache = tuple(categorias_unicas) # Usar tuple para que sea hasheable
                            sugerencia = st.session_state.codificaciones_ordinales_cache.get(clave_cache)
                            if not sugerencia:
                                st.session_state.log_messages.append(f"  Consultando LLM para categorías de '{col_actual_en_df_proc}': {clave_cache}.")
                                with st.spinner(f"Consultando LLM para '{col_actual_en_df_proc}'..."):
                                     sugerencia = get_llm_mapping_suggestion(categorias_unicas, client=openai_client)
                                if sugerencia: st.session_state.codificaciones_ordinales_cache[clave_cache] = sugerencia
                            else:
                                st.session_state.log_messages.append(f"  Usando codificación guardada para categorías de '{col_actual_en_df_proc}'.")

                            if sugerencia and sugerencia.get("is_ordinal") and isinstance(sugerencia.get("mapping_dict"), dict):
                                mapeo_texto_a_numero = {}
                                etiquetas_valor_spss_para_esta_col = {}
                                for k_texto, v_numero in sugerencia["mapping_dict"].items():
                                    try:
                                        val_num = int(v_numero) # Asegurar que es int
                                        mapeo_texto_a_numero[str(k_texto)] = val_num
                                        etiquetas_valor_spss_para_esta_col[val_num] = str(k_texto)[:120] # Límite SPSS
                                    except (ValueError, TypeError):
                                        st.session_state.log_messages.append(f"    Advertencia: Valor no numérico o nulo '{v_numero}' para '{k_texto}' en '{col_actual_en_df_proc}'. Omitiendo esta categoría del mapeo.")
                                
                                if not mapeo_texto_a_numero:
                                    st.session_state.log_messages.append(f"  No se generó un mapeo válido para '{col_actual_en_df_proc}' después de procesar sugerencia LLM.")
                                    continue

                                columna_destino_spss_name = ""
                                # Mapear usando la columna original de df_original (convertida a str)
                                # y asignar a la columna correspondiente en df_processed.
                                source_column_for_mapping = df_original[nombre_col_df_original].astype(str).replace('', 'nan')


                                if ordinal_encoding_mode == "Crear nuevas columnas (ej. VarName_num)":
                                    columna_destino_spss_name = f"{col_actual_en_df_proc}_num"
                                    temp_dest_name = columna_destino_spss_name
                                    cnt = 1
                                    while temp_dest_name in df_processed.columns:
                                        temp_dest_name = f"{col_actual_en_df_proc}_num{cnt}"
                                        cnt+=1
                                    columna_destino_spss_name = temp_dest_name
                                    
                                    df_processed[columna_destino_spss_name] = source_column_for_mapping.map(mapeo_texto_a_numero)
                                    original_var_label = spss_variable_labels_dict.get(col_actual_en_df_proc, col_actual_en_df_proc)
                                    spss_variable_labels_dict[columna_destino_spss_name] = f"{original_var_label} (Numérico)"[:256]
                                    st.session_state.log_messages.append(f"  Columna '{nombre_col_df_original}' mapeada a NUEVA '{columna_destino_spss_name}'.")
                                else: 
                                    columna_destino_spss_name = col_actual_en_df_proc 
                                    df_processed[columna_destino_spss_name] = source_column_for_mapping.map(mapeo_texto_a_numero)
                                    st.session_state.log_messages.append(f"  Valores en '{columna_destino_spss_name}' REEMPLAZADOS con codificación numérica.")
                                
                                spss_value_labels_dict[columna_destino_spss_name] = etiquetas_valor_spss_para_esta_col
                                
                                try:
                                    df_processed[columna_destino_spss_name] = pd.to_numeric(df_processed[columna_destino_spss_name], errors='coerce').astype(pd.Int64Dtype())
                                except Exception: 
                                    df_processed[columna_destino_spss_name] = pd.to_numeric(df_processed[columna_destino_spss_name], errors='coerce').astype(float)
                                
                                # Log de valores no mapeados
                                unmapped_count = df_processed[columna_destino_spss_name].isnull().sum()
                                original_non_null = source_column_for_mapping.notnull().sum()
                                if unmapped_count > 0:
                                     # Contar cuántos de los NaN en la columna procesada eran originalmente no-NaN en la source_column_for_mapping (después de .astype(str).replace('', 'nan'))
                                    actually_unmapped_count = source_column_for_mapping[df_processed[columna_destino_spss_name].isnull()].notnull().sum()
                                    if actually_unmapped_count > 0:
                                        st.session_state.log_messages.append(f"    Advertencia: {actually_unmapped_count} valores de '{nombre_col_df_original}' no se mapearon a números en '{columna_destino_spss_name}' (ahora son NaN).")


                            else: 
                                if sugerencia and not sugerencia.get("is_ordinal"):
                                    st.session_state.log_messages.append(f"  LLM (o caché) determinó que '{col_actual_en_df_proc}' no es ordinal.")
                                elif sugerencia and not isinstance(sugerencia.get("mapping_dict"), dict) :
                                     st.session_state.log_messages.append(f"  LLM (o caché) no generó un diccionario de mapeo válido para '{col_actual_en_df_proc}'.")
                                else: # sugerencia es None
                                     st.session_state.log_messages.append(f"  No se obtuvo sugerencia del LLM (o caché) para '{col_actual_en_df_proc}'.")

                        cols_to_encode_spinner.empty()
                    else: 
                        st.session_state.log_messages.append("Cliente OpenAI no configurado. No se realizará codificación ordinal.")
                    st.session_state.log_messages.append("--- Fin de Codificación Ordinal ---")
                else:
                    st.session_state.log_messages.append("\nCodificación ordinal omitida por el usuario.")


                st.session_state.df_processed = df_processed
                st.session_state.spss_variable_labels = spss_variable_labels_dict
                st.session_state.spss_value_labels = spss_value_labels_dict
                
                st.success("🎉 ¡Procesamiento completado!")
                st.subheader("Log del Proceso")
                st.text_area("Mensajes:", "\n".join(st.session_state.log_messages), height=300)

    except Exception as e:
        st.error(f"Ocurrió un error al cargar o procesar el archivo: {e}")
        st.exception(e)
        st.session_state.df_processed = None
        st.session_state.log_messages.append(f"ERROR FATAL: {e}")


if st.session_state.df_processed is not None:
    st.subheader("Vista Previa del DataFrame Procesado (antes de sanitización final para .sav)")
    st.dataframe(st.session_state.df_processed.head())

    temp_file_path = None
    try:
        df_to_write = st.session_state.df_processed.copy()
        
        # Log para sanitización final
        sanitization_log_messages = ["\n--- Iniciando Sanitización Final y Preparación para .sav ---"]

        current_col_names_before_final_sanitize = list(df_to_write.columns)
        final_sav_column_names = []
        
        temp_spss_variable_labels = st.session_state.spss_variable_labels.copy()
        temp_spss_value_labels = st.session_state.spss_value_labels.copy()

        final_spss_variable_labels_for_sav = {}
        final_spss_value_labels_for_sav = {}
        
        seen_final_names = set()

        for col_name_in_df_to_write in current_col_names_before_final_sanitize:
            base_sanitized_name = sanitize_spss_varname(col_name_in_df_to_write)
            unique_final_name = base_sanitized_name
            count = 1
            while unique_final_name in seen_final_names:
                prefix = base_sanitized_name[:SPSS_VAR_NAME_MAX_LEN - (len(str(count)) + 1)]
                unique_final_name = f"{prefix}_{count}"
                count += 1
            seen_final_names.add(unique_final_name)
            final_sav_column_names.append(unique_final_name)

            original_label = temp_spss_variable_labels.get(col_name_in_df_to_write, str(col_name_in_df_to_write)[:256])
            final_spss_variable_labels_for_sav[unique_final_name] = original_label

            if col_name_in_df_to_write in temp_spss_value_labels:
                final_spss_value_labels_for_sav[unique_final_name] = temp_spss_value_labels[col_name_in_df_to_write]
            
            if col_name_in_df_to_write != unique_final_name:
                 sanitization_log_messages.append(f"  Nombre de columna final para SPSS: '{col_name_in_df_to_write}' -> '{unique_final_name}'")

        df_to_write.columns = final_sav_column_names 
        sanitization_log_messages.append("Nombres de columna finales aplicados a df_to_write.")
        
        column_labels_list_for_sav = []
        for final_col_name in df_to_write.columns:
            column_labels_list_for_sav.append(final_spss_variable_labels_for_sav.get(final_col_name, str(final_col_name)[:256]))
        
        # Limpieza de tipos de datos y MODIFICACIÓN 2: missing_ranges para strings
        spss_missing_ranges = {}
        for col in df_to_write.columns:
            # Intentar convertir a numérico si es objeto y parece numérico
            if df_to_write[col].dtype == 'object':
                try:
                    # Chequear si todos los no-nulos pueden ser numéricos
                    is_potentially_numeric = True
                    # Convertir a string para la comprobación, luego intentar numérico
                    temp_series_str = df_to_write[col].astype(str)
                    # Si después de convertir a string, alguno no es 'nan' y no puede ser numérico, no es numérico.
                    # Esto es un poco complicado porque pd.to_numeric puede convertir 'nan' a np.nan.
                    # Primero, intentemos pd.to_numeric con errors='coerce'
                    numeric_series = pd.to_numeric(df_to_write[col], errors='coerce')
                    if not numeric_series.isnull().all(): # Si hay algún valor numérico
                         # Comprobar si los valores originales que se convirtieron a NaN eran realmente strings no numéricos
                        original_values_that_became_nan = df_to_write[col][numeric_series.isnull()]
                        # Si alguno de estos era un string no numérico (y no NaN/None original), entonces no era puramente numérico
                        if not all(pd.isna(x) or (isinstance(x, str) and x.lower() == 'nan') for x in original_values_that_became_nan):
                            is_potentially_numeric = False # Contiene strings que no son 'nan' y no son numéricos
                    
                    if is_potentially_numeric and not numeric_series.isnull().all(): # Si es numérico y no todo NaN
                        df_to_write[col] = numeric_series
                        sanitization_log_messages.append(f"  Columna '{col}' convertida a tipo numérico.")
                    else: # No se pudo convertir completamente a numérico o es todo NaN
                        df_to_write[col] = df_to_write[col].astype(str) # Convertir a string si no
                        spss_missing_ranges[col] = ['nan'] # Aplicar missing range para strings
                        sanitization_log_messages.append(f"  Columna '{col}' tratada como string. Missing range ['nan'] aplicado.")

                except ValueError: 
                    df_to_write[col] = df_to_write[col].astype(str) # Convertir a string si falla
                    spss_missing_ranges[col] = ['nan'] 
                    sanitization_log_messages.append(f"  Columna '{col}' (error en conversión numérica) tratada como string. Missing range ['nan'] aplicado.")
            
            # Si ya es un tipo string explícito (ej. pd.StringDtype)
            elif pd.api.types.is_string_dtype(df_to_write[col].dtype):
                spss_missing_ranges[col] = ['nan']
                sanitization_log_messages.append(f"  Columna '{col}' (tipo string) con missing range ['nan'] aplicado.")
            
            # Para tipos numéricos, pyreadstat maneja NaN como system missing por defecto.
            elif pd.api.types.is_numeric_dtype(df_to_write[col].dtype):
                 sanitization_log_messages.append(f"  Columna '{col}' es numérica. NaN se tratará como system missing.")


        sanitization_log_messages.append("--- Fin de Sanitización Final y Preparación ---")
        
        # Añadir logs de sanitización a los logs generales
        st.session_state.log_messages.extend(sanitization_log_messages)
        # Mostrar logs actualizados
        st.subheader("Log del Proceso (incluye sanitización final)")
        st.text_area("Mensajes (actualizado con sanitización final):", "\n".join(st.session_state.log_messages), height=200)


        with tempfile.NamedTemporaryFile(delete=False, suffix=".sav") as tmp_file:
            temp_file_path = tmp_file.name
        
    
        pyreadstat.write_sav(
            df_to_write, 
            temp_file_path, 
            column_labels=column_labels_list_for_sav, 
            variable_value_labels=final_spss_value_labels_for_sav,
            missing_ranges=spss_missing_ranges # MODIFICACIÓN 2: Pasar los missing ranges
        )

        with open(temp_file_path, "rb") as f:
            sav_bytes = f.read()
        
        output_filename_sav = "datos_procesados.sav"
        if uploaded_file:
            base, _ = os.path.splitext(uploaded_file.name)
            output_filename_sav = f"{base}_procesado.sav"

        st.download_button(
            label="📥 Descargar Archivo .sav Procesado",
            data=sav_bytes,
            file_name=output_filename_sav,
            mime="application/octet-stream",
        )

    except Exception as e:
        st.error(f"Error al generar el archivo .sav: {e}")
        st.exception(e)
        st.session_state.log_messages.append(f"ERROR FATAL DURANTE GENERACIÓN .SAV: {e}")
        # Mostrar logs actualizados con el error
        st.subheader("Log del Proceso (con error en .sav)")
        st.text_area("Mensajes (actualizado con error):", "\n".join(st.session_state.log_messages), height=200)

    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            try: os.remove(temp_file_path)
            except Exception as e_del: st.warning(f"No se pudo eliminar el archivo temporal {temp_file_path}: {e_del}")
else:
    if uploaded_file:
        st.info("Haz clic en 'Procesar Datos para .sav' en la barra lateral para comenzar.")
    else:
        st.info("Sube un archivo CSV o Excel para empezar.")

st.sidebar.markdown("---")
st.sidebar.caption(f"Usando modelo LLM: {MODELO_LLM_PRINCIPAL}")