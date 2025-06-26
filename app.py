import streamlit as st
import openai
import json
import pandas as pd
import re # Para un fallback si el LLM falla
import os
import io # Para manejar bytes en memoria para la descarga
import pyreadstat # Para guardar archivos .sav
import tempfile 

# --- Configuración de la Página de Streamlit ---
st.set_page_config(layout="wide", page_title="Procesador de Datos para SPSS")

st.title("Procesador de Datos de Encuestas con IA para SPSS (.sav)")
st.markdown("""
Sube un archivo CSV o Excel, elige las operaciones y descarga los resultados en formato `.sav`.
- **Simplificar Nombres de Columnas**: Usa un LLM para acortar los nombres largos de las columnas (serán los nombres de variable en SPSS).
- **Generar Etiquetas de Variables**: Usa un LLM para crear etiquetas descriptivas para cada variable (serán las etiquetas de variable en SPSS).
- **Codificar Variables Ordinales**: Identifica columnas con categorías ordinales (ej. escalas Likert) y las convierte a números, usando los textos originales como etiquetas de valor en SPSS.
""")

# --- Constantes ---
MAX_CATEGORIAS_PARA_LLM = 10 # Para la codificación ordinal
SPSS_VAR_NAME_MAX_LEN = 64 # Límite de longitud para nombres de variable en SPSS

# --- NUEVA FUNCIÓN DE SANITIZACIÓN ---
def sanitize_spss_varname(name_str):
    """
    Sanitiza un string para que sea un nombre de variable válido para SPSS.
    - Elimina caracteres inválidos.
    - Asegura que comience con una letra (prefija con 'V_' si no).
    - Trunca a SPSS_VAR_NAME_MAX_LEN.
    - Reemplaza espacios y algunos caracteres especiales con '_'.
    """
    name_str = str(name_str)
    # Reemplazar common problematic characters or sequences
    name_str = re.sub(r'[.:\-/]', '_', name_str) 
    name_str = re.sub(r'\s+', '_', name_str)     
    # Keep only alphanumerics and underscore
    name_str = re.sub(r'[^\w_]', '', name_str)  

    # Ensure starts with a letter
    if not name_str or not name_str[0].isalpha():
        name_str = "V_" + name_str 
    
    # Truncate to max length
    name_str = name_str[:SPSS_VAR_NAME_MAX_LEN]
    
    if not name_str: # Handle case where sanitization results in empty string
        return "UnnamedVar"
    return name_str

# --- Funciones LLM (Originales y Nuevas) ---
# ... (El resto de las funciones: simplify_survey_column_names_llm, basic_column_simplifier, 
# generate_variable_labels_llm, get_llm_mapping_suggestion se mantienen EXACTAMENTE IGUALES a tu última versión) ...
def simplify_survey_column_names_llm(column_names_list, client, model="gpt-4.1-mini"):
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
5.  Evita la jerga a menos que sea universalmente entendida en el contexto de encuestas.
6.  No dupliques nombres. Si es necesario, añade un sufijo numérico (ej. Item1, Item2).
7.  Mantiene el mismo idioma que el original.

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
            s = str(simplified)
            s = re.sub(r'\s+', '_', s) 
            s = re.sub(r'[^a-zA-Z0-9_]', '', s) 
            if not s or not s[0].isalpha(): 
                s = "Var_" + s 
            s = s[:SPSS_VAR_NAME_MAX_LEN] 
            validated_response[original] = s if s else f"Var_{original[:SPSS_VAR_NAME_MAX_LEN-4]}"
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

def basic_column_simplifier(column_name, max_words=3): # Usado como fallback o si LLM no se usa
    name = re.sub(r':Please say whether you AGREE or DISAGREE with the following statements\.', '', column_name, flags=re.IGNORECASE)
    name = re.sub(r'Please tell us.*', '', name, flags=re.IGNORECASE)
    name = re.sub(r'\(optional\)', '', name, flags=re.IGNORECASE)
    name = re.sub(r'[¿?.:!]', '', name)
    words = name.strip().split()
    # PascalCase or snake_case for simplifier
    simplified = ''.join([word.capitalize() for word in words[:max_words]]) if words else column_name
    
    # Apply general SPSS sanitization rules
    simplified = str(simplified)
    simplified = re.sub(r'\s+', '_', simplified)
    simplified = re.sub(r'[^a-zA-Z0-9_]', '', simplified)
    if not simplified or not simplified[0].isalpha():
        simplified = "V_" + simplified # Prefix to ensure it starts with a letter
    
    final_name = simplified[:SPSS_VAR_NAME_MAX_LEN]
    return final_name if final_name else "UnnamedVar"


def generate_variable_labels_llm(column_name_map, client, model="gpt-4.1-mini"):
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
  Etiqeuta de Variable: "Imagen Diana Boluarte"

Lista de nombres de variable y sus descripciones originales:
{items_for_prompt}

Formato JSON de salida esperado:
{{
  "NombreVariable1": "Etiqueta Descriptiva Larga para Variable 1",
  "NombreVariable2": "Etiqueta Descriptiva Larga para Variable 2",
  ...
}}
"""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Eres un experto en crear etiquetas de variable descriptivas para SPSS. Devuelves solo JSON."},
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
        final_labels = {key: str(value)[:256] for key, value in parsed_response.items()}
        missing_keys = [name for name in column_name_map_for_prompt.keys() if name not in final_labels]
        if missing_keys:
            st.warning(f"Advertencia: El LLM no devolvió etiquetas para las siguientes variables: {missing_keys}")
            for key in missing_keys:
                final_labels[key] = key 
        return final_labels
    except json.JSONDecodeError as e:
        st.error(f"Error al decodificar JSON de la respuesta del LLM (etiquetas de variable): {e}")
        st.text_area("Respuesta recibida del LLM (etiquetas de variable):", llm_response_json, height=150)
        return None

def get_llm_mapping_suggestion(categories_list, client, model="gpt-4-turbo-preview"):
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
3. Si es ordinal pero tiene alguna categoría que no se puede ordenar o representa un "no sabe/no contesta" (ej. "No sé", "Prefiero no responder", "N/A"), mapea las categorías ordenables y asigna un valor numérico alto (ej. 99, 98) a estas categorías no ordenables/missing. Estos se usarán luego como valores perdidos definidos por el usuario en SPSS. Si no hay tales categorías, no incluyas valores altos.
4. Devuelve tu respuesta ÚNICAMENTE como un objeto JSON con la estructura:
   {{
     "is_ordinal": true_or_false,
     "mapping_dict": {{ "categoria_texto_1": numero_1, "categoria_texto_2": numero_2, "No sé": 99, ... }}
   }}
   Si `is_ordinal` es false, `mapping_dict` debe ser `null`.
Usa las categorías de texto exactas proporcionadas como claves en `mapping_dict`.

Ejemplos de mapeo:
- ["Totalmente en desacuerdo", "En desacuerdo", "Neutral", "De acuerdo", "Totalmente de acuerdo"] -> {{"Totalmente de acuerdo": 1, "De acuerdo": 2, "Neutral": 3, "En desacuerdo": 4, "Totalmente en desacuerdo": 5}}
- ["Muy Malo", "Malo", "Regular", "Bueno", "Excelente", "No aplica"] -> {{"Muy Bueno": 1, "Bueno": 2, "Regular": 3, "Malo": 4, "Muy malo": 5, "No aplica": 99}}
- ["Bajo", "Medio", "Alto"] -> {{"Alto": 1, "Medio": 2, "Bajo": 3}}
- ["Sí", "No", "Quizás"] -> {{"Sí": 1, "Quizás": 2, "No": 3}} (orden lógico, no alfabético) o bien {{"Sí": 1, "No": 0, "Quizás": 2}} si es más intuitivo para sí/no. Sé consistente.
- ["Manzana", "Banana", "Cereza"] -> is_ordinal: false, mapping_dict: null (nominal)

Lista de categorías:
{categories_list_string}
"""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Eres un asistente experto en análisis de datos y encuestas, especializado en devolver respuestas en formato JSON para escalas ordinales."},
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
do_simplify_cols = st.sidebar.checkbox("Simplificar nombres de columnas (Nombres de Variable SPSS)", value=True)
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
if 'codificaciones_ordinales_cache' not in st.session_state: st.session_state.codificaciones_ordinales_cache = {}

if uploaded_file is not None:
    st.subheader("Vista Previa del Archivo Original")
    try:
        if uploaded_file.name.endswith('.csv'):
            df_original = pd.read_csv(uploaded_file, dtype=str)
        else:
            df_original = pd.read_excel(uploaded_file, dtype=str)
        st.dataframe(df_original.head())

        if st.sidebar.button("🚀 Procesar Datos para .sav"):
            with st.spinner("Procesando datos... Por favor espera."):
                df_processed = df_original.copy()
                log_messages = []
                
                spss_variable_labels_dict = {} 
                spss_value_labels_dict = {}  
                original_to_simplified_map_for_labels = {col: col for col in df_processed.columns}

                # 1. Simplificar Nombres de Columnas (Nombres de Variable SPSS) - Opcional
                if do_simplify_cols:
                    log_messages.append("--- Iniciando Simplificación de Nombres de Columnas ---")
                    if openai_client:
                        original_column_names_list = df_processed.columns.tolist()
                        with st.spinner("Simplificando nombres de columnas (Nombres de Variable SPSS) con LLM..."):
                            simplified_names_map_llm = simplify_survey_column_names_llm(original_column_names_list, client=openai_client)

                        if simplified_names_map_llm:
                            log_messages.append("Mapa de nombres de variable simplificados (LLM) obtenido.")
                            # Aplicar y asegurar unicidad
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
                            log_messages.append("Nombres de columna (variables SPSS) simplificados y aplicados.")
                        else: # Fallback si LLM falla completamente
                            log_messages.append("No se pudieron simplificar los nombres con LLM. Usando fallback básico para todos.")
                            fb_rename_map = {name: basic_column_simplifier(name) for name in df_processed.columns}
                            # Asegurar unicidad también en fallback
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
                    else: # No hay cliente OpenAI
                        log_messages.append("Cliente OpenAI no configurado. Aplicando simplificación básica de renombrado.")
                        # (Misma lógica de fallback con unicidad que arriba)
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
                    log_messages.append("--- Fin de Simplificación de Nombres ---")
                
                # `df_processed.columns` ahora tiene los nombres "simplificados" o los originales.
                # `original_to_simplified_map_for_labels` mapea el nombre original del df_original al nombre actual en df_processed.

                # 2. Generar Etiquetas de Variables (Etiquetas de Variable SPSS)
                if do_generate_var_labels:
                    log_messages.append("\n--- Iniciando Generación de Etiquetas de Variable ---")
                    if openai_client:
                        # Para generar etiquetas, necesitamos el nombre de variable actual en df_processed y su pregunta original.
                        # `original_to_simplified_map_for_labels` = {df_original_col: df_processed_col}
                        # Necesitamos invertirlo o iterar sobre él para construir: {df_processed_col: df_original_col}
                        map_current_name_to_original_question = {}
                        for original_q_name, current_df_proc_name in original_to_simplified_map_for_labels.items():
                             # Asegurarse que current_df_proc_name es una columna actual en df_processed
                            if current_df_proc_name in df_processed.columns:
                                map_current_name_to_original_question[current_df_proc_name] = original_q_name
                            else: # Si no está, y la simplificación no se hizo, el nombre original es el actual
                                if not do_simplify_cols and original_q_name in df_processed.columns:
                                     map_current_name_to_original_question[original_q_name] = original_q_name


                        if not map_current_name_to_original_question: # Fallback general si el mapeo anterior falla
                            map_current_name_to_original_question = {col: col for col in df_processed.columns}
                        
                        with st.spinner("Generando etiquetas de variable (Etiquetas SPSS) con LLM..."):
                            generated_labels = generate_variable_labels_llm(map_current_name_to_original_question, client=openai_client)
                        
                        if generated_labels:
                            spss_variable_labels_dict = generated_labels # Claves son nombres de df_processed
                            log_messages.append("Etiquetas de variable generadas por LLM.")
                        else:
                            log_messages.append("No se pudieron generar etiquetas de variable con LLM.")
                            spss_variable_labels_dict = {col: str(map_current_name_to_original_question.get(col, col))[:256] for col in df_processed.columns} 
                    else: # No hay cliente OpenAI
                        log_messages.append("Cliente OpenAI no configurado para etiquetas de variable.")
                        spss_variable_labels_dict = {col: str(original_to_simplified_map_for_labels.get(col,col))[:256] for col in df_processed.columns}
                    log_messages.append("--- Fin de Generación de Etiquetas de Variable ---")
                else: # No se generan etiquetas con LLM
                    # Usar la pregunta original (si está disponible a través del mapeo) o el nombre de columna actual como etiqueta.
                    for current_col_name_in_df_proc in df_processed.columns:
                        original_question = current_col_name_in_df_proc # Default
                        for orig_q, simpl_name in original_to_simplified_map_for_labels.items():
                            if simpl_name == current_col_name_in_df_proc:
                                original_question = orig_q
                                break
                        spss_variable_labels_dict[current_col_name_in_df_proc] = str(original_question)[:256]

                # 3. Codificar Variables Ordinales
                # ... (Esta sección se mantiene igual, pero es CRUCIAL que las claves en 
                # `spss_value_labels_dict` sean los nombres de columna *actuales* en `df_processed`
                # o los nuevos `_num` basados en esos nombres actuales)
                if do_encode_ordinal:
                    log_messages.append("\n--- Iniciando Codificación de Variables Ordinales ---")
                    if openai_client:
                        columnas_a_evaluar_para_ordinal = list(df_processed.columns) # Nombres actuales en df_processed
                        cols_to_encode_spinner = st.empty()

                        for i, col_actual_en_df_proc in enumerate(columnas_a_evaluar_para_ordinal):
                            cols_to_encode_spinner.info(f"Evaluando columna para codificación ordinal: '{col_actual_en_df_proc}' ({i+1}/{len(columnas_a_evaluar_para_ordinal)})")
                            log_messages.append(f"\nProcesando columna para codificación: '{col_actual_en_df_proc}'")

                            # Encontrar el nombre original en df_original que corresponde a col_actual_en_df_proc
                            nombre_col_df_original = None
                            for orig_name, current_name in original_to_simplified_map_for_labels.items():
                                if current_name == col_actual_en_df_proc:
                                    nombre_col_df_original = orig_name
                                    break
                            if not nombre_col_df_original and col_actual_en_df_proc in df_original.columns: # Fallback si no hubo simplificación
                                nombre_col_df_original = col_actual_en_df_proc
                            
                            if not nombre_col_df_original or nombre_col_df_original not in df_original.columns:
                                log_messages.append(f"  Advertencia: No se pudo encontrar la columna original para '{col_actual_en_df_proc}'. Omitiendo.")
                                continue
                            
                            # Obtener categorías de la columna original en df_original
                            try:
                                categorias_unicas_series = df_original[nombre_col_df_original].dropna().astype(str)
                                categorias_unicas = sorted(list(categorias_unicas_series.unique()))
                            except Exception as e:
                                log_messages.append(f"  Error al obtener categorías únicas de '{nombre_col_df_original}' para '{col_actual_en_df_proc}': {e}. Omitiendo.")
                                continue
                            
                            if not categorias_unicas or not (1 < len(categorias_unicas) <= MAX_CATEGORIAS_PARA_LLM):
                                # Logica de omision existente
                                if not categorias_unicas: log_messages.append(f"  Omitiendo '{col_actual_en_df_proc}': Sin categorías procesables.")
                                else: log_messages.append(f"  Omitiendo '{col_actual_en_df_proc}': {len(categorias_unicas)} categorías (límite 1-{MAX_CATEGORIAS_PARA_LLM}).")
                                continue

                            clave_cache = tuple(categorias_unicas)
                            sugerencia = st.session_state.codificaciones_ordinales_cache.get(clave_cache)
                            if not sugerencia:
                                log_messages.append(f"  Consultando LLM para categorías de '{col_actual_en_df_proc}': {clave_cache}.")
                                with st.spinner(f"Consultando LLM para '{col_actual_en_df_proc}'..."):
                                     sugerencia = get_llm_mapping_suggestion(categorias_unicas, client=openai_client)
                                if sugerencia: st.session_state.codificaciones_ordinales_cache[clave_cache] = sugerencia
                            else:
                                log_messages.append(f"  Usando codificación guardada para categorías de '{col_actual_en_df_proc}'.")

                            if sugerencia and sugerencia.get("is_ordinal") and isinstance(sugerencia.get("mapping_dict"), dict):
                                mapeo_texto_a_numero = {}
                                etiquetas_valor_spss_para_esta_col = {}
                                for k_texto, v_numero in sugerencia["mapping_dict"].items():
                                    try:
                                        val_num = int(v_numero)
                                        mapeo_texto_a_numero[str(k_texto)] = val_num
                                        etiquetas_valor_spss_para_esta_col[val_num] = str(k_texto)[:120]
                                    except (ValueError, TypeError):
                                        log_messages.append(f"    Advertencia: Valor no numérico '{v_numero}' para '{k_texto}' en '{col_actual_en_df_proc}'. Omitiendo esta categoría del mapeo.")
                                
                                if not mapeo_texto_a_numero:
                                    log_messages.append(f"  No se generó un mapeo válido para '{col_actual_en_df_proc}'.")
                                    continue

                                columna_destino_spss_name = ""
                                if ordinal_encoding_mode == "Crear nuevas columnas (ej. VarName_num)":
                                    # El nombre base para _num es col_actual_en_df_proc
                                    columna_destino_spss_name = f"{col_actual_en_df_proc}_num"
                                    # Asegurar unicidad para esta nueva columna _num también
                                    temp_dest_name = columna_destino_spss_name
                                    cnt = 1
                                    while temp_dest_name in df_processed.columns:
                                        temp_dest_name = f"{col_actual_en_df_proc}_num{cnt}"
                                        cnt+=1
                                    columna_destino_spss_name = temp_dest_name
                                    
                                    df_processed[columna_destino_spss_name] = df_original[nombre_col_df_original].astype(str).map(mapeo_texto_a_numero)
                                    # Añadir etiqueta de variable para la nueva columna numérica
                                    original_var_label = spss_variable_labels_dict.get(col_actual_en_df_proc, col_actual_en_df_proc)
                                    spss_variable_labels_dict[columna_destino_spss_name] = f"{original_var_label} (Numérico)"[:256]
                                    log_messages.append(f"  Columna '{nombre_col_df_original}' mapeada a NUEVA '{columna_destino_spss_name}'.")
                                else: # Reemplazar
                                    columna_destino_spss_name = col_actual_en_df_proc # Sobreescribir la columna existente en df_processed
                                    df_processed[columna_destino_spss_name] = df_original[nombre_col_df_original].astype(str).map(mapeo_texto_a_numero)
                                    log_messages.append(f"  Valores en '{columna_destino_spss_name}' REEMPLAZADOS con codificación numérica.")
                                
                                spss_value_labels_dict[columna_destino_spss_name] = etiquetas_valor_spss_para_esta_col
                                
                                try:
                                    df_processed[columna_destino_spss_name] = pd.to_numeric(df_processed[columna_destino_spss_name], errors='coerce').astype(pd.Int64Dtype())
                                except Exception: # Fallback a float si no puede ser Int64
                                    df_processed[columna_destino_spss_name] = pd.to_numeric(df_processed[columna_destino_spss_name], errors='coerce').astype(float)

                                # Log de valores no mapeados...
                            else: # No es ordinal o no hay mapeo
                                log_messages.append(f"  LLM (o caché) determinó que '{col_actual_en_df_proc}' no es ordinal o no generó mapeo válido.")
                        cols_to_encode_spinner.empty()
                    else: # No hay cliente OpenAI
                        log_messages.append("Cliente OpenAI no configurado. No se realizará codificación ordinal.")
                    log_messages.append("--- Fin de Codificación Ordinal ---")

                st.session_state.df_processed = df_processed
                # Guardar las etiquetas que se han generado hasta ahora, usando los nombres de columna *actuales* en df_processed
                st.session_state.spss_variable_labels = spss_variable_labels_dict
                st.session_state.spss_value_labels = spss_value_labels_dict
                
                st.success("🎉 ¡Procesamiento completado!")
                st.subheader("Log del Proceso")
                st.text_area("Mensajes:", "\n".join(log_messages), height=300)

    except Exception as e:
        st.error(f"Ocurrió un error al cargar o procesar el archivo: {e}")
        st.exception(e)
        st.session_state.df_processed = None

# Mostrar DataFrame procesado y botón de descarga .sav
if st.session_state.df_processed is not None:
    st.subheader("Vista Previa del DataFrame Procesado (antes de sanitización final para .sav)")
    st.dataframe(st.session_state.df_processed.head())

    temp_file_path = None
    try:
        # Copiar el DataFrame procesado para la sanitización final y escritura
        df_to_write = st.session_state.df_processed.copy()

        # --- INICIO DE SANITIZACIÓN FINAL OBLIGATORIA PARA NOMBRES DE VARIABLE SPSS ---
        #log_messages.append("\n--- Iniciando Sanitización Final de Nombres de Variable para SPSS ---")
        
        current_col_names_before_final_sanitize = list(df_to_write.columns)
        final_sav_column_names = []
        
        # Copiar las etiquetas actuales para remapearlas con los nombres finales sanitizados
        # Estas etiquetas ya están basadas en los nombres de df_processed (que pueden ser simplificados o no)
        temp_spss_variable_labels = st.session_state.spss_variable_labels.copy()
        temp_spss_value_labels = st.session_state.spss_value_labels.copy()

        final_spss_variable_labels_for_sav = {}
        final_spss_value_labels_for_sav = {}
        
        seen_final_names = set()

        for col_name_in_df_to_write in current_col_names_before_final_sanitize:
            # Aplicar la función de sanitización y truncamiento
            base_sanitized_name = sanitize_spss_varname(col_name_in_df_to_write)
            
            # Asegurar unicidad del nombre final sanitizado
            unique_final_name = base_sanitized_name
            count = 1
            while unique_final_name in seen_final_names:
                prefix = base_sanitized_name[:SPSS_VAR_NAME_MAX_LEN - (len(str(count)) + 1)]
                unique_final_name = f"{prefix}_{count}"
                count += 1
            seen_final_names.add(unique_final_name)
            final_sav_column_names.append(unique_final_name)

            # Remapear etiquetas de variable
            # La clave en temp_spss_variable_labels es col_name_in_df_to_write
            original_label = temp_spss_variable_labels.get(col_name_in_df_to_write, str(col_name_in_df_to_write)[:256])
            final_spss_variable_labels_for_sav[unique_final_name] = original_label

            # Remapear etiquetas de valor
            # La clave en temp_spss_value_labels es col_name_in_df_to_write
            if col_name_in_df_to_write in temp_spss_value_labels:
                final_spss_value_labels_for_sav[unique_final_name] = temp_spss_value_labels[col_name_in_df_to_write]
            
            #if col_name_in_df_to_write != unique_final_name:
            #     log_messages.append(f"  Nombre de columna final para SPSS: '{col_name_in_df_to_write}' -> '{unique_final_name}'")

        df_to_write.columns = final_sav_column_names # Aplicar los nombres finales al DataFrame
        #log_messages.append("--- Fin de Sanitización Final de Nombres ---")
        # --- FIN DE SANITIZACIÓN FINAL ---

        # Preparar la lista de etiquetas de columna en el orden correcto para pyreadstat
        column_labels_list_for_sav = []
        for final_col_name in df_to_write.columns:
            column_labels_list_for_sav.append(final_spss_variable_labels_for_sav.get(final_col_name, str(final_col_name)[:256]))
        
        # Limpieza de tipos de datos (como antes, pero sobre df_to_write)
        for col in df_to_write.columns:
            if df_to_write[col].dtype == 'object':
                try: pd.to_numeric(df_to_write[col]) # Chequear si es numérico
                except ValueError: df_to_write[col] = df_to_write[col].astype(str) # Convertir a string si no

        with tempfile.NamedTemporaryFile(delete=False, suffix=".sav") as tmp_file:
            temp_file_path = tmp_file.name
        
        pyreadstat.write_sav(
            df_to_write, 
            temp_file_path, 
            column_labels=column_labels_list_for_sav, # Usa la lista ordenada de etiquetas
            variable_value_labels=final_spss_value_labels_for_sav, # Usa el dict con claves finales
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
        # Mostrar logs actualizados con la sanitización final
        #st.subheader("Log del Proceso (incluye sanitización final)")
        #st.text_area("Mensajes finales:", "\n".join(log_messages), height=150)


    except Exception as e:
        st.error(f"Error al generar el archivo .sav: {e}")
        st.exception(e)
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
st.sidebar.markdown("Creado con Streamlit, OpenAI y Pyreadstat.")