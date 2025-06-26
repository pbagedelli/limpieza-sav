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

st.title("📊 Procesador de Datos de Encuestas con IA para SPSS (.sav)")
st.markdown("""
Sube un archivo CSV o Excel, elige las operaciones y descarga los resultados en formato `.sav`.
- **Simplificar Nombres de Columnas**: Usa un LLM para acortar los nombres largos de las columnas (serán los nombres de variable en SPSS).
- **Generar Etiquetas de Variables**: Usa un LLM para crear etiquetas descriptivas para cada variable (serán las etiquetas de variable en SPSS).
- **Codificar Variables Ordinales**: Identifica columnas con categorías ordinales (ej. escalas Likert) y las convierte a números, usando los textos originales como etiquetas de valor en SPSS.
""")

# --- Constantes ---
MAX_CATEGORIAS_PARA_LLM = 10 # Para la codificación ordinal

# --- Funciones LLM (Originales y Nuevas) ---

def simplify_survey_column_names_llm(column_names_list, client, model="gpt-4-turbo-preview"): # gpt-4.1-mini no existe aún, usando turbo
    """
    Simplifica nombres de columnas largos (típicamente preguntas de encuestas)
    a etiquetas cortas y significativas, usando un LLM. Serán los NOMBRES de variable en SPSS.
    """
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
1.  Deben ser muy concisos (idealmente 1-3 palabras, máximo 64 caracteres según SPSS, pero apunta a mucho menos).
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
                {"role": "system", "content": "Eres un experto en crear nombres de variable cortos y válidos para SPSS a partir de preguntas de encuestas. Devuelves solo JSON."},
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

        # Validar nombres de variable para SPSS (simple validación)
        validated_response = {}
        for original, simplified in parsed_response.items():
            # Quitar espacios, caracteres especiales (excepto _), asegurar que empieza con letra
            s = str(simplified)
            s = re.sub(r'\s+', '_', s) # Reemplazar espacios con guion bajo
            s = re.sub(r'[^a-zA-Z0-9_]', '', s) # Quitar caracteres no alfanuméricos (excepto _)
            if not s or not s[0].isalpha(): # Si está vacío o no empieza con letra
                s = "Var_" + s # Anteponer "Var_"
            s = s[:64] # Truncar a 64 caracteres
            validated_response[original] = s

        missing_keys = [name for name in column_names_list if name not in validated_response]
        if missing_keys:
            st.warning(f"Advertencia: El LLM no devolvió nombres para las siguientes columnas originales: {missing_keys}")
            for key in missing_keys:
                validated_response[key] = basic_column_simplifier(key) # Fallback para claves faltantes
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
    simplified = ''.join([word.capitalize() for word in words[:max_words]]) # PascalCase
    simplified = re.sub(r'[^a-zA-Z0-9_]', '', simplified)
    if not simplified or not simplified[0].isalpha():
        simplified = "Var_" + simplified
    return simplified[:64] if simplified else "UnnamedVar"

def generate_variable_labels_llm(column_name_map, client, model="gpt-4-turbo-preview"):
    """
    Genera etiquetas de variables descriptivas para SPSS usando un LLM.
    El input 'column_name_map' es un diccionario: { "nombre_variable_spss": "pregunta_original_o_descripcion" }
    o simplemente una lista de nombres de variable si no hay mapeo previo.
    Devuelve un diccionario: { "nombre_variable_spss": "Etiqueta Descriptiva Larga" }
    """
    if not client:
        st.error("Error: El cliente de OpenAI no está inicializado.")
        return None
    if not column_name_map:
        st.warning("La lista/mapa de nombres de columnas para generar etiquetas está vacía.")
        return {}

    # Si es una lista, convertirla a un mapa donde la clave y el valor son iguales (usar el nombre de variable como base)
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
1.  Deben ser descriptivas y explicar completamente lo que la variable representa (hasta 256 caracteres en SPSS).
2.  Deben estar en el mismo idioma que la pregunta original.
3.  Utiliza una redacción clara y gramaticalmente correcta.
4.  Si la pregunta original es muy larga, resúmela sin perder el significado esencial.
5.  No dupliques etiquetas si los nombres de variable son distintos pero conceptualmente similares; diferencia la etiqueta ligeramente.

Devuelve tu respuesta ÚNICAMENTE como un objeto JSON que mapee cada NOMBRE DE VARIABLE (el corto que te di) a su nueva ETIQUETA DE VARIABLE (la descriptiva).
Asegúrate de que CADA nombre de variable de la lista de entrada tenga su correspondiente etiqueta en el JSON de salida.

Ejemplos:
- Nombre de Variable: "CorpInfluence", Pregunta Original: "Corporations and the wealthy have too much influence over government in this country.:Please say whether you AGREE or DISAGREE with the following statements."
  Etiqueta de Variable: "Influence of Corporations and Wealthy on Government (Agree/Disagree)"
- Nombre de Variable: "LikelyToVote", Pregunta Original: "Some people vote in elections, while others choose NOT to vote. In the current CANADIAN FEDERAL election, how likely are you to vote?"
  Etiqueta de Variable: "Likelihood of Voting in Current Canadian Federal Election"
- Nombre de Variable: "EducationLevel", Pregunta Original: "What is your highest level of education completed?"
  Etiqueta de Variable: "Highest Level of Education Completed by Respondent"

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

        # Truncar etiquetas a 256 caracteres
        final_labels = {key: str(value)[:256] for key, value in parsed_response.items()}

        missing_keys = [name for name in column_name_map_for_prompt.keys() if name not in final_labels]
        if missing_keys:
            st.warning(f"Advertencia: El LLM no devolvió etiquetas para las siguientes variables: {missing_keys}")
            for key in missing_keys:
                final_labels[key] = key # Fallback: usar el nombre de variable como etiqueta
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
2. Si es ordinal, crea un diccionario de Python que mapee cada categoría de texto a un valor numérico entero. El mapeo debe respetar el orden lógico de la escala. Los valores numéricos deben ser enteros y preferiblemente comenzar desde 1 (o el valor más bajo apropiado para la escala).
3. Si es ordinal pero tiene alguna categoría que no se puede ordenar o representa un "no sabe/no contesta" (ej. "No sé", "Prefiero no responder", "N/A"), mapea las categorías ordenables y asigna un valor numérico alto (ej. 99, 98) a estas categorías no ordenables/missing. Estos se usarán luego como valores perdidos definidos por el usuario en SPSS. Si no hay tales categorías, no incluyas valores altos.
4. Devuelve tu respuesta ÚNICAMENTE como un objeto JSON con la estructura:
   {{
     "is_ordinal": true_or_false,
     "mapping_dict": {{ "categoria_texto_1": numero_1, "categoria_texto_2": numero_2, "No sé": 99, ... }}
   }}
   Si `is_ordinal` es false, `mapping_dict` debe ser `null`.
Usa las categorías de texto exactas proporcionadas como claves en `mapping_dict`.

Ejemplos de mapeo:
- ["Totalmente en desacuerdo", "En desacuerdo", "Neutral", "De acuerdo", "Totalmente de acuerdo"] -> {{"Totalmente en desacuerdo": 1, "En desacuerdo": 2, "Neutral": 3, "De acuerdo": 4, "Totalmente de acuerdo": 5}}
- ["Muy Malo", "Malo", "Regular", "Bueno", "Excelente", "No aplica"] -> {{"Muy Malo": 1, "Malo": 2, "Regular": 3, "Bueno": 4, "Excelente": 5, "No aplica": 99}}
- ["Bajo", "Medio", "Alto"] -> {{"Bajo": 1, "Medio": 2, "Alto": 3}}
- ["Sí", "No", "Quizás"] -> {{"No": 1, "Quizás": 2, "Sí": 3}} (orden lógico, no alfabético) o bien {{"Sí": 1, "No": 0, "Quizás": 2}} si es más intuitivo para sí/no. Sé consistente.
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

# API Key Input
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

# File Uploader
st.sidebar.header("📂 Cargar Archivo")
uploaded_file = st.sidebar.file_uploader("Sube tu archivo CSV o Excel", type=["csv", "xlsx"])

# Options
st.sidebar.header("⚙️ Opciones de Procesamiento")
do_simplify_cols = st.sidebar.checkbox("Simplificar nombres de columnas (Nombres de Variable SPSS)", value=True)
do_generate_var_labels = st.sidebar.checkbox("Generar etiquetas de variables (Etiquetas de Variable SPSS)", value=True)
do_encode_ordinal = st.sidebar.checkbox("Codificar variables ordinales (con Etiquetas de Valor SPSS)", value=True)

ordinal_encoding_mode = "Crear nuevas columnas (ej. col_num)"
if do_encode_ordinal:
    ordinal_encoding_mode = st.sidebar.radio(
        "Modo de codificación ordinal:",
        ("Crear nuevas columnas (ej. VarName_num)", "Reemplazar valores en columnas existentes"),
        index=0,
        help="Elige si las columnas codificadas reemplazan a las originales o se crean como nuevas."
    )

# Session state initialization
if 'df_processed' not in st.session_state:
    st.session_state.df_processed = None
if 'spss_variable_labels' not in st.session_state: # Para las etiquetas de variable
    st.session_state.spss_variable_labels = {}
if 'spss_value_labels' not in st.session_state: # Para las etiquetas de valor
    st.session_state.spss_value_labels = {}
if 'codificaciones_ordinales_cache' not in st.session_state:
    st.session_state.codificaciones_ordinales_cache = {} # Cache para mapeos ordinales

if uploaded_file is not None:
    st.subheader("Vista Previa del Archivo Original")
    try:
        if uploaded_file.name.endswith('.csv'):
            df_original = pd.read_csv(uploaded_file, dtype=str) # Leer todo como string inicialmente
        else:
            df_original = pd.read_excel(uploaded_file, dtype=str) # Leer todo como string inicialmente
        st.dataframe(df_original.head())

        if st.sidebar.button("🚀 Procesar Datos para .sav"):
            with st.spinner("Procesando datos... Por favor espera."):
                df_processed = df_original.copy()
                log_messages = []
                
                # Para pyreadstat
                spss_variable_labels_dict = {} # { "VarName": "Variable Label" }
                spss_value_labels_dict = {}  # { "VarName_num": {1: "Label1", 2: "Label2"} }
                
                # Guardar el mapeo de nombres originales a nombres simplificados para la generación de etiquetas
                original_to_simplified_map_for_labels = {col: col for col in df_processed.columns}

                # 1. Simplificar Nombres de Columnas (Nombres de Variable SPSS)
                if do_simplify_cols:
                    log_messages.append("--- Iniciando Simplificación de Nombres de Columnas ---")
                    if openai_client:
                        original_column_names = df_processed.columns.tolist()
                        with st.spinner("Simplificando nombres de columnas (Nombres de Variable SPSS) con LLM..."):
                            simplified_names_map_llm = simplify_survey_column_names_llm(original_column_names, client=openai_client)

                        if simplified_names_map_llm:
                            log_messages.append("Mapa de nombres de variable simplificados (LLM) obtenido.")
                            final_rename_map = {}
                            temp_original_to_simplified = {}
                            for original_name in original_column_names:
                                simplified_name = simplified_names_map_llm.get(original_name)
                                if simplified_name:
                                    final_rename_map[original_name] = simplified_name
                                    temp_original_to_simplified[original_name] = simplified_name
                                else:
                                    fallback_name = basic_column_simplifier(original_name)
                                    final_rename_map[original_name] = fallback_name
                                    temp_original_to_simplified[original_name] = fallback_name
                                    log_messages.append(f"  Usando fallback para '{original_name}': '{fallback_name}'")
                            
                            # Detectar y corregir duplicados en los nombres simplificados
                            seen_names = {}
                            corrected_rename_map = {}
                            corrected_original_to_simplified = {}

                            for original, simplified in final_rename_map.items():
                                count = 1
                                new_name = simplified
                                while new_name in seen_names.values():
                                    new_name = f"{simplified}_{count}"
                                    count += 1
                                corrected_rename_map[original] = new_name
                                seen_names[original] = new_name # Guardar el nombre final
                                corrected_original_to_simplified[original] = new_name
                            
                            df_processed.rename(columns=corrected_rename_map, inplace=True)
                            original_to_simplified_map_for_labels = corrected_original_to_simplified
                            log_messages.append("Nombres de columna (variables SPSS) simplificados y aplicados.")
                        else:
                            log_messages.append("No se pudieron simplificar los nombres con LLM. Usando fallback básico para todos.")
                            fallback_rename_map = {name: basic_column_simplifier(name) for name in df_processed.columns}
                            df_processed.rename(columns=fallback_rename_map, inplace=True)
                            original_to_simplified_map_for_labels = {col: df_processed.columns[i] for i, col in enumerate(original_column_names)}

                    else:
                        log_messages.append("Cliente OpenAI no configurado. Aplicando fallback básico de renombrado.")
                        fallback_rename_map = {name: basic_column_simplifier(name) for name in df_processed.columns}
                        df_processed.rename(columns=fallback_rename_map, inplace=True)
                        original_to_simplified_map_for_labels = {col: df_processed.columns[i] for i, col in enumerate(df_original.columns)}
                    log_messages.append("--- Fin de Simplificación de Nombres ---")

                # 2. Generar Etiquetas de Variables (Etiquetas de Variable SPSS)
                if do_generate_var_labels:
                    log_messages.append("\n--- Iniciando Generación de Etiquetas de Variable ---")
                    if openai_client:
                        # Usar las preguntas originales (keys de original_to_simplified_map_for_labels)
                        # y los nombres de variable simplificados (values de original_to_simplified_map_for_labels)
                        # El prompt necesita el nombre de variable SPSS y la pregunta original.
                        # input_for_label_gen = { simplified_name: original_question for original_question, simplified_name in original_to_simplified_map_for_labels.items()}
                        # No, el prompt espera: { "nombre_variable_spss": "pregunta_original_o_descripcion" }
                        # donde las "claves" son los nombres de variables ya simplificados (las columnas actuales de df_processed)
                        # y los "valores" son las preguntas originales.
                        
                        # Construir el map: {nombre_simplificado: nombre_original}
                        # df_original.columns y original_to_simplified_map_for_labels.values() deberían estar en orden si no hubo errores.
                        # Para ser más robusto, usar original_to_simplified_map_for_labels que tiene el mapeo directo.
                        map_spss_name_to_original_question = {}
                        original_cols_list = df_original.columns.tolist() # Nombres originales en orden
                        
                        # Iterar sobre las columnas originales para mantener el contexto
                        for original_col_name in original_cols_list:
                            spss_var_name = original_to_simplified_map_for_labels.get(original_col_name)
                            if spss_var_name: # Si el nombre original fue mapeado a un nombre SPSS
                                map_spss_name_to_original_question[spss_var_name] = original_col_name
                            else: # Si por alguna razón no está en el mapa (no debería pasar con el fallback)
                                # Esto podría ocurrir si df_processed.columns se usa y no se alineó bien
                                # Usar el nombre de columna procesado como clave y como descripción
                                if original_col_name in df_processed.columns: # Si el nombre no cambió
                                     map_spss_name_to_original_question[original_col_name] = original_col_name
                                # else: podría ser un problema, pero el fallback debería cubrirlo.
                        
                        if not map_spss_name_to_original_question and df_processed is not None: # Fallback si el mapeo falló
                             map_spss_name_to_original_question = {col:col for col in df_processed.columns}


                        with st.spinner("Generando etiquetas de variable (Etiquetas SPSS) con LLM..."):
                            generated_labels = generate_variable_labels_llm(map_spss_name_to_original_question, client=openai_client)
                        
                        if generated_labels:
                            spss_variable_labels_dict = generated_labels
                            log_messages.append("Etiquetas de variable generadas por LLM.")
                        else:
                            log_messages.append("No se pudieron generar etiquetas de variable con LLM. Se usarán los nombres de variable como etiquetas.")
                            spss_variable_labels_dict = {col: col[:256] for col in df_processed.columns} # Fallback
                    else:
                        log_messages.append("Cliente OpenAI no configurado. Se usarán los nombres de variable como etiquetas.")
                        spss_variable_labels_dict = {col: col[:256] for col in df_processed.columns} # Fallback
                    log_messages.append("--- Fin de Generación de Etiquetas de Variable ---")
                else: # Si no se generan etiquetas, usar nombres de columna como etiquetas
                    spss_variable_labels_dict = {col: str(col)[:256] for col in df_processed.columns}

                # 3. Codificar Variables Ordinales (con Etiquetas de Valor SPSS)
                if do_encode_ordinal:
                    log_messages.append("\n--- Iniciando Codificación de Variables Ordinales ---")
                    if openai_client:
                        columnas_a_evaluar = list(df_processed.columns)
                        cols_to_encode_spinner = st.empty()

                        for i, columna_actual_spss_name in enumerate(columnas_a_evaluar):
                            cols_to_encode_spinner.info(f"Evaluando columna para codificación ordinal: '{columna_actual_spss_name}' ({i+1}/{len(columnas_a_evaluar)})")
                            log_messages.append(f"\nProcesando columna para codificación: '{columna_actual_spss_name}'")

                            # Necesitamos la columna original de df_original para obtener las categorías de texto
                            # Encontrar el nombre original que corresponde a columna_actual_spss_name
                            nombre_col_original_para_categorias = None
                            for orig_name, spss_name in original_to_simplified_map_for_labels.items():
                                if spss_name == columna_actual_spss_name:
                                    nombre_col_original_para_categorias = orig_name
                                    break
                            
                            if not nombre_col_original_para_categorias:
                                # Si no se encontró (ej. si la simplificación no se hizo o falló el mapeo)
                                # y estamos en modo "reemplazar", la columna en df_processed aún es texto.
                                # y si estamos en "crear nueva", la columna original en df_processed es texto.
                                if columna_actual_spss_name in df_original.columns:
                                     nombre_col_original_para_categorias = columna_actual_spss_name # Asumir que no se renombró
                                else: # Fallback extremo
                                     log_messages.append(f"  Advertencia: No se pudo encontrar la columna original para '{columna_actual_spss_name}' para obtener categorías. Omitiendo.")
                                     continue

                            # Asegurarse de que la columna exista en el df_original (para categorías) y df_processed (para datos)
                            if nombre_col_original_para_categorias not in df_original.columns:
                                log_messages.append(f"  Advertencia: Columna original '{nombre_col_original_para_categorias}' no en df_original. Omitiendo '{columna_actual_spss_name}'.")
                                continue
                            if columna_actual_spss_name not in df_processed.columns:
                                log_messages.append(f"  Advertencia: Columna procesada '{columna_actual_spss_name}' no en df_processed. Omitiendo.")
                                continue

                            # Usar df_original para obtener categorías únicas de texto
                            # Asegurarse de que sea tratada como string para .unique()
                            try:
                                categorias_unicas_series = df_original[nombre_col_original_para_categorias].dropna().astype(str)
                                categorias_unicas = sorted(list(categorias_unicas_series.unique()))
                            except Exception as e:
                                log_messages.append(f"  Error al obtener categorías únicas de '{nombre_col_original_para_categorias}': {e}. Omitiendo '{columna_actual_spss_name}'.")
                                continue

                            if not categorias_unicas:
                                log_messages.append(f"  Omitiendo '{columna_actual_spss_name}': Sin valores procesables en original '{nombre_col_original_para_categorias}'.")
                                continue

                            num_categorias = len(categorias_unicas)
                            log_messages.append(f"  '{columna_actual_spss_name}' (original: '{nombre_col_original_para_categorias}') tiene {num_categorias} categorías únicas (ordenadas): {str(categorias_unicas[:5]).encode('utf-8', 'ignore').decode('utf-8')}...")

                            if not (1 < num_categorias <= MAX_CATEGORIAS_PARA_LLM) :
                                # ... mensajes de log existentes ...
                                continue
                            
                            clave_cache = tuple(categorias_unicas)
                            sugerencia = None
                            if clave_cache in st.session_state.codificaciones_ordinales_cache:
                                sugerencia = st.session_state.codificaciones_ordinales_cache[clave_cache]
                                log_messages.append(f"  Usando codificación guardada para categorías: {clave_cache}")
                            else:
                                log_messages.append(f"  Consultando LLM para categorías de '{columna_actual_spss_name}': {clave_cache}.")
                                with st.spinner(f"Consultando LLM para '{columna_actual_spss_name}'..."):
                                     sugerencia = get_llm_mapping_suggestion(categorias_unicas, client=openai_client)
                                if sugerencia:
                                    st.session_state.codificaciones_ordinales_cache[clave_cache] = sugerencia
                                    log_messages.append(f"  Nueva codificación guardada.")

                            if sugerencia and sugerencia.get("is_ordinal") and isinstance(sugerencia.get("mapping_dict"), dict):
                                mapeo_original_llm = sugerencia["mapping_dict"] # { "texto": numero }
                                mapeo_texto_a_numero = {}
                                etiquetas_valor_para_spss = {} # { numero: "texto" }

                                for k_texto, v_numero in mapeo_original_llm.items():
                                    try:
                                        val_num = int(v_numero)
                                        mapeo_texto_a_numero[str(k_texto)] = val_num
                                        etiquetas_valor_para_spss[val_num] = str(k_texto)[:120] # Límite SPSS para etiquetas de valor
                                    except (ValueError, TypeError):
                                        log_messages.append(f"    Advertencia: No se pudo convertir '{v_numero}' a entero para '{k_texto}' en '{columna_actual_spss_name}'. Se omitirá esta categoría del mapeo.")
                                
                                if not mapeo_texto_a_numero:
                                    log_messages.append(f"  No se generó un mapeo válido para '{columna_actual_spss_name}'.")
                                    continue

                                columna_destino_spss_name = ""
                                if ordinal_encoding_mode == "Crear nuevas columnas (ej. VarName_num)":
                                    columna_destino_spss_name = f"{columna_actual_spss_name}_num"
                                    # La nueva columna toma los valores de la columna *original* (texto) y los mapea
                                    df_processed[columna_destino_spss_name] = df_original[nombre_col_original_para_categorias].astype(str).map(mapeo_texto_a_numero)
                                    log_msg_base = f"  Columna original '{nombre_col_original_para_categorias}' (como '{columna_actual_spss_name}') mapeada a NUEVA columna '{columna_destino_spss_name}'."
                                    
                                    # Añadir etiqueta de variable para la nueva columna numérica
                                    original_label = spss_variable_labels_dict.get(columna_actual_spss_name, columna_actual_spss_name)
                                    spss_variable_labels_dict[columna_destino_spss_name] = f"{original_label} (Numérico)"[:256]
                                else: # Reemplazar valores en columnas existentes
                                    columna_destino_spss_name = columna_actual_spss_name
                                    # df_processed[columna_destino_spss_name] ya existe (era texto), ahora se sobreescribe con números.
                                    # Se mapea desde la columna original que tiene el texto, en caso de que columna_actual_spss_name ya haya sido modificada
                                    df_processed[columna_destino_spss_name] = df_original[nombre_col_original_para_categorias].astype(str).map(mapeo_texto_a_numero)
                                    log_msg_base = f"  Valores en la columna '{columna_destino_spss_name}' (originada de '{nombre_col_original_para_categorias}') REEMPLAZADOS con codificación numérica."

                                # Guardar las etiquetas de valor para SPSS para la columna de destino
                                spss_value_labels_dict[columna_destino_spss_name] = etiquetas_valor_para_spss
                                
                                # Convertir a tipo numérico con soporte para NA (IntegerArray)
                                try:
                                    df_processed[columna_destino_spss_name] = pd.to_numeric(df_processed[columna_destino_spss_name], errors='coerce').astype(pd.Int64Dtype())
                                except Exception as e:
                                    log_messages.append(f"    Advertencia: No se pudo convertir '{columna_destino_spss_name}' a Int64Dtype, intentando float. Error: {e}")
                                    try:
                                        df_processed[columna_destino_spss_name] = pd.to_numeric(df_processed[columna_destino_spss_name], errors='coerce').astype(float)
                                    except Exception as e2:
                                        log_messages.append(f"    Advertencia: No se pudo convertir '{columna_destino_spss_name}' a tipo numérico. Error: {e2}")

                                log_messages.append(log_msg_base)
                                if df_processed[columna_destino_spss_name].isnull().any():
                                     # Para "reemplazar", nombre_col_original_para_categorias sigue siendo la referencia de texto
                                    unmapped_values_original = df_original[nombre_col_original_para_categorias][df_processed[columna_destino_spss_name].isnull()].unique()
                                    log_messages.append(f"    Advertencia: '{columna_destino_spss_name}' contiene nulos/NA. Valores originales de '{nombre_col_original_para_categorias}' no mapeados: {unmapped_values_original}")
                            else:
                                log_messages.append(f"  LLM (o caché) determinó que '{columna_actual_spss_name}' no es ordinal o no generó mapeo válido.")
                        cols_to_encode_spinner.empty()
                    else:
                        log_messages.append("Cliente OpenAI no configurado. No se realizará codificación ordinal con LLM.")
                    log_messages.append("--- Fin de Codificación Ordinal ---")

                st.session_state.df_processed = df_processed
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
    st.subheader("Vista Previa del DataFrame Procesado (antes de guardar a .sav)")
    st.dataframe(st.session_state.df_processed.head())

    # Preparar etiquetas de variable para pyreadstat (lista en orden de columnas)
    final_spss_variable_labels_list = []
    if st.session_state.spss_variable_labels:
        for col_name in st.session_state.df_processed.columns:
            final_spss_variable_labels_list.append(st.session_state.spss_variable_labels.get(col_name, str(col_name)[:256]))
    else: # Fallback si está vacío por alguna razón
        final_spss_variable_labels_list = [str(col)[:256] for col in st.session_state.df_processed.columns]


    #sav_buffer = io.BytesIO()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".sav") as tmp:
        temp_file_path = tmp.name
    try:
        # Nota: pyreadstat puede inferir algunos tipos, pero para asegurar, convertir a tipos que maneje bien.
        # Por ejemplo, pd.NA en Int64Dtype se convierte a system missing.
        # Columnas de solo texto se guardan como string.
        # Columnas numéricas como double.
        df_for_sav = st.session_state.df_processed.copy()

        # pyreadstat no maneja bien object dtypes que contienen mezclas si no son strings puros o numéricos puros
        # Intentar convertir columnas object a string si no son numéricas ya.
        for col in df_for_sav.columns:
            if df_for_sav[col].dtype == 'object':
                try:
                    # Intenta convertir a numérico, si falla, probablemente sea texto mixto o texto puro.
                    pd.to_numeric(df_for_sav[col])
                except ValueError:
                     df_for_sav[col] = df_for_sav[col].astype(str) # Convertir a string si no es numérico

        pyreadstat.write_sav(
            df_for_sav, 
            temp_file_path, 
            column_labels=final_spss_variable_labels_list,
            variable_value_labels=st.session_state.spss_value_labels,
            # variable_measure = {}, # Podrías añadir esto si determinas la medida (nominal, ordinal, scale)
            # missing_ranges = {}, # Para definir rangos de valores perdidos
        )

        with open(temp_file_path, "rb") as f:
            sav_bytes = f.read()

        #sav_buffer.seek(0)
        
        #output_filename_sav = "datos_procesados.sav"
        if uploaded_file:
            base, _ = os.path.splitext(uploaded_file.name)
            output_filename_sav = f"{base}_procesado.sav"

        st.download_button(
            label="📥 Descargar Archivo .sav Procesado",
            data=sav_bytes,
            file_name=output_filename_sav,
            mime="application/octet-stream", # Mime type genérico para .sav
        )





    except Exception as e:
        st.error(f"Error al generar el archivo .sav: {e}")
        st.exception(e)
    finally:
        # Eliminar el archivo temporal después de su uso
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except Exception as e_rm:
                st.warning(f"No se pudo eliminar el archivo temporal {temp_file_path}: {e_rm}")

else:
    if uploaded_file:
        st.info("Haz clic en 'Procesar Datos para .sav' en la barra lateral para comenzar.")
    else:
        st.info("Sube un archivo CSV o Excel para empezar.")

st.sidebar.markdown("---")
st.sidebar.markdown("Creado con Streamlit, OpenAI y Pyreadstat.")