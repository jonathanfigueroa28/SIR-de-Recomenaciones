#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
HITO 2 v5 - Vectorización, ChromaDB y Inferencia RAG con Control de Fuera de Dominio (Guardrails)

Proyecto de Tesis: Sistema de Recomendación Inteligente RAG + LLM
                   para Diagnóstico de Soporte TI

Evolución:
  v1: Usa métrica Euclidiana L2.
  v2: Usa métrica Coseno y corrige scores negativos.
  v3: Clasificación de Query y Umbrales Dinámicos (Técnico: 0.75, Coloquial: 0.65).
  v4: Introduce Control de Dominio (Guardrails) para filtrar y rechazar consultas.
  v5: Elimina referencias al sector minero para generalización corporativa.
"""

import os
import re
import json
import urllib.request
import pandas as pd
from dotenv import load_dotenv

# Cargar variables de entorno (.env)
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Configuración del Hito 2 v5
VERSION = "v3"  # Reutilizamos la base de datos v3
DB_PATH = f"./chroma_db_{VERSION}"
COLLECTION_NAME = f"tickets_soporte_{VERSION}"
INPUT_FILE = "base_conocimiento_v3.csv"
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
OUTPUT_REPORT = "reporte_ejecucion_hito2_v5.csv"

# ============================================================================
# LLAMADA DIRECTA A GEMINI API
# ============================================================================
def llamar_gemini(prompt):
    """Realiza una llamada HTTP POST directa a la API de Gemini (gemini-2.5-flash)."""
    if not GEMINI_API_KEY:
        return "⚠ ERROR: GEMINI_API_KEY no configurada en el archivo .env"
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}"
    headers = {"Content-Type": "application/json"}
    
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.1,  # Más determinista para clasificación y guardrails
            "maxOutputTokens": 1000
        }
    }
    
    try:
        req = urllib.request.Request(
            url, 
            data=json.dumps(payload).encode("utf-8"), 
            headers=headers, 
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=30) as response:
            res_data = json.loads(response.read().decode("utf-8"))
            text = res_data["candidates"][0]["content"]["parts"][0]["text"]
            return text.strip()
    except Exception as e:
        return f"⚠ Error al comunicarse con Gemini API: {str(e)}"

# ============================================================================
# CLASIFICACIÓN DE QUERY Y ASIGNACIÓN DE UMBRAL DINÁMICO
# ============================================================================
def clasificar_consulta_y_umbral(query):
    """Clasifica la consulta y retorna el tipo y el umbral semántico."""
    query_lower = query.lower()
    
    patrones_tecnicos = [
        r"errorcode", r"errordescription", r"pipeline", r"data factory", 
        r"id de ejecución", r"run id", r"execution id", r"exception", 
        r"failed", r"error de pipeline", r"table:", r"modelrefresh", 
        r"columns on the one side", r"many-to-one", r"primary key",
        r"timeout", r"job", r"cluster", r"databricks", r"delta lake",
        r"information\{", r"errorCode", r"errorDescription"
    ]
    
    es_tecnico = any(re.search(patron, query_lower) for patron in patrones_tecnicos)
    
    if es_tecnico:
        tipo = "Técnica (Alerta/Log Crudo)"
        umbral = 0.75
    else:
        tipo = "Coloquial / Lenguaje Natural"
        umbral = 0.65
        
    return tipo, umbral

# ============================================================================
# PREPROCESADOR DE CONSULTAS (Query Sanitization)
# ============================================================================
def sanitizar_consulta(query):
    query_limpia = re.sub(r'[a-fA-F0-9]{8}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{12}', '', query)
    query_limpia = re.sub(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z?', '', query_limpia)
    query_limpia = re.sub(r'\s+', ' ', query_limpia).strip()
    return query_limpia

# ============================================================================
# INICIALIZACIÓN DE CHROMADB (Lectura)
# ============================================================================
def inicializar_cliente_y_modelo():
    if not os.path.exists(INPUT_FILE):
        print(f"  ✗ ERROR: No se encontró el archivo de entrada '{INPUT_FILE}'.")
        return None, None
    
    try:
        from sentence_transformers import SentenceTransformer
        embedding_model = SentenceTransformer(MODEL_NAME)
    except Exception as e:
        print(f"  ✗ ERROR: No se pudo cargar SentenceTransformer: {e}")
        return None, None

    try:
        import chromadb
        chroma_client = chromadb.PersistentClient(path=DB_PATH)
        collection = chroma_client.get_collection(name=COLLECTION_NAME)
    except Exception as e:
        print(f"  ✗ ERROR: No se pudo inicializar ChromaDB: {e}")
        return None, None
        
    return collection, embedding_model

# ============================================================================
# CONSULTA Y MOTOR RAG + FALLBACK CON UMBRAL DINÁMICO Y GUARDRAILS
# ============================================================================
def consultar_sistema(query, collection, embedding_model):
    """
    Procesa la query clasificándola, consultando ChromaDB y llamando a Gemini.
    Implementa el guardrail para rechazar temas fuera del dominio de soporte técnico de datos.
    """
    print("\n" + "-"*72)
    print(f"  CONSULTA: {query.strip()[:100]}...")
    print("-"*72)
    
    # 1. Clasificar consulta y obtener umbral dinámico
    tipo_consulta, umbral_dinamico = clasificar_consulta_y_umbral(query)
    print(f"  ✓ Clasificación Preliminar: {tipo_consulta}")
    print(f"  ✓ Umbral Semántico Asignado: {umbral_dinamico:.2f}")
    
    # 2. Sanitizar query
    query_sanitizada = sanitizar_consulta(query)
    
    # 3. Generar embedding de búsqueda
    query_vector = embedding_model.encode(query_sanitizada).tolist()
    
    # 4. Buscar en ChromaDB
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=1
    )
    
    similitud = 0.0
    best_id = "N/A"
    best_doc = ""
    best_meta = {}
    
    if results and results['ids'] and len(results['ids'][0]) > 0:
        best_id = results['ids'][0][0]
        best_doc = results['documents'][0][0]
        best_meta = results['metadatas'][0][0]
        best_distance = results['distances'][0][0]
        similitud = 1.0 - best_distance
        
    print(f"  ✓ Ticket similar recuperado: {best_id} (Similitud Coseno: {similitud:.3f})")
    
    # 5. Lógica de Enrutamiento con Guardrail
    prompt = ""
    
    if similitud >= umbral_dinamico:
        # Modo RAG Activo (superó umbral)
        # Pero aplicamos un guardrail liviano por si acaso la pregunta es de otro tipo pero el embedding de ChromaDB coincide con algo (falso positivo)
        print(f"\n  🟢 [MODO RAG ACTIVO] Supera el umbral dinámico ({similitud:.3f} >= {umbral_dinamico:.3f}).")
        
        prompt = f"""
Eres un asistente de soporte TI experto para una plataforma analítica corporativa. 
El ingeniero de soporte ha reportado la siguiente consulta o alerta de error:

"{query}"

Hemos buscado en la base de datos de lecciones aprendidas y tickets históricos, encontrando la siguiente coincidencia con similitud semántica alta (ID: {best_id}, Categoría: {best_meta.get('categoria')}, SubCategoría: {best_meta.get('subcategoria')}):

--------------------------------------------------
ANTECEDENTE HISTÓRICO:
{best_doc}
--------------------------------------------------

REGLA DE SEGURIDAD DE DOMINIO (GUARDRAIL):
Antes de redactar la recomendación, valida si el texto del usuario corresponde de verdad a soporte de la plataforma de datos corporativa. Si el texto del usuario es totalmente ajeno a soporte analítico (por ejemplo, preguntas generales, matemáticas, política, chistes, etc.), debes rechazarlo estrictamente respondiendo EXACTAMENTE con el siguiente mensaje de rechazo y nada más:
"Disculpe las molestias. Como asistente inteligente de recomendación, mi alcance está limitado estrictamente a responder preguntas e incidencias sobre la plataforma de analítica de datos (Azure Data Factory, Databricks, Power BI, SQL Server, SAP, etc.). No tengo autorización para responder consultas fuera de este dominio técnico."

Si la consulta es válida, elabora la recomendación estructurada para el ingeniero:
1. Confirma que se identificó un caso previo similar en la base de conocimiento (mencionando el ID {best_id}).
2. Explica el DIAGNÓSTICO del fallo de forma clara y sintetizada.
3. Detalla la SOLUCIÓN APLICADA paso a paso.
4. Redacta en formato Markdown con español técnico y viñetas.
"""
    else:
        # Modo Fallback (no superó umbral) - Aplicamos Guardrail estricto de Gemini
        print(f"\n  🟡 [MODO FALLBACK ACTIVO] No supera el umbral dinámico ({similitud:.3f} < {umbral_dinamico:.3f}).")
        
        prompt = f"""
Eres un asistente de soporte TI experto para una plataforma analítica corporativa.
El ingeniero de soporte ha reportado la siguiente consulta o requerimiento:

"{query}"

REGLA DE SEGURIDAD DE DOMINIO (GUARDRAIL CRÍTICO):
Evalúa si la consulta está relacionada con incidentes, soporte técnico, consultas de bases de datos, Power BI, Azure Data Factory, Databricks, modelado de datos, ETLs o buenas prácticas en flujos de datos analíticos.

Si la consulta NO TIENE ABSOLUTAMENTE NADA QUE VER con este dominio técnico (por ejemplo, preguntas de matemáticas generales como "cuánto es la raíz cuadrada de 100", chistes, recetas de cocina, geografía, política, historia, programación de frontend web CSS/HTML, o configuración de hardware/teléfonos personales ajenos a la plataforma), debes rechazarla respondiendo ESTRICTAMENTE con este mensaje y ningún otro texto adicional:
"Disculpe las molestias. Como asistente inteligente de recomendación, mi alcance está limitado estrictamente a responder preguntas e incidencias sobre la plataforma de analítica de datos (Azure Data Factory, Databricks, Power BI, SQL Server, SAP, etc.). No tengo autorización para responder consultas fuera de este dominio técnico."

Si la consulta SÍ es una duda o problema sobre la plataforma analítica de datos pero no cuenta con antecedentes cerrados en nuestra base vectorial, debes generar una recomendación experta basada en tus conocimientos generales y mejores prácticas de soporte de TI.
En este caso, debes:
1. Advertir de manera clara y cordial al inicio que la sugerencia se basa en recomendaciones técnicas generales de mejores prácticas al no existir un antecedente idéntico cerrado en el sistema de tickets.
2. Analizar el error o requerimiento (identificar si es de Azure Data Factory, Power BI, Databricks, base de datos, etc.).
3. Describir la causa probable y listar los pasos detallados de diagnóstico o solución general.
4. Redactar en formato Markdown estructurado en español técnico y con viñetas.
"""

    recomendacion = llamar_gemini(prompt)
    
    print("\n" + "="*72)
    print("  RECOMENDACIÓN DEL SISTEMA INTELIGENTE (V5):")
    print("="*72)
    print(recomendacion)
    print("="*72 + "\n")
    
    # Identificar el modo de inferencia final
    if "alcance está limitado estrictamente" in recomendacion:
        modo_final = "Rechazado (Fuera de Dominio)"
    elif similitud >= umbral_dinamico:
        modo_final = "RAG Activo"
    else:
        modo_final = "Fallback (Soporte General)"
        
    return similitud, best_id, recomendacion, tipo_consulta, umbral_dinamico, modo_final

# ============================================================================
# ENSAYO TÉCNICO V4
# ============================================================================
def ejecutar_ensayo_tecnico(collection, embedding_model):
    print("\n" + "="*72)
    print("  [ENSAYO TÉCNICO V5] VALIDACIÓN DE GUARDRAILS Y UMBRALES DINÁMICOS")
    print("="*72)
    
    pruebas = [
        {
            "id": 1,
            "caso": "1. Alerta Técnica Cruda (Dentro de Dominio - RAG Exitoso)",
            "query": 'Error de Pipeline adf-analitica-prd Pipeline: GobiernoDelDato_Dinamico Id: 31134b4c. errorCode: ModelRefresh_ShortMessage_ProcessingError, Column FECHA DE PUBLICACION in Table Comunicacion contains blank values.'
        },
        {
            "id": 2,
            "caso": "2. Requerimiento Coloquial (Dentro de Dominio - Fallback de Soporte General)",
            "query": 'Necesito optimizar una consulta pesada en una tabla para acelerar la carga de un reporte de ventas en Power BI.'
        },
        {
            "id": 3,
            "caso": "3. Fuera de Dominio A (Pregunta Matemática - Rechazo)",
            "query": '¿Cuánto es la raíz cuadrada de 100?'
        },
        {
            "id": 4,
            "caso": "4. Fuera de Dominio B (Pregunta Política - Rechazo)",
            "query": '¿Quién es el presidente actual de Francia?'
        }
    ]
    
    resultados = []
    
    for p in pruebas:
        print(f"\n>>> EJECUTANDO CASO DE PRUEBA: {p['caso']}")
        sim, best_id, rec, tipo, umb, modo_final = consultar_sistema(p['query'], collection, embedding_model)
        
        resultados.append({
            "ID": p["id"],
            "Caso": p["caso"][:35] + "...",
            "Similitud": round(sim, 3),
            "Umbral_Dinamico": umb,
            "Modo_Inferencia": modo_final,
            "Muestra_Respuesta": rec[:100] + ("..." if len(rec) > 100 else "")
        })
        
    # Guardar reporte
    df_res = pd.DataFrame(resultados)
    df_res.to_csv(OUTPUT_REPORT, index=False, encoding="utf-8-sig")
    
    print("\n" + "="*72)
    print("  RESUMEN DEL ENSAYO TÉCNICO HITO 2 V5")
    print("="*72)
    print(df_res.to_string(index=False))
    print("="*72 + "\n")

if __name__ == "__main__":
    collection, embedding_model = inicializar_cliente_y_modelo()
    if collection and embedding_model:
        ejecutar_ensayo_tecnico(collection, embedding_model)
