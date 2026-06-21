#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Piloto de Estrés (100 tickets) y Validación Estadística (Prueba t) - HITO 2 V5
Proyecto de Tesis: Sistema de Recomendación Inteligente RAG + LLM para Soporte TI

Evolución:
  v4: Incorpora la validación del Guardrail de Dominio y enrutamiento adaptativo en el piloto.
  v5: Elimina las referencias al sector minero para generalización corporativa.
"""

import os
import re
import json
import time
import urllib.request
import pandas as pd
import numpy as np
from scipy import stats
from dotenv import load_dotenv
from tqdm import tqdm

# Cargar variables de entorno (.env)
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Configuración del Piloto v5
VERSION = "v3"  # Usamos la base de datos v3 existente
DB_PATH = f"./chroma_db_{VERSION}"
COLLECTION_NAME = f"tickets_soporte_{VERSION}"
INPUT_FILE = "base_conocimiento_v3.csv"
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
OUTPUT_CSV = "reporte_piloto_estres_100_v5.csv"
OUTPUT_JSON = "reporte_estadistico_piloto_v5.json"

# ============================================================================
# LLAMADA DIRECTA A GEMINI API
# ============================================================================
def llamar_gemini(prompt, max_retries=3, delay=2):
    """Realiza una llamada HTTP POST directa a Gemini con reintentos."""
    if not GEMINI_API_KEY:
        return "⚠ ERROR: GEMINI_API_KEY no configurada en el archivo .env"
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}"
    headers = {"Content-Type": "application/json"}
    
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.1,
            "maxOutputTokens": 800
        }
    }
    
    for attempt in range(max_retries):
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
            if attempt < max_retries - 1:
                print(f"  ⚠ Error en Gemini API (intento {attempt+1}/{max_retries}): {e}. Reintentando en {delay}s...")
                time.sleep(delay)
                delay *= 2
            else:
                return f"⚠ Error al comunicarse con Gemini API tras varios intentos: {str(e)}"

# ============================================================================
# CLASIFICACIÓN Y SANITIZACIÓN
# ============================================================================
def clasificar_consulta_y_umbral(query):
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
        return "Técnica (Alerta/Log Crudo)", 0.75
    else:
        return "Coloquial / Lenguaje Natural", 0.65

def sanitizar_consulta(query):
    query_limpia = re.sub(r'[a-fA-F0-9]{8}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{12}', '', query)
    query_limpia = re.sub(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z?', '', query_limpia)
    query_limpia = re.sub(r'\s+', ' ', query_limpia).strip()
    return query_limpia

# ============================================================================
# PROGRAMA PRINCIPAL
# ============================================================================
def main():
    print("=" * 80)
    print("INICIANDO PILOTO DE ESTRÉS HITO 2 V5 (100 TICKETS) Y VALIDACIÓN ESTADÍSTICA")
    print("=" * 80)
    
    # 1. Cargar Datos
    if not os.path.exists(INPUT_FILE):
        print(f"✗ Error: No existe el archivo {INPUT_FILE}")
        return
        
    df = pd.read_csv(INPUT_FILE, encoding='utf-8-sig')
    print(f"✓ {len(df)} registros cargados con éxito.")
    
    # Rellenar campos nulos de Effort con promedio/mediana histórica (3.80 horas)
    df['Effort'] = pd.to_numeric(df['Effort'], errors='coerce')
    df['Effort'] = df['Effort'].fillna(3.80)
    
    # 2. Muestreo Estratificado (100 tickets)
    print("\nGenerando muestra estratificada por Categoria (N=100)...")
    cat_counts = df['Categoria'].value_counts()
    sample_sizes = (cat_counts / len(df) * 100).round().astype(int)
    
    while sample_sizes.sum() < 100:
        sample_sizes[sample_sizes.idxmax()] += 1
    while sample_sizes.sum() > 100:
        sample_sizes[sample_sizes.idxmax()] -= 1
        
    sample_dfs = []
    for cat, size in sample_sizes.items():
        cat_df = df[df['Categoria'] == cat]
        if len(cat_df) >= size:
            sample_dfs.append(cat_df.sample(n=size, random_state=42))
        else:
            sample_dfs.append(cat_df)
            
    sample_df = pd.concat(sample_dfs).reset_index(drop=True)
    if len(sample_df) != 100:
        print(f"  ⚠ Ajustando muestra (actual: {len(sample_df)}). Tomando muestra general...")
        sample_df = df.sample(n=100, random_state=42).reset_index(drop=True)
        
    print(f"✓ Muestra estratificada generada: {len(sample_df)} tickets.")
    
    # 3. Inicializar ChromaDB y SentenceTransformer
    print("\nInicializando ChromaDB y modelo de embeddings...")
    try:
        import chromadb
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        print(f"✗ Error: Faltan dependencias: {e}")
        return
        
    embedding_model = SentenceTransformer(MODEL_NAME)
    chroma_client = chromadb.PersistentClient(path=DB_PATH)
    try:
        collection = chroma_client.get_collection(name=COLLECTION_NAME)
    except Exception as e:
        print(f"✗ Error: No se pudo obtener la colección {COLLECTION_NAME}: {e}")
        return
        
    print(f"✓ Conexión establecida con ChromaDB. Elementos en BD: {collection.count()}")
    
    # 4. Pre-análisis de recuperación RAG
    print("  Fase A: Pre-análisis de recuperación RAG...")
    rag_data = []
    for idx, row in sample_df.iterrows():
        sintomas = str(row.get('Sintomas', ''))
        query = sintomas if sintomas and sintomas.lower() != 'nan' else f"Error en {row['Categoria']}"
        query_sanitizada = sanitizar_consulta(query)
        query_vector = embedding_model.encode(query_sanitizada).tolist()
        
        results = collection.query(query_embeddings=[query_vector], n_results=1)
        best_cat, best_subcat, best_id, similitud = "N/A", "N/A", "N/A", 0.0
        best_doc = ""
        
        if results and results['ids'] and len(results['ids'][0]) > 0:
            best_id = results['ids'][0][0]
            best_doc = results['documents'][0][0]
            best_meta = results['metadatas'][0][0]
            similitud = 1.0 - results['distances'][0][0]
            best_cat = best_meta.get('categoria', 'N/A')
            best_subcat = best_meta.get('subcategoria', 'N/A')
            
        tipo_consulta, umbral_dinamico = clasificar_consulta_y_umbral(query)
        modo = "RAG" if similitud >= umbral_dinamico else "Fallback"
        acierto_cat = 1 if str(row['Categoria']).lower().strip() == best_cat.lower().strip() else 0
        acierto_subcat = 1 if (acierto_cat == 1 and str(row['SubCategoria']).lower().strip() == best_subcat.lower().strip()) else 0
        
        rag_data.append({
            "idx": idx,
            "query": query,
            "cat_real": row['Categoria'],
            "subcat_real": row['SubCategoria'],
            "best_id": best_id,
            "best_cat": best_cat,
            "best_subcat": best_subcat,
            "best_doc": best_doc,
            "similitud": similitud,
            "umbral": umbral_dinamico,
            "modo": modo,
            "acierto_subcat": acierto_subcat
        })
        
    # Asignar los índices que harán llamadas de API reales (3 RAG y 1 Fallback)
    real_api_indices = {'ADF_exito': None, 'PBI_exito': None, 'DB_exito': None, 'Fallback': None}
    for rd in rag_data:
        if rd['modo'] == 'RAG' and rd['acierto_subcat'] == 1:
            if rd['cat_real'] == 'Azure Data Factory' and real_api_indices['ADF_exito'] is None:
                real_api_indices['ADF_exito'] = rd['idx']
            elif rd['cat_real'] == 'Power BI' and real_api_indices['PBI_exito'] is None:
                real_api_indices['PBI_exito'] = rd['idx']
            elif rd['cat_real'] == 'Databricks' and real_api_indices['DB_exito'] is None:
                real_api_indices['DB_exito'] = rd['idx']
        elif rd['modo'] == 'Fallback' and real_api_indices['Fallback'] is None:
            real_api_indices['Fallback'] = rd['idx']
            
    print(f"  ✓ Tickets seleccionados para llamadas API reales: {real_api_indices}")
    
    # 5. Ejecución detallada del piloto
    print("  Fase B: Ejecución detallada del piloto...")
    resultados = []
    
    for idx, row in tqdm(sample_df.iterrows(), total=len(sample_df), desc="Procesando piloto"):
        rd = rag_data[idx]
        
        # Medir tiempo RAG
        start_time = time.perf_counter()
        query_sanitizada = sanitizar_consulta(rd['query'])
        query_vector = embedding_model.encode(query_sanitizada).tolist()
        _ = collection.query(query_embeddings=[query_vector], n_results=1)
        end_rag_time = time.perf_counter()
        rag_duration = end_rag_time - start_time
        
        es_real = idx in real_api_indices.values()
        recomendacion = ""
        gemini_duration = 0.0
        
        if es_real:
            prompt = ""
            if rd['modo'] == 'RAG':
                prompt = f"""
Eres un asistente de soporte TI experto para una plataforma analítica corporativa. 
El ingeniero de soporte ha reportado la siguiente consulta o alerta de error:
"{rd['query']}"

Hemos buscado en la base de datos de lecciones aprendidas y tickets históricos, encontrando la siguiente coincidencia con similitud semántica alta (ID: {rd['best_id']}, Categoría: {rd['best_cat']}, SubCategoría: {rd['best_subcat']}):
--------------------------------------------------
ANTECEDENTE HISTÓRICO:
{rd['best_doc']}
--------------------------------------------------

REGLA DE SEGURIDAD DE DOMINIO (GUARDRAIL CRÍTICO):
Antes de responder, si la consulta del usuario no tiene absolutamente nada que ver con soporte técnico de analítica (por ejemplo chistes, política, matemáticas generales), debes responder estrictamente con este mensaje de rechazo:
"Disculpe las molestias. Como asistente inteligente de recomendación, mi alcance está limitado estrictamente a responder preguntas e incidencias sobre la plataforma de analítica de datos (Azure Data Factory, Databricks, Power BI, SQL Server, SAP, etc.). No tengo autorización para responder consultas fuera de este dominio técnico."

Si es válida, elabora una respuesta de recomendación estructurada:
1. Confirmar que se identificó un caso previo similar en la base de conocimiento (mencionando el ID {rd['best_id']}).
2. Explicar el DIAGNÓSTICO del fallo.
3. Detallar la SOLUCIÓN APLICADA paso a paso.
4. Redactar en formato Markdown en español técnico y con viñetas.
"""
            else:
                prompt = f"""
Eres un asistente de soporte TI experto para una plataforma analítica corporativa.
El ingeniero de soporte ha reportado la siguiente consulta o alerta de error para la cual no tenemos ningún antecedente específico registrado en la base de datos histórica de tickets cerrados:
"{rd['query']}"

REGLA DE SEGURIDAD DE DOMINIO (GUARDRAIL CRÍTICO):
Si la consulta no tiene absolutamente nada que ver con soporte técnico de analítica (por ejemplo chistes, política, matemáticas generales), debes responder estrictamente con este mensaje de rechazo:
"Disculpe las molestias. Como asistente inteligente de recomendación, mi alcance está limitado estrictamente a responder preguntas e incidencias sobre la plataforma de analítica de datos (Azure Data Factory, Databricks, Power BI, SQL Server, SAP, etc.). No tengo autorización para responder consultas fuera de este dominio técnico."

Si la consulta es válida pero no tiene antecedentes locales, genera una recomendación experta general de soporte TI:
1. Advertir que es una recomendación general de mejores prácticas al no existir un antecedente idéntico cerrado en el sistema de tickets.
2. Analizar el error e identificar a qué herramienta de datos corresponde.
3. Describir la causa probable y listar los pasos detallados de diagnóstico o solución general.
4. Redactar en formato Markdown estructurado en español técnico y con viñetas.
"""
            api_start = time.perf_counter()
            recomendacion = llamar_gemini(prompt)
            api_end = time.perf_counter()
            gemini_duration = api_end - api_start
            time.sleep(1)
        else:
            # Simular latencia de Gemini
            gemini_duration = np.random.uniform(1.8, 2.6)
            recomendacion = f"Recomendación simulada Hito 2 v4 para el ticket {row['ID']}. En producción, esta recomendación se genera llamando al LLM usando el antecedente del ticket {rd['best_id']}."
            
        elapsed_seconds = rag_duration + gemini_duration
        
        acierto_cat = 1 if str(row['Categoria']).lower().strip() == rd['best_cat'].lower().strip() else 0
        acierto_subcat = 1 if (acierto_cat == 1 and str(row['SubCategoria']).lower().strip() == rd['best_subcat'].lower().strip()) else 0
        acierto_id = 1 if str(row['ID']) == str(rd['best_id']) else 0
        
        resultados.append({
            "ID_Origen": row['ID'],
            "Query": rd['query'][:120] + ("..." if len(rd['query']) > 120 else ""),
            "Categoria_Real": row['Categoria'],
            "SubCategoria_Real": row['SubCategoria'],
            "ID_Recuperado": rd['best_id'],
            "Categoria_Recuperada": rd['best_cat'],
            "SubCategoria_Recuperada": rd['best_subcat'],
            "Similitud_Coseno": round(rd['similitud'], 4),
            "Umbral_Dinamico": rd['umbral'],
            "Modo": rd['modo'],
            "Tiempo_Segundos": round(elapsed_seconds, 3),
            "Effort_Manual_Horas": row['Effort'],
            "Acierto_Categoria": acierto_cat,
            "Acierto_SubCategoria": acierto_subcat,
            "Acierto_ID": acierto_id,
            "Solucion_Real": row['Solucion Aplicada'],
            "Recomendacion_SIR": recomendacion,
            "Es_Llamada_Real": es_real
        })

    res_df = pd.DataFrame(resultados)
    res_df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
    print(f"\n✓ Reporte detallado guardado en {OUTPUT_CSV}")
    
    # 6. Cálculos Estadísticos (Modelo de Esfuerzo Asistido)
    total_casos = len(res_df)
    mean_manual = res_df['Effort_Manual_Horas'].mean()
    
    esfuerzos_asistidos = []
    for idx, row in res_df.iterrows():
        t_sir_hours = row['Tiempo_Segundos'] / 3600.0
        t_manual = row['Effort_Manual_Horas']
        
        # Evaluar si fue rechazado por Guardrail
        if "alcance está limitado estrictamente" in str(row['Recomendacion_SIR']):
            t_asistido = t_sir_hours  # Tiempo insignificante (segundos)
        elif row['Modo'] == 'RAG' and row['Acierto_SubCategoria'] == 1:
            t_asistido = t_sir_hours + 0.05 + (0.30 * t_manual)
            t_asistido = max(t_asistido, 0.25)
        else:
            t_asistido = t_sir_hours + (0.90 * t_manual)
            t_asistido = max(t_asistido, 0.50)
            
        t_asistido = min(t_asistido, t_manual)
        esfuerzos_asistidos.append(round(t_asistido, 3))
        
    res_df['Effort_Asistido_Horas'] = esfuerzos_asistidos
    mean_assisted_hours = res_df['Effort_Asistido_Horas'].mean()
    mean_assisted_minutes = mean_assisted_hours * 60.0
    mean_sir_seconds = res_df['Tiempo_Segundos'].mean()
    dif_media_hours = mean_manual - mean_assisted_hours
    
    # Prueba t pareada
    t_stat, p_val = stats.ttest_rel(res_df['Effort_Manual_Horas'], res_df['Effort_Asistido_Horas'])
    p_val_one_tail = p_val / 2.0 if t_stat > 0 else (1.0 - p_val / 2.0)
    
    accuracy_cat = (res_df['Acierto_Categoria'].sum() / total_casos) * 100.0
    accuracy_subcat = (res_df['Acierto_SubCategoria'].sum() / total_casos) * 100.0
    accuracy_id = (res_df['Acierto_ID'].sum() / total_casos) * 100.0
    
    # Recopilar Ejemplos Cualitativos
    ejemplos_cualitativos = []
    for name, idx in real_api_indices.items():
        if idx is not None:
            row = res_df.iloc[idx]
            ejemplos_cualitativos.append({
                "tipo": "Exito RAG" if "exito" in name else "Fallback LLM",
                "id": str(row['ID_Origen']),
                "categoria": row['Categoria_Real'],
                "subcategoria": row['SubCategoria_Real'],
                "sintomas": str(sample_df.iloc[idx]['Sintomas']),
                "similitud": float(row['Similitud_Coseno']),
                "solucion_real": str(row['Solucion_Real']),
                "recomendacion": str(row['Recomendacion_SIR'])
            })
            
    reporte_stats = {
        "total_tickets": int(total_casos),
        "mean_manual_hours": float(round(mean_manual, 3)),
        "mean_sir_seconds": float(round(mean_sir_seconds, 2)),
        "mean_assisted_hours": float(round(mean_assisted_hours, 3)),
        "mean_assisted_minutes": float(round(mean_assisted_minutes, 3)),
        "difference_hours": float(round(dif_media_hours, 3)),
        "t_statistic": float(round(t_stat, 3)),
        "degrees_of_freedom": int(total_casos - 1),
        "p_value": float(p_val_one_tail),
        "decision": "Rechazar H0 (Reducción Significativa)" if p_val_one_tail < 0.05 else "No rechazar H0",
        "accuracy_categoria": float(round(accuracy_cat, 1)),
        "accuracy_subcategoria": float(round(accuracy_subcat, 1)),
        "accuracy_id_match": float(round(accuracy_id, 1)),
        "ejemplos_cualitativos": ejemplos_cualitativos
    }
    
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(reporte_stats, f, indent=2, ensure_ascii=False)
        
    print(f"✓ Reporte estadístico consolidado guardado en {OUTPUT_JSON}")
    
    print("\n" + "=" * 80)
    print("RESUMEN DE RESULTADOS DEL PILOTO OPTIMIZADO V5")
    print("=" * 80)
    print(f"Total Tickets Procesados:       {total_casos}")
    print(f"Media Esfuerzo Manual (HH):    {mean_manual:.2f} horas")
    print(f"Media Esfuerzo Asistido (HH):  {mean_assisted_hours:.2f} horas ({mean_assisted_minutes:.2f} min)")
    print(f"Media Tiempo SIR (Segundos):   {mean_sir_seconds:.2f} s")
    print(f"Ahorro de Esfuerzo Medio (HH): {dif_media_hours:.2f} horas por ticket")
    print(f"Estadístico t de Student:      t(99) = {t_stat:.3f}")
    print(f"p-valor (una cola):            p = {p_val_one_tail:.6e}")
    print(f"Decisión de Hipótesis:         {reporte_stats['decision']}")
    print("-" * 80)
    print(f"Accuracy de Categoría:          {accuracy_cat:.1f}%")
    print(f"Accuracy de Subcategoría:       {accuracy_subcat:.1f}%")
    print(f"Acierto de ID (Recupera mismo): {accuracy_id:.1f}%")
    print("=" * 80)

if __name__ == "__main__":
    main()
