#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
HITO 1 v3 - Sistematización, Limpieza, Anonimización y Deduplicación
             de la Base de Conocimiento para RAG

Proyecto de Tesis: Sistema de Recomendación Inteligente RAG + LLM
               para Diagnóstico de Soporte TI

Evolución:
  v1 (limpieza_hito1.py):    Limpieza HTML + re-categorización por keywords
  v2 (limpieza_hito1_v2.py): Limpieza HTML + re-categorización por Gemini API
  v3 (este script):          v1/v2 + Anonimización + Eliminación de redundancia
                             + Formateo optimizado para RAG

Fuente: Toma como input el output de v1 o v2 (auto-detecta)
        Fallback: Soporte-AnaliticaDescriptiva - Completado.csv

Archivos generados:
  - dataset_limpio_hito1_v3.csv           → Dataset completo (anonimizado)
  - base_conocimiento_v3.csv              → Base para RAG (sin redundancia)
  - reporte_metricas_hito1_v3.txt         → Reporte de métricas
  - reporte_anonimizacion_v3.csv          → Detalle de anonimización
  - reporte_redundancia_v3.csv            → Tickets eliminados por redundancia
"""

import pandas as pd
import re
import os
import hashlib
from datetime import datetime
from collections import defaultdict
from difflib import SequenceMatcher

# ============================================================================
# CONFIGURACIÓN
# ============================================================================
VERSION = "v3"

# Auto-detección de input: prefiere v2, luego v1, luego original
INPUT_CANDIDATES = [
    ("dataset_limpio_hito1_v2.csv", "v2 (Gemini API)"),
    ("dataset_limpio_hito1.csv", "v1 (keywords)"),
]

OUTPUT_FILE = f"dataset_limpio_hito1_{VERSION}.csv"
OUTPUT_KB_FILE = f"base_conocimiento_{VERSION}.csv"
REPORT_FILE = f"reporte_metricas_hito1_{VERSION}.txt"
ANON_REPORT_FILE = f"reporte_anonimizacion_{VERSION}.csv"
REDUNDANCIA_REPORT_FILE = f"reporte_redundancia_{VERSION}.csv"

COLUMNAS_TEXTO = [
    'Sintomas', 'Diagnostico', 'Solucion Aplicada', 'Description', 'Comentario'
]

# Umbral de similitud para considerar tickets redundantes (0.0 - 1.0)
SIMILITUD_UMBRAL = 0.85

# ============================================================================
# CATÁLOGO OFICIAL (para validación)
# ============================================================================
CATALOGO_OFICIAL = {
    'Power BI': [
        'Publicación / Deploy', 'Conectividad', 'Visualizaciones',
        'Performance / Optimización', 'Seguridad / RLS', 'Modelado de datos',
        'Actualización de dataset', 'Inconsistencia de Datos', 'Requerimiento',
    ],
    'Azure Data Factory': [
        'Pipelines', 'Linked Services', 'Triggers', 'Data Flows',
        'Accesos / Autenticación', 'Requerimiento',
        'Error de Tipo de Dato', 'Error de Time out',
    ],
    'Databricks': [
        'Conectividad', 'Clusters / Jobs', 'Performance',
        'Unity Catalog / Seguridad', 'Delta Lake / Lakehouse',
        'Integración con Power BI', 'Requerimiento',
    ],
    'Azure SQL Database': [
        'Consultas / Performance', 'ETL Manual', 'Integración',
        'Seguridad / Permisos', 'Modelado lógico / físico',
    ],
    'Azure Analysis Services': [
        'Conectividad', 'Actualización de modelos', 'Seguridad',
    ],
    'Fabric / OneLake': [
        'Espacios de trabajo', 'Lakehouse', 'Data Activator / Copilot',
    ],
    'General / Buenas prácticas': [
        'Documentación / Plantillas', 'Automatización', 'Control de versiones',
        'Recomendaciones de arquitectura', 'Mejora de proceso',
    ],
    'Gobierno del Dato': [
        'Calidad de Datos', 'Catálogo / Linaje',
    ],
    'Web': [
        'Falla de sincronización', 'Inconsistencia de Datos', 'Seguridad / Accesos',
    ],
    'RFC SAP': ['Inconsistencia de Datos'],
    'Soporte Analitico': ['Derivación del caso', 'Requerimiento'],
    'SAP': [
        'Falla en el IR-Minsur', 'Integración', 'Conectividad',
        'Inconsistencia de Datos', 'Requerimiento',
    ],
}


# ============================================================================
# MÓDULO 1: ANONIMIZACIÓN
# ============================================================================

def extraer_nombres_del_dataset(df):
    """
    Extrae nombres de personas del campo 'Assigned To' para usarlos
    en la anonimización de los campos de texto.
    Formato típico: 'Nombre Apellido <email@dom.com>'
    """
    nombres = set()
    if 'Assigned To' not in df.columns:
        return nombres

    for val in df['Assigned To'].dropna().unique():
        val_str = str(val).strip()
        # Extraer nombre (antes del <email>)
        match = re.match(r'^([^<]+)', val_str)
        if match:
            nombre_completo = match.group(1).strip()
            if nombre_completo:
                nombres.add(nombre_completo)
                # También agregar variantes parciales
                partes = nombre_completo.split()
                if len(partes) >= 2:
                    nombres.add(partes[0])  # Nombre solo
                    nombres.add(f"{partes[0]} {partes[1]}")  # Nombre + Apellido

    # Extraer nombres de otras columnas de contexto
    for col in ['Empresa Area']:
        if col in df.columns:
            for val in df[col].dropna().unique():
                # No agregar nombres de áreas como nombres de personas
                pass

    return nombres


def extraer_emails_del_dataset(df):
    """Extrae todos los emails encontrados en los campos de texto."""
    emails = set()
    for col in COLUMNAS_TEXTO + ['Assigned To']:
        if col not in df.columns:
            continue
        for val in df[col].dropna():
            found = re.findall(r'[\w.+-]+@[\w-]+\.[\w.]+', str(val))
            emails.update(found)
    return emails


def anonimizar_texto(texto, nombres_personas, emails_conocidos):
    """
    Anonimiza un campo de texto reemplazando:
    - Emails → [EMAIL]
    - Nombres de personas conocidas → [USUARIO]
    - Rutas de archivo locales → [RUTA_LOCAL]
    - Direcciones IP → [IP]
    - Patrones de cabecera de email (De:, Para:, CC:, Enviado:)

    Retorna: (texto_anonimizado, lista_de_cambios)
    """
    if not texto or not isinstance(texto, str) or not texto.strip():
        return texto, []

    cambios = []
    texto_original = texto

    # 1. Anonimizar bloques de cabecera de email completos
    # Patrón: "De: Nombre <email>\nEnviado: fecha\nPara: Nombre <email>"
    patron_cabecera = (
        r'(?:-----?Mensaje original-----?)?'
        r'(?:\s*De:\s*[^\n]+\n)'
        r'(?:\s*Enviado(?:\s+el)?:\s*[^\n]+\n)?'
        r'(?:\s*Para:\s*[^\n]+\n)?'
        r'(?:\s*CC:\s*[^\n]+\n)?'
        r'(?:\s*Asunto:\s*[^\n]+\n)?'
    )
    matches_cabecera = list(re.finditer(patron_cabecera, texto, re.IGNORECASE))
    for match in reversed(matches_cabecera):
        cambios.append(('cabecera_email', match.group()[:80]))
        texto = texto[:match.start()] + '[CABECERA_EMAIL_REDACTADA]\n' + texto[match.end():]

    # 2. Emails individuales restantes → [EMAIL]
    emails_encontrados = re.findall(r'[\w.+-]+@[\w-]+\.[\w.]+', texto)
    for email in set(emails_encontrados):
        if email in texto:
            cambios.append(('email', email))
            texto = texto.replace(email, '[EMAIL]')

    # 3. Rutas de archivo Windows → [RUTA_LOCAL]
    rutas = re.findall(r'[A-Z]:\\[\w\\\s./-]+', texto)
    for ruta in set(rutas):
        cambios.append(('ruta_local', ruta))
        texto = texto.replace(ruta, '[RUTA_LOCAL]')

    # 4. Rutas de archivo Linux/URL → [RUTA]
    rutas_linux = re.findall(r'/(?:home|tmp|var|usr|opt|mnt)/[\w/.-]+', texto)
    for ruta in set(rutas_linux):
        cambios.append(('ruta_linux', ruta))
        texto = texto.replace(ruta, '[RUTA]')

    # 5. Direcciones IP → [IP]
    ips = re.findall(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', texto)
    for ip in set(ips):
        # No anonimizar versiones de software (ej. "1.2.3.4" podría ser versión)
        # Verificar que parezca una IP real
        octetos = ip.split('.')
        es_ip = all(0 <= int(o) <= 255 for o in octetos)
        if es_ip:
            cambios.append(('ip', ip))
            texto = texto.replace(ip, '[IP]')

    # 6. Nombres de personas conocidas → [USUARIO]
    # Ordenar de más largo a más corto para evitar reemplazos parciales
    for nombre in sorted(nombres_personas, key=len, reverse=True):
        if nombre and len(nombre) > 3 and nombre in texto:
            cambios.append(('nombre', nombre))
            texto = texto.replace(nombre, '[USUARIO]')

    # 7. Patrones de nombre no capturados: "Hola Nombre," / "Saludos, Nombre"
    texto = re.sub(r'(?i)(?:hola|estimad[oa]|saludos)\s*,?\s*([A-ZÁÉÍÓÚÑ][a-záéíóúñ]+)',
                   lambda m: m.group(0).replace(m.group(1), '[USUARIO]'), texto)

    # 8. Números de ticket/códigos internos → mantener (son útiles para trazabilidad)
    # NO anonimizamos IDs de ticket, son parte del contexto técnico

    return texto, cambios


# ============================================================================
# MÓDULO 2: ELIMINACIÓN DE REDUNDANCIA
# ============================================================================

def normalizar_para_comparacion(texto):
    """Normaliza texto para comparación de similitud."""
    if not texto or not isinstance(texto, str):
        return ""
    texto = texto.lower()
    texto = re.sub(r'[^\w\s]', ' ', texto)
    texto = re.sub(r'\s+', ' ', texto)
    return texto.strip()


def calcular_similitud(texto1, texto2):
    """Calcula similitud entre dos textos usando SequenceMatcher."""
    t1 = normalizar_para_comparacion(texto1)
    t2 = normalizar_para_comparacion(texto2)
    if not t1 or not t2:
        return 0.0
    # Usar solo primeros 500 chars para eficiencia
    return SequenceMatcher(None, t1[:500], t2[:500]).ratio()


def calcular_score_calidad(row):
    """
    Calcula un score de calidad del ticket (mayor = mejor redactado).
    Usado para decidir cuál ticket conservar en caso de redundancia.
    """
    score = 0

    for col in ['Sintomas', 'Diagnostico', 'Solucion Aplicada']:
        val = row.get(col, '')
        if isinstance(val, str):
            palabras = len(val.split())
            score += min(palabras, 100)  # Cap en 100 palabras por campo

    # Bonus por tener Description
    desc = row.get('Description', '')
    if isinstance(desc, str) and len(desc.strip()) > 20:
        score += 10

    # Bonus por tener Comentario
    com = row.get('Comentario', '')
    if isinstance(com, str) and len(com.strip()) > 10:
        score += 5

    # Bonus por Effort válido
    effort = row.get('Effort', 0)
    if effort and effort > 0:
        score += 3

    return score


def eliminar_redundancia(df, umbral=SIMILITUD_UMBRAL):
    """
    Elimina tickets redundantes dentro de cada grupo (Categoria, SubCategoria).
    Usa similitud de texto para detectar casi-duplicados.

    Retorna: (df_limpio, lista_eliminados)
    """
    eliminados = []
    indices_a_eliminar = set()

    # Agrupar por categoría + subcategoría
    grupos = df.groupby(['Categoria', 'SubCategoria'])

    total_comparaciones = 0

    for (cat, subcat), grupo in grupos:
        if len(grupo) < 2:
            continue

        indices = grupo.index.tolist()

        # Pre-calcular textos normalizados de síntomas para comparación
        textos = {}
        scores = {}
        for idx in indices:
            sintomas = str(df.loc[idx, 'Sintomas'])
            diagnostico = str(df.loc[idx, 'Diagnostico'])
            textos[idx] = f"{sintomas} {diagnostico}"
            scores[idx] = calcular_score_calidad(df.loc[idx])

        # Comparar cada par dentro del grupo
        for i in range(len(indices)):
            idx_i = indices[i]
            if idx_i in indices_a_eliminar:
                continue

            for j in range(i + 1, len(indices)):
                idx_j = indices[j]
                if idx_j in indices_a_eliminar:
                    continue

                total_comparaciones += 1
                sim = calcular_similitud(textos[idx_i], textos[idx_j])

                if sim >= umbral:
                    # Eliminar el de menor calidad
                    if scores[idx_i] >= scores[idx_j]:
                        idx_eliminar = idx_j
                        idx_conservar = idx_i
                    else:
                        idx_eliminar = idx_i
                        idx_conservar = idx_j

                    indices_a_eliminar.add(idx_eliminar)
                    eliminados.append({
                        'ID_Eliminado': df.loc[idx_eliminar, 'ID'],
                        'ID_Conservado': df.loc[idx_conservar, 'ID'],
                        'Categoria': cat,
                        'SubCategoria': subcat,
                        'Similitud': round(sim, 3),
                        'Score_Eliminado': scores[idx_eliminar],
                        'Score_Conservado': scores[idx_conservar],
                        'Titulo_Eliminado': str(df.loc[idx_eliminar, 'Title'])[:80],
                        'Titulo_Conservado': str(df.loc[idx_conservar, 'Title'])[:80],
                    })

    df_limpio = df.drop(index=indices_a_eliminar).reset_index(drop=True)

    return df_limpio, eliminados, total_comparaciones


# ============================================================================
# MÓDULO 3: EVALUACIÓN DE CALIDAD
# ============================================================================

def evaluar_calidad_texto(texto):
    """Evalúa calidad del texto para vectorización."""
    if pd.isna(texto) or not isinstance(texto, str) or not texto.strip():
        return 'vacío'
    palabras = len(texto.split())
    if palabras >= 10:
        return 'bueno'
    elif palabras >= 3:
        return 'aceptable'
    else:
        return 'pobre'


def combo_es_valido(categoria, subcategoria):
    """Verifica si la combinación categoría-subcategoría existe en el catálogo."""
    return (categoria in CATALOGO_OFICIAL and
            subcategoria in CATALOGO_OFICIAL[categoria])


# ============================================================================
# PROCESO PRINCIPAL
# ============================================================================

def main():
    print("=" * 72)
    print(f"  HITO 1 {VERSION.upper()} — LIMPIEZA + ANONIMIZACIÓN + DEDUPLICACIÓN")
    print("  Proyecto de Tesis: Asistente RAG para Soporte TI")
    print(f"  Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 72)

    # ------------------------------------------------------------------
    # PASO 1: AUTO-DETECCIÓN DE INPUT
    # ------------------------------------------------------------------
    print("\n[PASO 1] Detectando mejor fuente disponible...")

    input_file = None
    input_desc = None
    for candidate, desc in INPUT_CANDIDATES:
        if os.path.exists(candidate):
            input_file = candidate
            input_desc = desc
            break

    if not input_file:
        print("  ⚠ No se encontró output de v1 ni v2.")
        print("  Buscando CSV original...")
        original = "Soporte-AnaliticaDescriptiva - Completado.csv"
        if os.path.exists(original):
            input_file = original
            input_desc = "original (sin limpiar)"
            print(f"  ⚠ Usando CSV original. Considera ejecutar v1 o v2 primero.")
        else:
            print("  ✗ ERROR: No se encontró ningún archivo de datos.")
            return

    print(f"  ✓ Fuente seleccionada: {input_file} ({input_desc})")

    # ------------------------------------------------------------------
    # PASO 2: CARGA DE DATOS
    # ------------------------------------------------------------------
    print(f"\n[PASO 2] Cargando {input_file}...")
    df = pd.read_csv(input_file, encoding='utf-8-sig')
    total_original = len(df)
    print(f"  ✓ {total_original} registros, {len(df.columns)} columnas")

    cats_antes = df['Categoria'].nunique()
    subcats_antes = df['SubCategoria'].nunique()
    print(f"  Categorías: {cats_antes} | SubCategorías: {subcats_antes}")

    # ------------------------------------------------------------------
    # PASO 3: EXTRACCIÓN DE ENTIDADES PARA ANONIMIZACIÓN
    # ------------------------------------------------------------------
    print("\n[PASO 3] Extrayendo entidades identificables (PII)...")

    nombres_personas = extraer_nombres_del_dataset(df)
    emails_conocidos = extraer_emails_del_dataset(df)

    print(f"  ✓ Personas encontradas: {len(nombres_personas)}")
    print(f"  ✓ Emails encontrados: {len(emails_conocidos)}")

    # Mostrar algunas para verificación
    print(f"  Ejemplos de nombres: {list(sorted(nombres_personas, key=len, reverse=True))[:5]}")

    # ------------------------------------------------------------------
    # PASO 4: ANONIMIZACIÓN
    # ------------------------------------------------------------------
    print("\n[PASO 4] Anonimizando datos personales...")

    df_anon = df.copy()
    anon_log = []
    total_cambios_anon = 0
    cambios_por_tipo = defaultdict(int)

    for col in COLUMNAS_TEXTO:
        if col not in df_anon.columns:
            continue

        cambios_col = 0
        for idx in df_anon.index:
            texto_original = df_anon.at[idx, col]
            texto_anon, cambios = anonimizar_texto(
                texto_original, nombres_personas, emails_conocidos
            )

            if cambios:
                df_anon.at[idx, col] = texto_anon
                cambios_col += len(cambios)

                for tipo, valor in cambios:
                    cambios_por_tipo[tipo] += 1
                    anon_log.append({
                        'ID': df_anon.at[idx, 'ID'],
                        'Campo': col,
                        'Tipo_PII': tipo,
                        'Valor_Original': valor[:80] if valor else '',
                        'Reemplazado_Por': {
                            'email': '[EMAIL]',
                            'nombre': '[USUARIO]',
                            'ruta_local': '[RUTA_LOCAL]',
                            'ruta_linux': '[RUTA]',
                            'ip': '[IP]',
                            'cabecera_email': '[CABECERA_EMAIL_REDACTADA]',
                        }.get(tipo, '[REDACTADO]'),
                    })

        total_cambios_anon += cambios_col
        if cambios_col > 0:
            print(f"  ✓ {col}: {cambios_col} elementos anonimizados")

    # Anonimizar columna 'Assigned To' (nombres de responsables)
    if 'Assigned To' in df_anon.columns:
        asignados_unicos = df_anon['Assigned To'].nunique()
        # Crear mapeo de nombres a códigos anónimos
        personas_unicas = sorted(df_anon['Assigned To'].dropna().unique())
        mapeo_personas = {}
        for i, persona in enumerate(personas_unicas, 1):
            mapeo_personas[persona] = f"ANALISTA_{i:03d}"
        df_anon['Assigned To'] = df_anon['Assigned To'].map(mapeo_personas)
        print(f"  ✓ Assigned To: {asignados_unicos} personas → códigos anónimos")

    print(f"\n  Resumen de anonimización:")
    for tipo, count in sorted(cambios_por_tipo.items(), key=lambda x: -x[1]):
        print(f"    {tipo}: {count}")
    print(f"    Total: {total_cambios_anon} elementos anonimizados")

    # ------------------------------------------------------------------
    # PASO 5: ELIMINACIÓN DE REDUNDANCIA
    # ------------------------------------------------------------------
    print(f"\n[PASO 5] Eliminando tickets redundantes (umbral={SIMILITUD_UMBRAL})...")

    antes_dedup = len(df_anon)
    df_dedup, eliminados, total_comparaciones = eliminar_redundancia(
        df_anon, umbral=SIMILITUD_UMBRAL
    )
    tickets_eliminados = antes_dedup - len(df_dedup)

    print(f"  Comparaciones realizadas: {total_comparaciones}")
    print(f"  Tickets antes: {antes_dedup}")
    print(f"  Tickets eliminados (redundantes): {tickets_eliminados}")
    print(f"  Tickets después: {len(df_dedup)}")

    if eliminados:
        # Resumen por categoría
        elim_por_cat = defaultdict(int)
        for e in eliminados:
            elim_por_cat[e['Categoria']] += 1
        print(f"\n  Eliminados por categoría:")
        for cat, count in sorted(elim_por_cat.items(), key=lambda x: -x[1]):
            print(f"    {cat}: {count}")

    # ------------------------------------------------------------------
    # PASO 6: EVALUACIÓN DE CALIDAD POST-PROCESO
    # ------------------------------------------------------------------
    print(f"\n[PASO 6] Evaluando calidad del texto...")

    calidad_resultados = {}
    for col in ['Sintomas', 'Diagnostico', 'Solucion Aplicada']:
        if col in df_dedup.columns:
            calidades = df_dedup[col].apply(evaluar_calidad_texto)
            conteo = calidades.value_counts()
            calidad_resultados[col] = conteo
            bueno = conteo.get('bueno', 0)
            total = len(df_dedup)
            print(f"  {col}: {bueno}/{total} buenos ({bueno/total*100:.1f}%)")

    # ------------------------------------------------------------------
    # PASO 7: CONSTRUCCIÓN DE BASE DE CONOCIMIENTO PARA RAG
    # ------------------------------------------------------------------
    print(f"\n[PASO 7] Construyendo base de conocimiento optimizada para RAG...")

    def construir_texto_rag(row):
        """
        Construye el texto combinado optimizado para embedding y retrieval.
        Formato estructurado que facilita tanto la búsqueda como la generación.
        """
        partes = []

        # Contexto tecnológico (metadatos que ayudan al retrieval)
        partes.append(f"[CATEGORÍA: {row['Categoria']}]")
        partes.append(f"[SUBCATEGORÍA: {row['SubCategoria']}]")
        if pd.notna(row.get('Servicio Clasificacion')) and str(row['Servicio Clasificacion']).strip():
            partes.append(f"[SERVICIO: {row['Servicio Clasificacion']}]")

        # Título del problema
        if pd.notna(row.get('Title')) and str(row['Title']).strip():
            partes.append(f"\nPROBLEMA: {row['Title']}")

        # Bloque de diagnóstico (input para el RAG)
        partes.append(f"\nSÍNTOMAS: {row.get('Sintomas', '')}")
        partes.append(f"\nDIAGNÓSTICO: {row.get('Diagnostico', '')}")

        # Bloque de resolución (output que el RAG debe retornar)
        partes.append(f"\nSOLUCIÓN APLICADA: {row.get('Solucion Aplicada', '')}")

        # Contexto adicional (si es significativo)
        desc = row.get('Description', '')
        if pd.notna(desc) and isinstance(desc, str) and len(desc.strip()) > 20:
            # Limitar para no exceder ventana de embedding
            desc_truncada = desc[:500] if len(desc) > 500 else desc
            partes.append(f"\nDESCRIPCIÓN ADICIONAL: {desc_truncada}")

        comentario = row.get('Comentario', '')
        if pd.notna(comentario) and isinstance(comentario, str) and len(comentario.strip()) > 10:
            comentario_truncado = comentario[:300] if len(comentario) > 300 else comentario
            partes.append(f"\nCOMENTARIO: {comentario_truncado}")

        # Esfuerzo (dato para análisis de eficiencia)
        effort = row.get('Effort', 0)
        if effort and effort > 0:
            partes.append(f"\n[ESFUERZO: {effort}h]")

        return '\n'.join(partes)

    df_dedup['texto_combinado'] = df_dedup.apply(construir_texto_rag, axis=1)

    # Seleccionar columnas para KB
    cols_kb = [
        'ID', 'Title', 'Categoria', 'SubCategoria', 'Servicio Clasificacion',
        'Servicio Soporte', 'Sintomas', 'Diagnostico', 'Solucion Aplicada',
        'Description', 'Comentario', 'texto_combinado',
        'Tipo Ticket', 'State', 'Report Date', 'Effort',
    ]
    cols_disponibles = [c for c in cols_kb if c in df_dedup.columns]
    df_kb = df_dedup[cols_disponibles].copy()

    # Métricas de texto
    df_kb['longitud_texto'] = df_kb['texto_combinado'].apply(len)
    df_kb_viable = df_kb[df_kb['longitud_texto'] > 50].copy()

    print(f"  Total registros (post-dedup): {len(df_dedup)}")
    print(f"  Viables para RAG (>50 chars): {len(df_kb_viable)}")
    print(f"  Descartados: {len(df_dedup) - len(df_kb_viable)}")

    avg_len = df_kb_viable['longitud_texto'].mean()
    med_len = df_kb_viable['longitud_texto'].median()
    max_len = df_kb_viable['longitud_texto'].max()
    min_len = df_kb_viable['longitud_texto'].min()

    print(f"\n  Longitud del texto combinado:")
    print(f"    Promedio: {avg_len:.0f} chars")
    print(f"    Mediana:  {med_len:.0f} chars")
    print(f"    Rango:    {min_len} - {max_len} chars")

    # ------------------------------------------------------------------
    # PASO 8: ESTADÍSTICAS FINALES
    # ------------------------------------------------------------------
    print(f"\n[PASO 8] Estadísticas finales...")

    cats_final = df_dedup['Categoria'].nunique()
    subcats_final = df_dedup['SubCategoria'].nunique()

    print(f"\n  Categorías: {cats_final} | SubCategorías: {subcats_final}")

    print("\n" + "=" * 62)
    print(f"  {'CATEGORÍA':<35} {'TICKETS':>8} {'%':>7}")
    print("=" * 62)
    cat_counts = df_dedup['Categoria'].value_counts()
    for cat, count in cat_counts.items():
        print(f"  {cat:<33} {count:>8} {count/len(df_dedup)*100:>6.1f}%")
    print("-" * 62)
    print(f"  {'TOTAL':<33} {len(df_dedup):>8} {'100.0%':>7}")

    # Validación final
    combos_invalidos = []
    for idx, row in df_dedup.iterrows():
        if not combo_es_valido(row['Categoria'], row['SubCategoria']):
            combos_invalidos.append((row['Categoria'], row['SubCategoria'], row['ID']))

    if combos_invalidos:
        print(f"\n  ⚠ {len(combos_invalidos)} combos inválidos")
    else:
        print("\n  ✅ Todos los combos son válidos")

    # ------------------------------------------------------------------
    # PASO 9: GUARDAR RESULTADOS
    # ------------------------------------------------------------------
    print(f"\n[PASO 9] Guardando resultados ({VERSION})...")

    # Dataset completo anonimizado y deduplicado
    df_dedup.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
    print(f"  ✓ {OUTPUT_FILE} ({len(df_dedup)} registros)")

    # Base de conocimiento para RAG
    df_kb_viable.to_csv(OUTPUT_KB_FILE, index=False, encoding='utf-8-sig')
    print(f"  ✓ {OUTPUT_KB_FILE} ({len(df_kb_viable)} registros)")

    # Reporte de anonimización
    if anon_log:
        pd.DataFrame(anon_log).to_csv(ANON_REPORT_FILE, index=False, encoding='utf-8-sig')
        print(f"  ✓ {ANON_REPORT_FILE} ({len(anon_log)} cambios)")

    # Reporte de redundancia
    if eliminados:
        pd.DataFrame(eliminados).to_csv(
            REDUNDANCIA_REPORT_FILE, index=False, encoding='utf-8-sig')
        print(f"  ✓ {REDUNDANCIA_REPORT_FILE} ({len(eliminados)} eliminados)")

    # ------------------------------------------------------------------
    # PASO 10: GENERAR REPORTE DE MÉTRICAS
    # ------------------------------------------------------------------
    print(f"\n[PASO 10] Generando reporte de métricas...")

    reporte = []
    reporte.append("=" * 72)
    reporte.append(f"  REPORTE DE MÉTRICAS — HITO 1 {VERSION.upper()}")
    reporte.append("  Limpieza + Anonimización + Eliminación de Redundancia")
    reporte.append(f"  Fuente: {input_file} ({input_desc})")
    reporte.append(f"  Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    reporte.append("=" * 72)

    reporte.append("\n1. RESUMEN GENERAL")
    reporte.append(f"   Registros de entrada: {total_original}")
    reporte.append(f"   Registros post-anonimización: {total_original}")
    reporte.append(f"   Registros eliminados (redundancia): {tickets_eliminados}")
    reporte.append(f"   Registros finales: {len(df_dedup)}")
    reporte.append(f"   Viables para RAG: {len(df_kb_viable)}")

    reporte.append("\n2. ANONIMIZACIÓN")
    reporte.append(f"   Total elementos anonimizados: {total_cambios_anon}")
    for tipo, count in sorted(cambios_por_tipo.items(), key=lambda x: -x[1]):
        reporte.append(f"   {tipo}: {count}")
    reporte.append(f"   Personas en 'Assigned To': {len(mapeo_personas) if 'Assigned To' in df_anon.columns else 0}")

    reporte.append("\n3. ELIMINACIÓN DE REDUNDANCIA")
    reporte.append(f"   Umbral de similitud: {SIMILITUD_UMBRAL}")
    reporte.append(f"   Comparaciones realizadas: {total_comparaciones}")
    reporte.append(f"   Tickets redundantes eliminados: {tickets_eliminados}")
    if eliminados:
        reporte.append(f"   Detalle por categoría:")
        elim_por_cat = defaultdict(int)
        for e in eliminados:
            elim_por_cat[e['Categoria']] += 1
        for cat, count in sorted(elim_por_cat.items(), key=lambda x: -x[1]):
            reporte.append(f"     {cat}: {count}")

    reporte.append("\n4. CALIDAD DEL TEXTO")
    for col, conteo in calidad_resultados.items():
        reporte.append(f"\n   {col}:")
        for nivel, cantidad in conteo.items():
            reporte.append(f"     {nivel}: {cantidad} ({cantidad/len(df_dedup)*100:.1f}%)")

    reporte.append("\n5. DISTRIBUCIÓN POR CATEGORÍA (FINAL)")
    for cat, count in cat_counts.items():
        reporte.append(f"   {cat}: {count} ({count/len(df_dedup)*100:.1f}%)")

    reporte.append("\n6. COMBINACIONES FINALES")
    combos = df_dedup.groupby(['Categoria', 'SubCategoria']).size().reset_index(name='count')
    combos = combos.sort_values(['Categoria', 'count'], ascending=[True, False])
    for _, r in combos.iterrows():
        v = "✓" if combo_es_valido(r['Categoria'], r['SubCategoria']) else "⚠"
        reporte.append(f"   {v} {r['Categoria']:<30} | {r['SubCategoria']:<30} | {r['count']}")

    reporte.append("\n7. TEXTO COMBINADO (PARA VECTORIZACIÓN)")
    reporte.append(f"   Promedio: {avg_len:.0f} chars")
    reporte.append(f"   Mediana:  {med_len:.0f} chars")
    reporte.append(f"   Max: {max_len} chars | Min: {min_len} chars")

    reporte.append("\n8. COMPARACIÓN DE VERSIONES")
    reporte.append("   v1: Limpieza HTML + re-categorización por keywords")
    reporte.append("   v2: v1 + re-categorización por Gemini API")
    reporte.append("   v3: v1/v2 + Anonimización + Deduplicación + KB optimizada")
    reporte.append(f"   Input usado para v3: {input_file} ({input_desc})")
    reporte.append(f"   Registros: {total_original} → {len(df_dedup)} "
                   f"(−{tickets_eliminados} redundantes)")

    if combos_invalidos:
        reporte.append(f"\n9. ⚠ COMBOS INVÁLIDOS ({len(combos_invalidos)})")
        for cat, subcat, tid in combos_invalidos[:20]:
            reporte.append(f"   ID {tid}: {cat} > {subcat}")

    reporte.append("\n" + "=" * 72)
    reporte.append("  FIN DEL REPORTE")
    reporte.append("=" * 72)

    with open(REPORT_FILE, 'w', encoding='utf-8') as f:
        f.write('\n'.join(reporte))
    print(f"  ✓ {REPORT_FILE}")

    # ------------------------------------------------------------------
    # RESUMEN FINAL
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print(f"  ✅ HITO 1 {VERSION.upper()} COMPLETADO")
    print("=" * 72)
    print(f"\n  Archivos generados:")
    print(f"    1. {OUTPUT_FILE} — Dataset anonimizado y deduplicado")
    print(f"    2. {OUTPUT_KB_FILE} — Base de conocimiento para RAG (H2)")
    print(f"    3. {REPORT_FILE} — Reporte de métricas")
    if anon_log:
        print(f"    4. {ANON_REPORT_FILE} — Detalle de anonimización")
    if eliminados:
        print(f"    5. {REDUNDANCIA_REPORT_FILE} — Tickets eliminados")

    print(f"\n  Pipeline completo:")
    print(f"    Original: 1,134 registros")
    print(f"    → v1/v2: Limpieza + categorización ({total_original} registros)")
    print(f"    → v3:    + Anonimización ({total_cambios_anon} elementos)")
    print(f"    → v3:    + Deduplicación (−{tickets_eliminados} redundantes)")
    print(f"    → Final: {len(df_kb_viable)} registros listos para RAG")
    print(f"\n  Texto combinado: promedio {avg_len:.0f} chars, "
          f"mediana {med_len:.0f} chars")
    print()

    return df_dedup, df_kb_viable


if __name__ == "__main__":
    df_final, df_kb = main()
