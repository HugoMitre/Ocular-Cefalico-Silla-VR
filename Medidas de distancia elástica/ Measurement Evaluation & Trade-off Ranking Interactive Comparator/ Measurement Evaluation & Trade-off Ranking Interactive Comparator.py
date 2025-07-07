"""
EVALUADOR HÍBRIDO DE MEDIDAS DE SIMILITUD
Sistema con Optimización Multiobjetivo usando F(x) = α·Cost_comp(x) - β·Accuracy_align(x)4
Usuario ingresa α y β, ranking SÍ cambia según parámetros
"""

import re
import unicodedata
import time
import glob
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import json
from collections import defaultdict

# ══════════════ 1. CONFIGURACIÓN Y CONSTANTES ACTUALIZADAS ══════════════
COMMANDS = [
    "Right Turn", "Front", "Left Turn",
    "Front-Right Diagonal", "Front-Left Diagonal",
    "Back-Left Diagonal", "Back-Right Diagonal", "Back"
]

# ============ MEDIDAS ACTUALIZADAS CON LCSS Y cDTW ============
SIMILARITY_MEASURES = {
    "MSM": ["msm", "move_split_merge"],
    "ERP": ["erp", "edit_distance_real_penalty"],
    "TWE": ["twe", "time_warp_edit"],
    "ADTW": ["adtw", "adaptive_dimension"],
    "ESDTW": ["esdtw", "ensemble_shapelet"],
    "SSDTW": ["ssdtw", "subsequence_sakoe"],
    "WDTW": ["wdtw", "weighted_dtw"],
    "BDTW": ["bdtw", "basic_dtw"],
    # ============ NUEVAS MEDIDAS AGREGADAS ============
    "LCSS": ["lcss", "longest_common_subsequence"],
    "cDTW": ["cdtw", "constrained_dtw", "constrained_dynamic_time_warping"]
}

EXPECTED_FOLDER_PATTERNS = [
    "{measure}_results_matrices_flops",
    "{measure}_results_with_exact_flops", 
    "{measure}_results_matrices",
    "{measure}_results_flops",
    "{measure}_results",
    "results_{measure}_matrices_flops",
    "results_{measure}_matrices",
    "analisis_errores_{measure}"
]

CMD_CAT = pd.api.types.CategoricalDtype(COMMANDS, ordered=True)

# ══════════════ 2. SOLICITUD INTERACTIVA DE PARÁMETROS ══════════════
def solicitar_parametros_usuario() -> Tuple[float, float]:
    """
    Función interactiva para solicitar α y β al usuario
    """
    print("🎯 CONFIGURACIÓN DE PARÁMETROS MULTIOBJETIVO")
    print("=" * 50)
    print("Los parámetros determinan el balance entre:")
    print("• α (alfa): Peso del COSTO COMPUTACIONAL (0.0 - 1.0)")
    print("• β (beta): Peso de la PRECISIÓN DE ALINEACIÓN (0.0 - 1.0)")
    print("• Recomendación: α + β = 1.0 para balance apropiado")
    print()
    print("Ejemplos:")
    print("  α=0.8, β=0.2 → Prioriza EFICIENCIA computacional")
    print("  α=0.5, β=0.5 → BALANCE equilibrado")
    print("  α=0.2, β=0.8 → Prioriza PRECISIÓN de alineación")
    print()
    
    while True:
        try:
            alpha_input = input("Ingresa α (peso del costo computacional) [default: 0.6]: ").strip()
            alpha = float(alpha_input) if alpha_input else 0.6
            
            beta_input = input("Ingresa β (peso de la precisión) [default: 0.4]: ").strip()
            beta = float(beta_input) if beta_input else 0.4
            
            # Validaciones
            if not (0 <= alpha <= 1 and 0 <= beta <= 1):
                print("❌ Los valores deben estar entre 0.0 y 1.0")
                continue
            
            suma = alpha + beta
            if abs(suma - 1.0) > 0.1:
                print(f"⚠️  α + β = {suma:.2f} (recomendado: ~1.0)")
                continuar = input("¿Continuar con estos valores? (s/n) [s]: ").strip().lower()
                if continuar and continuar.startswith('n'):
                    continue
            
            # Confirmación
            print(f"\n✅ PARÁMETROS CONFIRMADOS:")
            print(f"   α = {alpha:.1f} ({'Alta' if alpha > 0.7 else 'Media' if alpha > 0.3 else 'Baja'} prioridad a eficiencia)")
            print(f"   β = {beta:.1f} ({'Alta' if beta > 0.7 else 'Media' if beta > 0.3 else 'Baja'} prioridad a precisión)")
            
            # Predicción del comportamiento
            if alpha > beta:
                print(f"🎯 Predicción: Favorecerá algoritmos MÁS EFICIENTES")
            elif beta > alpha:
                print(f"🎯 Predicción: Favorecerá algoritmos MÁS PRECISOS")
            else:
                print(f"🎯 Predicción: Buscará el MEJOR BALANCE")
            
            return alpha, beta
            
        except ValueError:
            print("❌ Por favor ingresa números válidos (ejemplo: 0.6)")
        except KeyboardInterrupt:
            print("\n👋 Cancelado por el usuario")
            exit(0)

# ══════════════ 3. UTILIDADES DE NORMALIZACIÓN (MANTENIDAS DEL PROGRAMA 1) ══════════════
def _strip(txt: str) -> str:
    """Elimina acentos y caracteres especiales"""
    return "".join(ch for ch in unicodedata.normalize("NFD", txt)
                   if unicodedata.category(ch) != "Mn")

def canon_cmd(raw: str) -> str | None:
    """Convierte texto a comando estándar"""
    if not isinstance(raw, str) or pd.isna(raw): 
        return None
    
    s = re.sub(r"\s+", " ", re.sub(r"[^a-zA-Z ]+", " ", _strip(raw))).strip().lower()
    diag = ("diag" in s) or ("diagonal" in s)
    
    if "right" in s and "turn" in s and "front" not in s:  
        return "Right Turn"
    if "left"  in s and "turn" in s and "front" not in s:  
        return "Left Turn"
    if s == "front" or ("front" in s and not diag and "turn" not in s): 
        return "Front"
    if "front" in s and "right" in s and diag:  
        return "Front-Right Diagonal"
    if "front" in s and "left"  in s and diag:  
        return "Front-Left Diagonal"
    if "back"  in s and "left"  in s and diag:  
        return "Back-Left Diagonal"
    if "back"  in s and "right" in s and diag:  
        return "Back-Right Diagonal"
    if s == "back" or ("back" in s and not diag):          
        return "Back"
    
    return None

def meta(path: Path) -> Tuple[str, str]:
    """Extrae ID de participante y etapa de la ruta"""
    pid = next((p.split("_")[1] for p in path.parts if p.lower().startswith("participant_")), "unknown")
    m = re.search(r"etapa[_\-]?(\d)", str(path), re.I)
    return pid, (m.group(1) if m else "1")

# ══════════════ 4. IDENTIFICACIÓN ROBUSTA DE CARPETAS (ACTUALIZADA) ══════════════
def find_measure_directories(base_path: Path) -> Dict[str, Path]:
    """Encuentra carpetas de medidas de similitud de forma ultra-robusta - AHORA CON 10 MEDIDAS"""
    print(f"🔍 Buscando carpetas de medidas en: {base_path}")
    print(f"📊 Medidas a buscar: {', '.join(SIMILARITY_MEASURES.keys())} (Total: {len(SIMILARITY_MEASURES)})")
    
    found_measures = {}
    all_dirs = []
    
    # Buscar recursivamente hasta 3 niveles
    for level in range(4):
        if level == 0:
            all_dirs.extend([d for d in base_path.iterdir() if d.is_dir()])
        else:
            new_dirs = []
            for existing_dir in all_dirs:
                if existing_dir.is_dir():
                    try:
                        new_dirs.extend([d for d in existing_dir.iterdir() if d.is_dir()])
                    except PermissionError:
                        continue
            all_dirs.extend(new_dirs)
    
    # Para cada medida, buscar su carpeta
    for measure, aliases in SIMILARITY_MEASURES.items():
        print(f"\n📂 Buscando carpeta para {measure}...")
        
        best_match = None
        best_score = 0
        
        for dir_path in all_dirs:
            dir_name_lower = dir_path.name.lower()
            clean_name = re.sub(r'[^a-z0-9]', ' ', dir_name_lower)
            words_in_name = set(clean_name.split())
            
            # Calcular puntuación de coincidencia
            score = 0
            
            # Coincidencia directa con medida o alias
            for alias in [measure.lower()] + aliases:
                if alias in dir_name_lower:
                    score += 10
                    break
            
            # Palabras clave importantes
            important_keywords = ["results", "matrices", "flops", "analisis", "errores"]
            for keyword in important_keywords:
                if keyword in words_in_name:
                    score += 2
            
            # Patrones específicos conocidos
            for pattern in EXPECTED_FOLDER_PATTERNS:
                expected = pattern.format(measure=measure.lower())
                if expected in dir_name_lower:
                    score += 15
                    break
            
            # Verificar si la carpeta contiene archivos relevantes
            if score > 5:
                try:
                    csv_files = list(dir_path.glob("*.csv"))
                    excel_files = list(dir_path.glob("*.xlsx"))
                    if csv_files or excel_files:
                        score += 3
                except:
                    pass
            
            # Actualizar mejor coincidencia
            if score > best_score and score >= 8:
                best_score = score
                best_match = dir_path
        
        if best_match:
            found_measures[measure] = best_match
            print(f"  ✓ Encontrado: {best_match.name} (puntuación: {best_score})")
        else:
            print(f"   No encontrado para {measure}")
    
    print(f"\n📊 RESUMEN DE DETECCIÓN:")
    print(f"   Total medidas encontradas: {len(found_measures)}/{len(SIMILARITY_MEASURES)}")
    
    # Mostrar cuáles se encontraron
    found_list = list(found_measures.keys())
    missing_list = [m for m in SIMILARITY_MEASURES.keys() if m not in found_measures]
    
    if found_list:
        print(f"   ✅ Encontradas: {', '.join(found_list)}")
    if missing_list:
        print(f"   ❌ No encontradas: {', '.join(missing_list)}")
    
    return found_measures

# ══════════════ 5. LECTURA ROBUSTA DE ARCHIVOS (DEL PROGRAMA 1) ══════════════
def read_csv_flexible(path: Path) -> pd.DataFrame:
    """Lee CSV o Excel con manejo flexible de columnas y errores"""
    try:
        if path.suffix.lower() in ['.xlsx', '.xls']:
            df = pd.read_excel(path)
        else:
            # Intentar diferentes encodings
            for encoding in ['utf-8', 'latin-1', 'cp1252']:
                try:
                    df = pd.read_csv(path, encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue
            else:
                df = pd.read_csv(path, encoding='utf-8', errors='ignore')
        
        # Limpiar nombres de columnas
        df.columns = [c.strip() for c in df.columns]
        
        # Mapeo de variantes de nombres de columnas
        col_variants = {
            "Command": ["command", "cmd", "direction", "action", "movement", "comando"],
            "real_index_x": [
                "real_index_x", "realindex_x", "real_x", "idx_real_x",
                "block_index_x_real", "block_real_x", "real_block_x", "blockindex_x_real"
            ],
            "ideal_index_x": [
                "ideal_index_x", "idealindex_x", "ideal_x", "idx_ideal_x",
                "block_index_x_ideal", "block_ideal_x", "ideal_block_x", "blockindex_x_ideal"
            ],
            "real_index_z": [
                "real_index_z", "realindex_z", "real_z", "idx_real_z",
                "block_index_z_real", "block_real_z", "real_block_z", "blockindex_z_real"
            ],
            "ideal_index_z": [
                "ideal_index_z", "idealindex_z", "ideal_z", "idx_ideal_z",
                "block_index_z_ideal", "block_ideal_z", "ideal_block_z", "blockindex_z_ideal"
            ],
            "path_i_x": ["path_i_x", "pathi_x", "i_x", "path_idx_x"],
            "path_j_x": ["path_j_x", "pathj_x", "j_x", "path_idy_x"],
            "path_i_z": ["path_i_z", "pathi_z", "i_z", "path_idx_z"],
            "path_j_z": ["path_j_z", "pathj_z", "j_z", "path_idy_z"]
        }
        
        # Normalizar nombres de columnas
        for req_col, variants in col_variants.items():
            if req_col not in df.columns:
                for var in variants:
                    for col in df.columns:
                        col_clean = col.lower().replace(" ", "_").replace("-", "_")
                        if var == col_clean:
                            df.rename(columns={col: req_col}, inplace=True)
                            break
        
        # Procesar comandos si existe la columna
        if "Command" in df.columns:
            df["Command"] = df["Command"].apply(canon_cmd)
            df = df[df["Command"].notna()]
        
        return df
        
    except Exception as e:
        print(f"❌ Error leyendo {path}: {e}")
        return pd.DataFrame()

# ══════════════ 6. LÓGICA PARA FLOPS Y TIEMPO (DEL PROGRAMA 1) ══════════════
def procesar_medida_elastica_flops_tiempo(measure_dir: Path, measure_name: str) -> Dict[str, float]:
    """Procesa una medida elástica específica y retorna sus promedios globales de FLOPS y tiempo."""
    print(f"  📊 Calculando FLOPS y Tiempo para {measure_name}...")
    
    if not measure_dir.exists():
        print(f"    ❌ El directorio no existe: {measure_dir}")
        return {"total_flops": np.nan, "total_time_seconds": np.nan}
    
    # Diccionarios para almacenar todos los datos por comando
    datos_por_comando = defaultdict(lambda: {'tiempos': [], 'flops': []})
    archivos_encontrados = 0
    archivos_procesados = 0
    errores = 0
    
    # Buscar todas las carpetas de participantes (participant_01 a participant_30)
    for participante_num in range(1, 31):
        participante_folder = f"participant_{participante_num:02d}"
        ruta_participante = measure_dir / participante_folder
        
        if not ruta_participante.exists():
            continue
        
        # Buscar las 3 etapas
        for etapa in range(1, 4):
            etapa_folder = f"etapa_{etapa}"
            ruta_etapa = ruta_participante / etapa_folder
            
            if not ruta_etapa.exists():
                continue
            
            # Buscar SOLO archivo "similarity" (no "paths")
            patron_similarity = ruta_etapa / "*similarity*.csv"
            archivos_patron = list(glob.glob(str(patron_similarity)))
            archivos_similarity = [f for f in archivos_patron if "paths" not in Path(f).name.lower()]
            
            if archivos_similarity:
                archivo_similarity = Path(archivos_similarity[0])
                archivos_encontrados += 1
                
                try:
                    df = pd.read_csv(archivo_similarity)
                    
                    # Verificar columnas necesarias (variantes)
                    columnas_tiempo = ['command_processing_time_seconds', 'processing_time_seconds', 'time_seconds', 'tiempo_segundos']
                    columnas_flops = ['total_flops', 'flops', 'computational_cost']
                    columnas_comando = ['command', 'cmd', 'direction', 'action']
                    
                    # Encontrar columnas existentes
                    col_tiempo = None
                    col_flops = None
                    col_comando = None
                    
                    for col in df.columns:
                        col_lower = col.lower().replace(" ", "_").replace("-", "_")
                        if not col_tiempo and any(var in col_lower for var in columnas_tiempo):
                            col_tiempo = col
                        if not col_flops and any(var in col_lower for var in columnas_flops):
                            col_flops = col
                        if not col_comando and any(var in col_lower for var in columnas_comando):
                            col_comando = col
                    
                    if col_tiempo and col_flops:  # Al menos tiempo y flops
                        # Procesar todos los comandos
                        for index, fila in df.iterrows():
                            comando = fila.get(col_comando, f'comando_{index+1}') if col_comando else f'comando_{index+1}'
                            tiempo = fila.get(col_tiempo, 'N/A')
                            flops = fila.get(col_flops, 'N/A')
                            
                            # Normalizar comando si es posible
                            if isinstance(comando, str):
                                comando_normalizado = canon_cmd(comando)
                                if comando_normalizado:
                                    comando = comando_normalizado
                            
                            # Guardar datos para promedios globales
                            if tiempo != 'N/A' and flops != 'N/A':
                                try:
                                    datos_por_comando[comando]['tiempos'].append(float(tiempo))
                                    datos_por_comando[comando]['flops'].append(float(flops))
                                except (ValueError, TypeError):
                                    pass
                        
                        archivos_procesados += 1
                    else:
                        print(f"    ⚠️ Columnas faltantes en {archivo_similarity.name}")
                    
                except Exception as e:
                    errores += 1
                    print(f"     Error procesando {archivo_similarity.name}: {e}")
    
    print(f"     Archivos encontrados: {archivos_encontrados}")
    print(f"    ✓ Archivos procesados: {archivos_procesados}")
    if errores > 0:
        print(f"     Errores: {errores}")

    if not datos_por_comando:
        print(f"     No se encontraron datos válidos para {measure_name}")
        return {"total_flops": np.nan, "total_time_seconds": np.nan}
    
    # Calcular promedios por comando
    comandos_ordenados = sorted(datos_por_comando.keys())
    promedios_tiempo_por_comando = []
    promedios_flops_por_comando = []
    
    print(f"     Calculando promedios por comando:")
    for comando in comandos_ordenados:
        tiempos = datos_por_comando[comando]['tiempos']
        flops = datos_por_comando[comando]['flops']
        
        if tiempos and flops:
            promedio_tiempo = sum(tiempos) / len(tiempos)
            promedio_flops = sum(flops) / len(flops)
            num_muestras = len(tiempos)
            
            print(f"      - {comando}: T={promedio_tiempo:.6f}s, F={promedio_flops:.2f}, N={num_muestras}")
            
            promedios_tiempo_por_comando.append(promedio_tiempo)
            promedios_flops_por_comando.append(promedio_flops)
    
    # Calcular promedio global de promedios
    if promedios_tiempo_por_comando and promedios_flops_por_comando:
        promedio_global_tiempo = sum(promedios_tiempo_por_comando) / len(promedios_tiempo_por_comando)
        promedio_global_flops = sum(promedios_flops_por_comando) / len(promedios_flops_por_comando)
        
        print(f"    ✅ Promedios globales: Tiempo={promedio_global_tiempo:.6f}s, FLOPs={promedio_global_flops:.2f}")
        
        return {
            "total_flops": promedio_global_flops,
            "total_time_seconds": promedio_global_tiempo
        }
    else:
        print(f"    ❌ No se pudieron calcular promedios para {measure_name}")
        return {"total_flops": np.nan, "total_time_seconds": np.nan}

# ══════════════ 7. CÁLCULO DE ERRORES (DEL PROGRAMA 1) ══════════════
def build_ground_truth_mappings(gt_df: pd.DataFrame) -> Dict[str, Dict[str, Dict[int, int]]]:
    """Construye diccionarios de mapeo i→j para cada comando y eje"""
    mappings = {}
    
    if "Command" not in gt_df.columns:
        print("⚠️ Ground-truth sin columna 'Command'. Asumiendo 'Front' para todos.")
        gt_df["Command"] = "Front"
    
    # Convertir a enteros las columnas de path
    for col in ["path_i_x", "path_j_x", "path_i_z", "path_j_z"]:
        if col in gt_df.columns:
            gt_df[col] = pd.to_numeric(gt_df[col], errors='coerce').fillna(-1).astype(int)
    
    for cmd in COMMANDS:
        cmd_data = gt_df[gt_df["Command"] == cmd]
        
        if cmd_data.empty:
            mappings[cmd] = {"x": {}, "z": {}}
            continue
        
        # Mapeo para X
        x_data = cmd_data[cmd_data["path_i_x"] >= 0]
        x_mapping = (x_data[["path_i_x", "path_j_x"]]
                    .drop_duplicates("path_i_x")
                    .set_index("path_i_x")["path_j_x"]
                    .to_dict())
        
        # Mapeo para Z
        z_data = cmd_data[cmd_data["path_i_z"] >= 0]
        z_mapping = (z_data[["path_i_z", "path_j_z"]]
                    .drop_duplicates("path_i_z")
                    .set_index("path_i_z")["path_j_z"]
                    .to_dict())
        
        mappings[cmd] = {"x": x_mapping, "z": z_mapping}
    
    return mappings

def calculate_alignment_error(path_df: pd.DataFrame, gt_mappings: Dict[str, Dict[int, int]], 
                            command: str) -> Dict[str, float]:
    """
    Calcula el error absoluto medio de alineación según la fórmula matemática:
    ε_mean = (1/N) * Σ|w_i - w_i^ref|
    """
    if command not in gt_mappings:
        return {
            "e_mean_x": np.nan, "e_mean_z": np.nan, "e_mean": np.nan,
            "std_x": np.nan, "std_z": np.nan, "e_max": np.nan,
            "n_points": 0, "coverage_x": 0.0, "coverage_z": 0.0
        }
    
    gt_x = gt_mappings[command]["x"]
    gt_z = gt_mappings[command]["z"]
    
    # Convertir a enteros
    for col in ["real_index_x", "real_index_z", "ideal_index_x", "ideal_index_z"]:
        if col in path_df.columns:
            path_df[col] = pd.to_numeric(path_df[col], errors='coerce').fillna(-1).astype(int)
    
    # Calcular errores absolutos
    errors_x = []
    errors_z = []
    
    for _, row in path_df.iterrows():
        real_x = int(row.get("real_index_x", -1))
        real_z = int(row.get("real_index_z", -1))
        ideal_x = int(row.get("ideal_index_x", -1))
        ideal_z = int(row.get("ideal_index_z", -1))
        
        # Error en X
        if real_x in gt_x and real_x >= 0 and ideal_x >= 0:
            error_x = abs(ideal_x - gt_x[real_x])
            errors_x.append(error_x)
        
        # Error en Z
        if real_z in gt_z and real_z >= 0 and ideal_z >= 0:
            error_z = abs(ideal_z - gt_z[real_z])
            errors_z.append(error_z)
    
    # Aplicar la fórmula
    e_mean_x = np.mean(errors_x) if errors_x else np.nan
    e_mean_z = np.mean(errors_z) if errors_z else np.nan
    
    all_errors = errors_x + errors_z
    e_mean = np.mean(all_errors) if all_errors else np.nan
    
    # Estadísticas adicionales
    std_x = np.std(errors_x) if len(errors_x) > 1 else np.nan
    std_z = np.std(errors_z) if len(errors_z) > 1 else np.nan
    e_max = np.max(all_errors) if all_errors else np.nan
    
    # Cobertura
    total_points = len(path_df)
    coverage_x = len(errors_x) / total_points if total_points > 0 else 0.0
    coverage_z = len(errors_z) / total_points if total_points > 0 else 0.0
    
    return {
        "e_mean_x": round(e_mean_x, 4) if not np.isnan(e_mean_x) else np.nan,
        "e_mean_z": round(e_mean_z, 4) if not np.isnan(e_mean_z) else np.nan,
        "e_mean": round(e_mean, 4) if not np.isnan(e_mean) else np.nan,
        "std_x": round(std_x, 4) if not np.isnan(std_x) else np.nan,
        "std_z": round(std_z, 4) if not np.isnan(std_z) else np.nan,
        "e_max": round(e_max, 4) if not np.isnan(e_max) else np.nan,
        "n_points": total_points,
        "coverage_x": round(coverage_x, 4),
        "coverage_z": round(coverage_z, 4)
    }

def find_path_files(measure_dir: Path, measure_name: str) -> List[Path]:
    """Encuentra archivos de caminos para una medida específica"""
    print(f"  🔍 Buscando archivos de caminos para {measure_name}...")
    
    all_files = []
    for pattern in ["*.csv", "*.xlsx"]:
        all_files.extend(list(measure_dir.rglob(pattern)))
    
    path_files = []
    
    # Palabras clave que indican archivos de caminos/paths
    path_keywords = [
        "path", "paths", "camino", "caminos", "ruta", "rutas", "trayectoria", "trayectorias",
        "participant", "participante", "etapa", "stage", "alignment", "alineacion",
        measure_name.lower()
    ]
    
    # Palabras que excluyen archivos maestros/resumen
    exclude_keywords = [
        "master", "maestro", "similarity", "distance", "matrix", "matriz", "summary",
        "resumen", "consolidado", "global", "total", "scores", "puntuaciones"
    ]
    
    for file_path in all_files:
        filename_lower = file_path.name.lower()
        
        # Verificar patrón específico de participante_XX_etapa_Y
        participant_pattern = re.search(r'participant_\d+.*etapa_\d+', filename_lower)
        
        # Verificar si contiene palabras clave de caminos
        has_path_keyword = any(keyword in filename_lower for keyword in path_keywords)
        
        # Verificar que no sea un archivo maestro
        is_master_file = any(keyword in filename_lower for keyword in exclude_keywords)
        
        # Incluir si:
        # 1. Tiene patrón de participante_etapa, O
        # 2. Tiene palabra clave de path Y NO es archivo maestro
        if participant_pattern or (has_path_keyword and not is_master_file):
            path_files.append(file_path)
            print(f"    ✓ Encontrado: {file_path.name}")
    
    print(f"    📁 Total encontrados: {len(path_files)} archivos de caminos")
    return path_files

def calculate_measure_error(measure_dir: Path, measure_name: str, gt_mappings: Dict) -> float:
    """Calcula el error agregado para una medida"""
    print(f"  📊 Calculando error para {measure_name}...")
    
    path_files = find_path_files(measure_dir, measure_name)
    
    if not path_files:
        print(f"    ⚠️ No se encontraron archivos de caminos para {measure_name}")
        return np.nan
    
    all_errors = []
    processed_files = 0
    
    for path_file in path_files:
        try:
            print(f"    📄 Procesando: {path_file.name}")
            df = read_csv_flexible(path_file)
            
            if df.empty:
                print(f"    ⚠️ Archivo vacío: {path_file.name}")
                continue
            
            # Verificar si tiene las columnas necesarias para calcular error
            required_cols = ["real_index_x", "ideal_index_x", "real_index_z", "ideal_index_z"]
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                print(f"    ⚠️ Columnas faltantes en {path_file.name}: {missing_cols}")
                print(f"    📋 Columnas disponibles: {list(df.columns)}")
                continue
            
            # Si no hay columna Command, asumir que todos son del mismo comando
            if "Command" not in df.columns:
                print(f"    ⚠️ Sin columna 'Command' en {path_file.name}, asumiendo 'Front'")
                df["Command"] = "Front"
            
            processed_files += 1
            file_errors = []
            
            for cmd, group in df.groupby("Command"):
                if cmd not in COMMANDS:
                    print(f"    ⚠️ Comando desconocido: {cmd}")
                    continue
                
                error_stats = calculate_alignment_error(group.copy(), gt_mappings, cmd)
                
                if not np.isnan(error_stats["e_mean"]):
                    file_errors.append(error_stats["e_mean"])
                    print(f"    ✓ {cmd}: error = {error_stats['e_mean']:.4f}")
            
            if file_errors:
                all_errors.extend(file_errors)
        
        except Exception as e:
            print(f"    ❌ Error procesando {path_file.name}: {e}")
            continue
    
    if all_errors:
        mean_error = np.mean(all_errors)
        print(f"    ✅ Error total calculado: {mean_error:.4f} (de {processed_files} archivos)")
        return mean_error
    else:
        print(f"    ❌ No se pudo calcular error para {measure_name} ({processed_files} archivos procesados)")
        return np.nan

# ══════════════ 8. SISTEMA DE RANKING CORREGIDO (ADAPTADO DEL PROGRAMA 2) ══════════════
def calculate_multiobjetive_ranking_fixed(results_df: pd.DataFrame, 
                                         alpha: float, 
                                         beta: float) -> pd.DataFrame:
    """
    SISTEMA DE RANKING CORREGIDO QUE SÍ RESPONDE A α Y β
    Implementa la metodología del Programa 2 con la robustez del Programa 1
    """
    print(f"\n🎯 APLICANDO OPTIMIZACIÓN MULTIOBJETIVO CON PARÁMETROS INTERACTIVOS")
    print(f"📐 Fórmula: F(x) = {alpha:.1f} × Cost_comp(x) - {beta:.1f} × Accuracy_align(x)")
    print(f"🎯 Objetivo: Minimizar F(x) (menor = mejor balance)")
    print("=" * 70)
    
    df = results_df.copy()
    
    # Mostrar datos originales
    print("\n📊 DATOS ORIGINALES:")
    print("Medida   | Error    | FLOPs     | Tiempo    ")
    print("-" * 45)
    for _, row in df.iterrows():
        print(f"{row['Medida']:8s} | {row['Error_Agregado']:8.3f} | {row['FLOPS_Totales']:9.0f} | {row['Tiempo_Total_s']:8.6f}s")
    
    # 1. NORMALIZACIÓN MIN-MAX (0-1) - MÉTODO DIRECTO
    print(f"\n📐 PASO 1: Normalización Min-Max (0-1)")
    
    # Error: menor es mejor → normalizar directo
    error_min, error_max = df['Error_Agregado'].min(), df['Error_Agregado'].max()
    if error_max != error_min:
        df['Error_norm'] = (df['Error_Agregado'] - error_min) / (error_max - error_min)
    else:
        df['Error_norm'] = 0.5
    
    # FLOPs: menor es mejor → normalizar directo  
    flops_min, flops_max = df['FLOPS_Totales'].min(), df['FLOPS_Totales'].max()
    if flops_max != flops_min:
        df['FLOPS_norm'] = (df['FLOPS_Totales'] - flops_min) / (flops_max - flops_min)
    else:
        df['FLOPS_norm'] = 0.5
    
    # Tiempo: menor es mejor → normalizar directo
    tiempo_min, tiempo_max = df['Tiempo_Total_s'].min(), df['Tiempo_Total_s'].max()
    if tiempo_max != tiempo_min:
        df['Tiempo_norm'] = (df['Tiempo_Total_s'] - tiempo_min) / (tiempo_max - tiempo_min)
    else:
        df['Tiempo_norm'] = 0.5
    
    print(f"   ✓ Error: [{error_min:.3f}, {error_max:.3f}] → normalizado")
    print(f"   ✓ FLOPs: [{flops_min:.0f}, {flops_max:.0f}] → normalizado")
    print(f"   ✓ Tiempo: [{tiempo_min:.6f}, {tiempo_max:.6f}] → normalizado")
    
    # 2. CÁLCULO DE COMPONENTES (MÉTODO SIMPLIFICADO Y DIRECTO)
    print(f"\n🔧 PASO 2: Cálculo de Componentes")
    
    # Cost_comp: Promedio de FLOPs y Tiempo normalizados (menor es mejor)
    df['Cost_comp'] = (df['FLOPS_norm'] + df['Tiempo_norm']) / 2
    
    # Accuracy: Inverso del error normalizado (mayor es mejor)
    df['Accuracy'] = 1.0 - df['Error_norm']
    
    print(f"   ✓ Cost_comp = (FLOPs_norm + Tiempo_norm) / 2")
    print(f"   ✓ Accuracy = 1.0 - Error_norm")
    
    # 3. APLICACIÓN DIRECTA DE LA FÓRMULA F(x)
    print(f"\n⚡ PASO 3: Aplicación de F(x) = {alpha:.1f} × Cost_comp - {beta:.1f} × Accuracy")
    
    # FÓRMULA DIRECTA QUE SÍ DEPENDE DE α Y β
    df['F_x'] = alpha * df['Cost_comp'] - beta * df['Accuracy']
    
    # 4. MOSTRAR CÁLCULOS DETALLADOS
    print(f"\n📋 CÁLCULOS DETALLADOS:")
    print("Medida   | Cost_comp | Accuracy | α×Cost | -β×Acc | F(x)     ")
    print("-" * 60)
    
    for _, row in df.iterrows():
        medida = row['Medida']
        cost_comp = row['Cost_comp']
        accuracy = row['Accuracy']
        alpha_contrib = alpha * cost_comp
        beta_contrib = -beta * accuracy
        f_x = row['F_x']
        
        print(f"{medida:8s} | {cost_comp:8.3f} | {accuracy:7.3f} | "
              f"{alpha_contrib:+6.3f} | {beta_contrib:+6.3f} | {f_x:+7.3f}")
    
    # 5. RANKING FINAL (menor F(x) = mejor)
    df = df.sort_values('F_x')
    df['Ranking_Multiobjetivo'] = range(1, len(df) + 1)
    
    print(f"\n🏆 RANKING FINAL:")
    print("-" * 50)
    
    for _, row in df.iterrows():
        ranking = int(row['Ranking_Multiobjetivo'])
        medida = row['Medida']
        f_x = row['F_x']
        cost_comp = row['Cost_comp']
        accuracy = row['Accuracy']
        
        # Emoji según ranking
        if ranking == 1:
            emoji = "🥇"
        elif ranking == 2:
            emoji = "🥈"
        elif ranking == 3:
            emoji = "🥉"
        else:
            emoji = f"{ranking:2d}."
        
        print(f"{emoji} {medida:8s} → F(x)={f_x:+6.3f} | Cost={cost_comp:.3f} | Acc={accuracy:.3f}")
        
        # Explicar por qué gana el primero
        if ranking == 1:
            if alpha > beta:
                reason = f"GANA por ser MÁS EFICIENTE (α={alpha:.1f} > β={beta:.1f})"
            elif beta > alpha:
                reason = f"GANA por ser MÁS PRECISO (β={beta:.1f} > α={alpha:.1f})"
            else:
                reason = f"GANA por MEJOR BALANCE (α={alpha:.1f} = β={beta:.1f})"
            print(f"     🎯 {reason}")
    
    # 6. REORDENAR COLUMNAS PARA COMPATIBILIDAD
    column_order = [
        "Ranking_Multiobjetivo", "Medida", "F_x",
        "Error_Agregado", "FLOPS_Totales", "Tiempo_Total_s",
        "Cost_comp", "Accuracy", 
        "Error_norm", "FLOPS_norm", "Tiempo_norm",
        "Carpeta_Fuente", "Metodo_Calculo"
    ]
    
    df = df[[col for col in column_order if col in df.columns]]
    
    return df

# ══════════════ 9. EXPORTACIÓN Y REPORTES (MEJORADOS) ══════════════
def export_multiobjetive_analysis_improved(results_df: pd.DataFrame, 
                                          output_dir: Path,
                                          alpha: float,
                                          beta: float):
    """Exporta análisis detallado de optimización multiobjetivo con parámetros interactivos"""
    
    # 1. Archivo de ranking multiobjetivo
    ranking_file = output_dir / f"ranking_multiobjetivo_a{alpha:.1f}_b{beta:.1f}.csv"
    ranking_data = results_df[["Ranking_Multiobjetivo", "Medida", "F_x", 
                              "Error_Agregado", "FLOPS_Totales", "Tiempo_Total_s"]].copy()
    ranking_data.to_csv(ranking_file, index=False, encoding='utf-8')
    
    # 2. Archivo de métricas normalizadas
    normalized_file = output_dir / f"metricas_normalizadas_a{alpha:.1f}_b{beta:.1f}.csv"
    normalized_data = results_df[["Medida", "Cost_comp", "Accuracy", 
                                 "Error_norm", "FLOPS_norm", "Tiempo_norm"]].copy()
    normalized_data.to_csv(normalized_file, index=False, encoding='utf-8')
    
    # 3. Análisis matemático detallado estilo ejemplo
    analysis_file = output_dir / f"analisis_detallado_a{alpha:.1f}_b{beta:.1f}.txt"
    
    with open(analysis_file, 'w', encoding='utf-8') as f:
        f.write("EVALUADOR HÍBRIDO - ANÁLISIS MULTIOBJETIVO DETALLADO\n")
        f.write("Combinando robustez del Programa 1 con interactividad del Programa 2\n")
        f.write("ACTUALIZACIÓN: Incluye LCSS y cDTW (Total: 10 medidas)\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("**Medidas Evaluadas:**\n")
        medidas_evaluadas = results_df["Medida"].tolist()
        for i, medida in enumerate(medidas_evaluadas, 1):
            f.write(f"{i:2d}. {medida}\n")
        f.write(f"\nTotal: {len(medidas_evaluadas)} medidas de similitud\n\n")
        
        f.write("**Paso 1: Definición de la Fórmula**\n")
        f.write("**Función Objetivo:**\n\n")
        f.write("```\n")
        f.write(f"F(x) = α · Cost_comp(x) - β · Accuracy_align(x)\n")
        f.write("```\n\n")
        f.write("**Donde:**\n")
        f.write(f"* α = {alpha:.1f} (peso del costo computacional)\n")
        f.write(f"* β = {beta:.1f} (peso de la precisión)\n")
        f.write("* Menor valor F(x) = Mejor algoritmo\n\n")
        
        f.write("**Paso 2: Normalización de Datos**\n")
        f.write("**Datos Originales:**\n\n")
        f.write("```\n")
        f.write("Algoritmo | Error   | FLOPs     | Tiempo\n")
        f.write("-" * 40 + "\n")
        
        for _, row in results_df.iterrows():
            f.write(f"{row['Medida']:9s} | {row['Error_Agregado']:7.3f} | {row['FLOPS_Totales']:9.0f} | {row['Tiempo_Total_s']:.6f}\n")
        
        f.write("```\n\n")
        f.write("**Normalización Min-Max (0-1):**\n")
        f.write("**Para Costo Computacional (FLOPs + Tiempo):**\n")
        f.write("* Cost_comp = (FLOPs_norm + Tiempo_norm) / 2\n")
        f.write("**Para Precisión (inverso del Error):**\n")
        f.write("* Accuracy = 1 - Error_normalizado\n\n")
        
        f.write("**Paso 3: Cálculos Detallados**\n")
        f.write("**Cálculo de Cost_comp:**\n\n")
        f.write("```\n")
        for _, row in results_df.iterrows():
            f.write(f"{row['Medida']}: Cost_comp = ({row['FLOPS_norm']:.3f} + {row['Tiempo_norm']:.3f}) / 2 = {row['Cost_comp']:.3f}\n")
        f.write("```\n\n")
        
        f.write("**Cálculo de Accuracy:**\n\n")
        f.write("```\n")
        for _, row in results_df.iterrows():
            f.write(f"{row['Medida']}: Accuracy = 1 - {row['Error_norm']:.3f} = {row['Accuracy']:.3f}\n")
        f.write("```\n\n")
        
        f.write("**Paso 4: Aplicación de la Fórmula**\n\n")
        f.write("```\n")
        f.write(f"F(x) = {alpha:.1f} × Cost_comp - {beta:.1f} × Accuracy\n")
        f.write("```\n\n")
        f.write("**Cálculos Finales:**\n\n")
        f.write("```\n")
        f.write("Algoritmo | Cost_comp | Accuracy | F(x) = α×Cost - β×Acc | Ranking\n")
        f.write("-" * 65 + "\n")
        
        for _, row in results_df.iterrows():
            ranking = int(row['Ranking_Multiobjetivo'])
            cost_comp = row['Cost_comp']
            accuracy = row['Accuracy']
            f_x = row['F_x']
            alpha_term = alpha * cost_comp
            beta_term = beta * accuracy
            
            f.write(f"{row['Medida']:9s} | {cost_comp:8.3f} | {accuracy:7.3f} | "
                   f"{alpha_term:.3f} - {beta_term:.3f} = {f_x:+6.3f} | {ranking}º\n")
        
        f.write("```\n\n")
        
        f.write("**Paso 5: Ranking Final**\n")
        f.write("**🏆 RANKING DE MEDIDAS ELÁSTICAS (ACTUALIZADO CON 10 MEDIDAS)**\n\n")
        f.write("```\n")
        
        for _, row in results_df.head(10).iterrows():
            ranking = int(row["Ranking_Multiobjetivo"])
            medida = row["Medida"]
            f_x = row["F_x"]
            
            if ranking == 1:
                symbol = "🥇"
            elif ranking == 2:
                symbol = "🥈"
            elif ranking == 3:
                symbol = "🥉"
            else:
                symbol = f"{ranking}º"
            
            f.write(f"{symbol} {medida:8s} → F(x) = {f_x:+6.3f}")
            if ranking == 1:
                f.write(" (GANADOR)")
            f.write("\n")
        
        f.write("```\n\n")
        
        # Justificación del resultado
        best = results_df.iloc[0]
        f.write("**Justificación del Resultado**\n")
        f.write(f"**{best['Medida']} resulta ganador porque:**\n")
        f.write(f"1. **Cost_comp = {best['Cost_comp']:.3f}** (eficiencia computacional)\n")
        f.write(f"2. **Accuracy = {best['Accuracy']:.3f}** (precisión de alineación)\n")
        f.write(f"3. **Balance óptimo** con α={alpha:.1f}, β={beta:.1f}\n")
        
        if alpha > beta:
            f.write(f"4. **Configuración favorece EFICIENCIA** (α > β)\n")
        elif beta > alpha:
            f.write(f"4. **Configuración favorece PRECISIÓN** (β > α)\n")
        else:
            f.write(f"4. **Configuración BALANCEADA** (α = β)\n")
        
        f.write(f"\n**La configuración α={alpha:.1f}, β={beta:.1f} refleja:**\n")
        f.write(f"* {'Alta' if alpha > 0.7 else 'Media' if alpha > 0.3 else 'Baja'} importancia de la eficiencia computacional\n")
        f.write(f"* {'Alta' if beta > 0.7 else 'Media' if beta > 0.3 else 'Baja'} consideración de la precisión\n")
        f.write(f"* Balance {'orientado a eficiencia' if alpha > beta else 'orientado a precisión' if beta > alpha else 'equilibrado'} para aplicaciones modernas\n\n")
        
        f.write("**Inclusión de Nuevas Medidas**\n")
        nuevas_medidas = [m for m in medidas_evaluadas if m in ["LCSS", "cDTW"]]
        if nuevas_medidas:
            f.write(f"En esta evaluación se incluyeron las nuevas medidas: {', '.join(nuevas_medidas)}\n")
            for medida in nuevas_medidas:
                row = results_df[results_df['Medida'] == medida].iloc[0]
                ranking = int(row['Ranking_Multiobjetivo'])
                f.write(f"• {medida}: Posición #{ranking} con F(x) = {row['F_x']:+.3f}\n")
        f.write("\n")
        
        f.write("**Conclusión**\n")
        f.write(f"El análisis multiobjetivo confirma que **{best['Medida']}** ofrece la mejor relación ")
        f.write("costo-beneficio para aplicaciones de alineamiento de series temporales con los parámetros ")
        f.write(f"especificados (α={alpha:.1f}, β={beta:.1f}).\n")
        f.write(f"La evaluación incluye ahora {len(medidas_evaluadas)} medidas de similitud, ")
        f.write("proporcionando una comparación más completa del estado del arte.\n")
    
    print(f"✓ Ranking multiobjetivo: {ranking_file}")
    print(f"✓ Métricas normalizadas: {normalized_file}")
    print(f"✓ Análisis detallado: {analysis_file}")
    
    return ranking_file, normalized_file, analysis_file

# ══════════════ 10. VISUALIZACIONES MEJORADAS CON 10 MEDIDAS ══════════════
def generate_improved_plots(results_df: pd.DataFrame, output_dir: Path, alpha: float, beta: float):
    """Genera gráficos mejorados para 10 medidas de similitud"""
    output_dir.mkdir(exist_ok=True)
    
    plt.style.use('default')
    fig_size = (16, 10)  # Aumentamos el tamaño para 10 medidas
    
    # 1. Gráfico principal: F(x) - Función objetivo con valores claramente visibles
    plt.figure(figsize=fig_size)
    
    # Usar colores diferenciados para cada posición (ahora 10 colores)
    colors = ['#FFD700', '#C0C0C0', '#CD7F32', '#87CEEB', '#98FB98', 
              '#F0E68C', '#DDA0DD', '#B0C4DE', '#FFA07A', '#20B2AA']
    colors = colors[:len(results_df)] + ['#87CEEB'] * max(0, len(results_df) - len(colors))
    
    bars = plt.bar(results_df["Medida"], results_df["F_x"], 
                   color=colors, edgecolor='navy', linewidth=1.5, alpha=0.8)
    
    # Línea de referencia en F(x) = 0
    plt.axhline(y=0, color='red', linestyle='--', alpha=0.7, linewidth=2, label='F(x) = 0')
    
    plt.title(f"Ranking por Optimización Multiobjetivo - 10 Medidas\nF(x) = {alpha:.1f}·Cost_comp - {beta:.1f}·Accuracy\n(Menor F(x) = Mejor)", 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel("Medida de Similitud", fontsize=14, fontweight='bold')
    plt.ylabel("F(x) - Función Objetivo", fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=11)
    plt.yticks(fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    plt.legend(fontsize=12)
    
    # Añadir valores GRANDES y CLAROS en las barras
    for i, (bar, value) in enumerate(zip(bars, results_df["F_x"])):
        height = bar.get_height()
        
        # Valor F(x) encima/debajo de la barra
        offset = 0.02 if height >= 0 else -0.03
        plt.text(bar.get_x() + bar.get_width()/2., height + offset,
                f'{value:.3f}', ha='center', va='bottom' if height >= 0 else 'top', 
                fontweight='bold', fontsize=10, 
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
        
        # Ranking dentro de la barra
        ranking = i + 1
        text_y = height/2 if abs(height) > 0.05 else 0
        plt.text(bar.get_x() + bar.get_width()/2., text_y,
                f'#{ranking}', ha='center', va='center', fontsize=12, 
                fontweight='bold', color='white' if abs(height) > 0.1 else 'black')
    
    plt.tight_layout()
    plt.savefig(output_dir / f"ranking_funcion_objetivo_10medidas_a{alpha:.1f}_b{beta:.1f}.png", 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Gráfico de barras comparativo de métricas normalizadas
    plt.figure(figsize=(18, 10))
    
    # Preparar datos para gráfico de barras agrupadas
    medidas = results_df["Medida"].tolist()
    x = np.arange(len(medidas))
    width = 0.25
    
    # Crear barras agrupadas
    bars1 = plt.bar(x - width, results_df["Error_norm"], width, 
                   label='Error (normalizado)', color='lightcoral', alpha=0.8)
    bars2 = plt.bar(x, results_df["Cost_comp"], width, 
                   label='Cost_comp', color='lightblue', alpha=0.8)
    bars3 = plt.bar(x + width, results_df["Accuracy"], width, 
                   label='Accuracy', color='lightgreen', alpha=0.8)
    
    plt.title(f'Comparación de Métricas Normalizadas - 10 Medidas\n(α={alpha:.1f}, β={beta:.1f})', 
              fontsize=16, fontweight='bold')
    plt.xlabel('Medidas de Similitud', fontsize=14, fontweight='bold')
    plt.ylabel('Valor Normalizado [0-1]', fontsize=14, fontweight='bold')
    plt.xticks(x, medidas, rotation=45, ha='right', fontsize=11)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    
    # Añadir valores en las barras
    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    add_value_labels(bars1)
    add_value_labels(bars2)
    add_value_labels(bars3)
    
    plt.tight_layout()
    plt.savefig(output_dir / f"metricas_comparadas_10medidas_a{alpha:.1f}_b{beta:.1f}.png", 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Gráfico de ranking horizontal con detalles
    plt.figure(figsize=(14, 12))  # Más alto para 10 medidas
    
    # Crear gráfico horizontal para mejor legibilidad
    y_pos = np.arange(len(results_df))
    
    # Barras horizontales con F(x)
    bars = plt.barh(y_pos, results_df["F_x"], 
                   color=colors[:len(results_df)], alpha=0.8, edgecolor='black')
    
    plt.yticks(y_pos, [f"{int(row['Ranking_Multiobjetivo'])}º {row['Medida']}" 
                      for _, row in results_df.iterrows()], fontsize=11)
    plt.xlabel('F(x) - Función Objetivo', fontsize=14, fontweight='bold')
    plt.title(f'Ranking Detallado - 10 Medidas (α={alpha:.1f}, β={beta:.1f})\nMenor F(x) = Mejor Posición', 
              fontsize=16, fontweight='bold')
    plt.grid(axis='x', alpha=0.3)
    
    # Añadir valores al final de cada barra
    for i, (bar, row) in enumerate(zip(bars, results_df.itertuples())):
        width = bar.get_width()
        plt.text(width + (0.01 if width >= 0 else -0.01), bar.get_y() + bar.get_height()/2,
                f'{width:.3f}', ha='left' if width >= 0 else 'right', va='center', 
                fontweight='bold', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_dir / f"ranking_detallado_10medidas_a{alpha:.1f}_b{beta:.1f}.png", 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Gráficos mejorados para 10 medidas generados en: {output_dir}")

# ══════════════ 11. PIPELINE PRINCIPAL HÍBRIDO ACTUALIZADO ══════════════
def evaluate_all_similarity_measures_hybrid(base_path: Path, 
                                           alpha: float = None,
                                           beta: float = None,
                                           export_excel: bool = True) -> pd.DataFrame:
    """Pipeline principal híbrido con parámetros interactivos - ACTUALIZADO PARA 10 MEDIDAS"""
    
    print("🚀 EVALUADOR HÍBRIDO DE MEDIDAS DE SIMILITUD - VERSIÓN ACTUALIZADA")
    print("Programa 1 (robusto) + Programa 2 (interactivo) + LCSS y cDTW")
    print("=" * 70)
    print(f"📊 Medidas a evaluar: {', '.join(SIMILARITY_MEASURES.keys())}")
    print(f"🔢 Total: {len(SIMILARITY_MEASURES)} medidas de similitud")
    
    # 1. Solicitar parámetros si no se proporcionan
    if alpha is None or beta is None:
        alpha, beta = solicitar_parametros_usuario()
    
    print(f"\n📐 Fórmula configurada: F(x) = {alpha:.1f}·Cost_comp(x) - {beta:.1f}·Accuracy_align(x)")
    print("🎯 Objetivo: Minimizar F(x) para balance óptimo costo-precisión")
    
    # 2. Cargar ground-truth (del Programa 1)
    print("\n📖 PASO 1: Cargando Ground-Truth...")
    gt_path = None
    possible_gt_paths = [
        base_path / "ground_truth_salida" / "ground_truth_todo.xlsx",
        base_path / "ground_truth_salida" / "ground_truth_todo.csv",
        base_path / "ground_truth_salida" / "ground_truth.xlsx",
        base_path / "ground_truth_salida" / "ground_truth.csv",
        base_path / "ground_truth_todo.xlsx",
        base_path / "ground_truth_todo.csv"
    ]
    
    for path in possible_gt_paths:
        if path.exists():
            gt_path = path
            break
    
    if gt_path is None:
        gt_dir = base_path / "ground_truth_salida"
        if gt_dir.exists():
            excel_files = list(gt_dir.glob("*.xlsx")) + list(gt_dir.glob("*.xls"))
            csv_files = list(gt_dir.glob("*.csv"))
            gt_path = excel_files[0] if excel_files else (csv_files[0] if csv_files else None)
    
    if gt_path is None:
        raise FileNotFoundError("❌ No se encontró archivo ground-truth")
    
    print(f"✓ Ground-truth encontrado: {gt_path}")
    gt_df = read_csv_flexible(gt_path)
    gt_mappings = build_ground_truth_mappings(gt_df)
    
    # 3. Identificar carpetas de medidas (del Programa 1 - ACTUALIZADO)
    print("\n📂 PASO 2: Identificando Carpetas de Medidas (Incluyendo LCSS y cDTW)...")
    measure_directories = find_measure_directories(base_path)
    
    if not measure_directories:
        raise FileNotFoundError("❌ No se encontraron carpetas de medidas")
    
    # Verificar si se encontraron las nuevas medidas
    nuevas_encontradas = [m for m in ["LCSS", "cDTW"] if m in measure_directories]
    if nuevas_encontradas:
        print(f"🎉 Nuevas medidas encontradas: {', '.join(nuevas_encontradas)}")
    else:
        print("⚠️ No se encontraron las nuevas medidas LCSS y/o cDTW")
    
    # 4. Procesar cada medida (del Programa 1)
    print("\n📊 PASO 3: Procesando Medidas...")
    results = []
    
    for measure_name, measure_dir in measure_directories.items():
        print(f"\n🔬 Procesando {measure_name}...")
        print(f"   📁 Carpeta: {measure_dir.name}")
        
        # 4.1 Extraer métricas usando lógica del Programa 1
        flops_tiempo_metrics = procesar_medida_elastica_flops_tiempo(measure_dir, measure_name)
        
        # 4.2 Calcular error agregado
        error_agregado = calculate_measure_error(measure_dir, measure_name, gt_mappings)
        
        # 4.3 Consolidar resultados
        result = {
            "Medida": measure_name,
            "Error_Agregado": error_agregado,
            "FLOPS_Totales": flops_tiempo_metrics["total_flops"],
            "Tiempo_Total_s": flops_tiempo_metrics["total_time_seconds"],
            "Carpeta_Fuente": measure_dir.name,
            "Metodo_Calculo": f"Hybrid_10medidas_a{alpha:.1f}_b{beta:.1f}"
        }
        
        results.append(result)
        
        print(f"    ✓ Completado: Error={error_agregado:.4f}, "
              f"FLOPS={flops_tiempo_metrics['total_flops']:.2e}, "
              f"Tiempo={flops_tiempo_metrics['total_time_seconds']:.6f}s")
    
    # 5. Crear DataFrame de resultados
    results_df = pd.DataFrame(results)
    
    print(f"\n📊 RESUMEN DE MEDIDAS PROCESADAS:")
    print(f"   Total medidas evaluadas: {len(results_df)}")
    print(f"   Medidas con datos válidos: {len(results_df.dropna())}")
    
    # 6. Aplicar optimización multiobjetivo CORREGIDA (del Programa 2)
    print("\n🎯 PASO 4: Aplicando Optimización Multiobjetivo Corregida...")
    results_df = calculate_multiobjetive_ranking_fixed(results_df, alpha, beta)
    
    # 7. Generar reportes
    print("\n📋 PASO 5: Generando Reportes...")
    reports_dir = base_path / f"evaluacion_hibrida_10medidas_a{alpha:.1f}_b{beta:.1f}"
    reports_dir.mkdir(exist_ok=True)
    
    # 7.1 Exportar análisis mejorado
    ranking_file, normalized_file, analysis_file = export_multiobjetive_analysis_improved(
        results_df, reports_dir, alpha, beta)
    
    # 7.2 Reporte detallado en Excel
    if export_excel:
        excel_file = reports_dir / f"evaluacion_hibrida_10medidas_a{alpha:.1f}_b{beta:.1f}_completa.xlsx"
        
        with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
            # Hoja principal con ranking multiobjetivo
            results_df.to_excel(writer, sheet_name='Ranking_Multiobjetivo', index=False)
            
            # Hoja con métricas originales
            raw_metrics = results_df[["Ranking_Multiobjetivo", "Medida", "F_x",
                                     "Error_Agregado", "FLOPS_Totales", "Tiempo_Total_s"]]
            raw_metrics.to_excel(writer, sheet_name='Metricas_Originales', index=False)
            
            # Hoja con componentes de optimización
            opt_components = results_df[["Medida", "Cost_comp", "Accuracy", 
                                        "Error_norm", "FLOPS_norm", "Tiempo_norm", "F_x"]]
            opt_components.to_excel(writer, sheet_name='Componentes_Optimizacion', index=False)
            
            # Hoja de parámetros
            params_data = {
                "Parametro": ["alpha", "beta", "formula", "objetivo", "ganador", "f_x_optimo", 
                             "configuracion", "total_medidas", "nuevas_medidas"],
                "Valor": [alpha, beta, f"F(x) = {alpha}·Cost_comp - {beta}·Accuracy",
                         "Minimizar F(x)", results_df.iloc[0]["Medida"], 
                         f"{results_df.iloc[0]['F_x']:.3f}",
                         f"{'Prioriza eficiencia' if alpha > beta else 'Prioriza precisión' if beta > alpha else 'Balance equilibrado'}",
                         len(results_df),
                         ", ".join([m for m in results_df["Medida"] if m in ["LCSS", "cDTW"]])]
            }
            params_df = pd.DataFrame(params_data)
            params_df.to_excel(writer, sheet_name='Parametros', index=False)
        
        print(f"✓ Reporte Excel: {excel_file}")
    
    # 8. Generar gráficos mejorados
    print("\n📊 PASO 6: Generando Gráficos Mejorados para 10 Medidas...")
    plots_dir = reports_dir / "graficos"
    generate_improved_plots(results_df, plots_dir, alpha, beta)
    
    # 9. Mostrar resultados finales
    print("\n🏆 RESULTADOS FINALES - RANKING HÍBRIDO CON 10 MEDIDAS")
    print("=" * 70)
    print(f"Fórmula: F(x) = {alpha:.1f}·Cost_comp - {beta:.1f}·Accuracy")
    print("(Menor F(x) = Mejor balance costo-precisión)")
    print()
    
    for _, row in results_df.head(len(results_df)).iterrows():
        ranking = int(row["Ranking_Multiobjetivo"])
        medida = row["Medida"]
        f_x = row["F_x"]
        error = row["Error_Agregado"]
        flops = row["FLOPS_Totales"]
        tiempo = row["Tiempo_Total_s"]
        
        # Emoji según ranking y destacar nuevas medidas
        if ranking == 1:
            emoji = "🥇"
        elif ranking == 2:
            emoji = "🥈"
        elif ranking == 3:
            emoji = "🥉"
        else:
            emoji = f"{ranking:2d}."
        
        # Destacar nuevas medidas
        nueva_marca = " 🆕" if medida in ["LCSS", "cDTW"] else ""
        
        print(f"{emoji} {medida:8s}{nueva_marca} → F(x)={f_x:+6.3f} | "
              f"Error: {error:6.3f} | FLOPS: {flops:8.0f} | "
              f"Tiempo: {tiempo:8.6f}s")
    
    # 10. Análisis final del ganador
    best = results_df.iloc[0]
    print(f"\n🎯 ANÁLISIS DEL GANADOR: {best['Medida']}")
    print("-" * 50)
    print(f"• F(x) óptimo: {best['F_x']:.3f}")
    print(f"• Cost_comp: {best['Cost_comp']:.3f} (eficiencia computacional)")
    print(f"• Accuracy: {best['Accuracy']:.3f} (precisión de alineación)")
    print(f"• Error original: {best['Error_Agregado']:.4f}")
    print(f"• FLOPs: {best['FLOPS_Totales']:.0f}")
    print(f"• Tiempo: {best['Tiempo_Total_s']:.6f}s")
    
    # Verificar si es una nueva medida
    if best['Medida'] in ["LCSS", "cDTW"]:
        print(f"🎉 ¡El ganador es una NUEVA MEDIDA agregada!")
    
    # Interpretación de la configuración
    if alpha > beta:
        interpretation = f"PRIORIZA EFICIENCIA (α={alpha:.1f} > β={beta:.1f})"
    elif beta > alpha:
        interpretation = f"PRIORIZA PRECISIÓN (β={beta:.1f} > α={alpha:.1f})"
    else:
        interpretation = f"BALANCE EQUILIBRADO (α={alpha:.1f} = β={beta:.1f})"
    
    print(f"• Configuración: {interpretation}")
    
    # Análisis de las nuevas medidas
    nuevas_en_resultados = results_df[results_df['Medida'].isin(["LCSS", "cDTW"])]
    if not nuevas_en_resultados.empty:
        print(f"\n📊 RENDIMIENTO DE NUEVAS MEDIDAS:")
        print("-" * 50)
        for _, row in nuevas_en_resultados.iterrows():
            ranking = int(row['Ranking_Multiobjetivo'])
            medida = row['Medida']
            f_x = row['F_x']
            print(f"• {medida}: Posición #{ranking} con F(x) = {f_x:+6.3f}")
            
            if ranking <= 3:
                print(f"  🎉 ¡Excelente! {medida} está en el TOP 3")
            elif ranking <= 5:
                print(f"  ✅ Buen rendimiento de {medida}")
            else:
                print(f"  📊 {medida} muestra rendimiento competitivo")
    
    print(f"\n📁 ARCHIVOS GENERADOS")
    print("-" * 50)
    print(f"🎯 Ranking: {ranking_file}")
    print(f"📊 Métricas: {normalized_file}")
    print(f"📝 Análisis: {analysis_file}")
    if export_excel:
        print(f"📈 Excel: {excel_file}")
    print(f"📊 Gráficos: {plots_dir}")
    print(f"📂 Directorio: {reports_dir}")
    
    return results_df

# ══════════════ 12. FUNCIÓN PRINCIPAL INTERACTIVA ACTUALIZADA ══════════════
def main_hybrid(base_path: Path = Path("."), 
               alpha: float = None,
               beta: float = None,
               export_excel: bool = True):
    """Función principal del evaluador híbrido - ACTUALIZADA PARA 10 MEDIDAS"""
    
    start_time = time.perf_counter()
    
    try:
        # Ejecutar evaluación híbrida
        results_df = evaluate_all_similarity_measures_hybrid(
            base_path, alpha, beta, export_excel)
        
        execution_time = time.perf_counter() - start_time
        
        print(f"\n✅ EVALUACIÓN HÍBRIDA COMPLETADA CON 10 MEDIDAS")
        print(f"⏱️ Tiempo total de ejecución: {execution_time:.2f} segundos")
        print(f"📊 {len(results_df)} medidas evaluadas")
        print(f"🏆 Ganador: {results_df.iloc[0]['Medida']} (F(x) = {results_df.iloc[0]['F_x']:.3f})")
        
        # Mostrar qué tipo de configuración se usó
        alpha_used = float(results_df.iloc[0]['Metodo_Calculo'].split('_a')[1].split('_b')[0])
        beta_used = float(results_df.iloc[0]['Metodo_Calculo'].split('_b')[1])
        
        if alpha_used > beta_used:
            config_type = "EFICIENCIA"
        elif beta_used > alpha_used:
            config_type = "PRECISIÓN"
        else:
            config_type = "BALANCE"
        
        print(f"🎯 Configuración: {config_type} (α={alpha_used:.1f}, β={beta_used:.1f})")
        
        # Resumen de nuevas medidas
        nuevas_medidas = [m for m in results_df['Medida'] if m in ["LCSS", "cDTW"]]
        if nuevas_medidas:
            print(f" Nuevas medidas incluidas: {', '.join(nuevas_medidas)}")
            mejor_nueva = min([(int(results_df[results_df['Medida'] == m].iloc[0]['Ranking_Multiobjetivo']), m) 
                              for m in nuevas_medidas])
            print(f"🏅 Mejor nueva medida: {mejor_nueva[1]} (posición #{mejor_nueva[0]})")
        
        return results_df
        
    except Exception as e:
        print(f"\n❌ Error durante la evaluación: {e}")
        import traceback
        traceback.print_exc()
        return None

# ══════════════ 13. FUNCIÓN DE VERIFICACIÓN ACTUALIZADA ══════════════
def test_parametros_diferentes():
    """
    Función de prueba para verificar que diferentes α y β producen diferentes rankings
    """
    print("🧪 PRUEBA DE VERIFICACIÓN - DIFERENTES PARÁMETROS CON 10 MEDIDAS")
    print("=" * 65)
    
    # Datos sintéticos con 10 medidas incluidas LCSS y cDTW
    test_data = pd.DataFrame({
        "Medida": ["EFICIENTE", "BALANCEADO", "PRECISO", "LENTO", "MSM", "ERP", "TWE", "LCSS", "cDTW", "ADTW"],
        "Error_Agregado": [0.40, 0.20, 0.10, 0.50, 0.25, 0.30, 0.35, 0.15, 0.22, 0.28],
        "FLOPS_Totales": [500, 2000, 8000, 3000, 1500, 4000, 6000, 1200, 1800, 2500],
        "Tiempo_Total_s": [0.001, 0.005, 0.020, 0.025, 0.008, 0.015, 0.018, 0.006, 0.007, 0.012],
        "Carpeta_Fuente": ["test"] * 10,
        "Metodo_Calculo": ["test"] * 10
    })
    
    print("📊 DATOS DE PRUEBA PARA 10 MEDIDAS:")
    print(test_data[["Medida", "Error_Agregado", "FLOPS_Totales", "Tiempo_Total_s"]].to_string(index=False))
    
    # Probar 3 configuraciones diferentes
    configs = [
        (0.9, 0.1, "PRIORIZA EFICIENCIA"),
        (0.5, 0.5, "BALANCE"),
        (0.1, 0.9, "PRIORIZA PRECISIÓN")
    ]
    
    ganadores = []
    
    for alpha, beta, descripcion in configs:
        print(f"\n{'='*50}")
        print(f"🎯 {descripcion} (α={alpha}, β={beta})")
        print(f"{'='*50}")
        
        resultado = calculate_multiobjetive_ranking_fixed(test_data.copy(), alpha, beta)
        ganador = resultado.iloc[0]["Medida"]
        ganadores.append(ganador)
        
        print(f"🏆 GANADOR: {ganador}")
        
        # Mostrar top 3 para ver variación
        print("🥇🥈🥉 TOP 3:")
        for i in range(min(3, len(resultado))):
            medida = resultado.iloc[i]["Medida"]
            f_x = resultado.iloc[i]["F_x"]
            nueva = " " if medida in ["LCSS", "cDTW"] else ""
            print(f"   {i+1}. {medida}{nueva} (F(x) = {f_x:+.3f})")
    
    print(f"\n📊 RESUMEN DE GANADORES:")
    print(f"   α=0.9, β=0.1 (eficiencia): {ganadores[0]}")
    print(f"   α=0.5, β=0.5 (balance):    {ganadores[1]}")
    print(f"   α=0.1, β=0.9 (precisión):  {ganadores[2]}")
    
    ganadores_unicos = len(set(ganadores))
    print(f"\n✅ GANADORES ÚNICOS: {ganadores_unicos}/3")
    
    if ganadores_unicos > 1:
        print("🎉 ¡LA PRUEBA ES EXITOSA! El sistema SÍ cambia según α y β")
        
        # Verificar si alguna nueva medida ganó
        nuevas_ganadoras = [g for g in ganadores if g in ["LCSS", "cDTW"]]
        if nuevas_ganadoras:
            print(f" ¡Nueva(s) medida(s) ganadora(s): {', '.join(set(nuevas_ganadoras))}!")
        
        return True
    else:
        print("❌ FALLO: El mismo ganador siempre")
        return False

# ══════════════ 14. PUNTO DE ENTRADA PRINCIPAL ACTUALIZADO ══════════════
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Evaluador Híbrido de Medidas de Similitud (Programa 1 + Programa 2) - ACTUALIZADO CON 10 MEDIDAS"
    )
    parser.add_argument("--base", type=Path, default=".", 
                       help="Directorio base (default: directorio actual)")
    parser.add_argument("--alpha", type=float, default=None,
                       help="Peso del costo computacional (0.0-1.0)")
    parser.add_argument("--beta", type=float, default=None,
                       help="Peso de la precisión (0.0-1.0)")
    parser.add_argument("--no-excel", action="store_true", 
                       help="No generar archivo Excel")
    parser.add_argument("--test", action="store_true",
                       help="Ejecutar prueba de verificación con 10 medidas")
    
    args = parser.parse_args()
    
    # Ejecutar prueba si se solicita
    if args.test:
        test_parametros_diferentes()
        exit(0)
    
    # Validar parámetros si se proporcionan
    if args.alpha is not None and args.beta is not None:
        if not (0 <= args.alpha <= 1 and 0 <= args.beta <= 1):
            print("❌ Los parámetros α y β deben estar entre 0.0 y 1.0")
            exit(1)
        
        if abs(args.alpha + args.beta - 1.0) > 0.1:
            print(f"⚠️ Advertencia: α + β = {args.alpha + args.beta:.2f} (recomendado: ~1.0)")
    
    # Ejecutar evaluación principal
    results_df = main_hybrid(
        base_path=args.base.resolve(),
        alpha=args.alpha,
        beta=args.beta,
        export_excel=not args.no_excel
    )
    
    if results_df is not None:
        print(f"\n🎉 EVALUACIÓN HÍBRIDA EXITOSA CON {len(SIMILARITY_MEASURES)} MEDIDAS")
        print(f"Combina la robustez del Programa 1 con la interactividad del Programa 2")
        print(f" Incluye las nuevas medidas: LCSS y cDTW")

# ══════════════ EJEMPLOS DE USO ACTUALIZADOS ══════════════
"""
🎯 EJEMPLOS DE USO DEL EVALUADOR HÍBRIDO ACTUALIZADO (10 MEDIDAS):

1. MODO INTERACTIVO (recomendado):
   python evaluador_hibrido_10medidas.py
   
   -> Te pedirá α y β interactivamente
   -> Procesará las 10 medidas: MSM, ERP, TWE, ADTW, ESDTW, SSDTW, WDTW, BDTW, LCSS, cDTW
   -> Ranking que SÍ cambia como Programa 2

2. PARÁMETROS DIRECTOS:
   python evaluador_hibrido_10medidas.py --alpha 0.8 --beta 0.2
   python evaluador_hibrido_10medidas.py --alpha 0.2 --beta 0.8
   
3. VERIFICAR QUE FUNCIONA CON 10 MEDIDAS:
   python evaluador_hibrido_10medidas.py --test
   
   -> Ejecuta prueba con datos sintéticos de 10 medidas
   -> Verifica que diferentes α,β dan diferentes rankings
   -> Incluye LCSS y cDTW en la prueba

4. DESDE CÓDIGO PYTHON:
   from evaluador_hibrido_10medidas import main_hybrid, test_parametros_diferentes
   
   # Con parámetros específicos
   results = main_hybrid(Path("."), alpha=0.7, beta=0.3)
   
   # Interactivo
   results = main_hybrid(Path("."))
   
   # Verificación con 10 medidas
   test_parametros_diferentes()

═══════════════════════════════════════════════════════════════
🔧 CARACTERÍSTICAS DEL EVALUADOR HÍBRIDO ACTUALIZADO:
═══════════════════════════════════════════════════════════════

✅ TODAS LAS CARACTERÍSTICAS ANTERIORES PLUS:
• ✨ Soporte para LCSS (Longest Common Subsequence)
• ✨ Soporte para cDTW (Constrained Dynamic Time Warping)
• 📊 Evaluación completa de 10 medidas de similitud
• 🎨 Gráficos adaptados para mostrar 10 medidas claramente
• 📋 Reportes que destacan las nuevas medidas incluidas
• 🔍 Detección automática de LCSS_results_matrices_flops y cDTW_results_matrices_flops

🆕 NUEVAS MEDIDAS INCLUIDAS:
═══════════════════════════════════════════════════════════════
📌 LCSS (Longest Common Subsequence):
   • Encuentra subsecuencia común más larga
   • Robusto ante ruido y valores faltantes
   • Complejidad O(mn) - eficiente
   • Ideal para series con discontinuidades

📌 cDTW (Constrained Dynamic Time Warping):
   • DTW con restricciones de ventana
   • Reduce complejidad vs DTW clásico
   • Evita alineaciones patológicas
   • Balance óptimo flexibilidad-eficiencia

🧪 PARA VERIFICAR CON 10 MEDIDAS:
python evaluador_hibrido_10medidas.py --test

¡AHORA CON 10 MEDIDAS DE SIMILITUD PARA COMPARACIÓN COMPLETA!
═══════════════════════════════════════════════════════════════
"""
