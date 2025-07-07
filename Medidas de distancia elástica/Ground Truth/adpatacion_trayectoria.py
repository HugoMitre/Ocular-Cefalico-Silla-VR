#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Adaptador de longitud de trayectorias:
1. Carga de datos → Procesa archivos CSV con trayectorias reales e ideales desde "DTW_dataset"
2. Adaptación de longitud → Ajusta la longitud de las trayectorias reales para que coincidan con las ideales
3. Guardado → Guarda los archivos adaptados en "DTW_dataset_adaptada" con la misma estructura
"""

from __future__ import annotations
import argparse, re, unicodedata, time, glob
from pathlib import Path
from typing import List, Tuple, Dict, Any
import numpy as np
import pandas as pd
import pyarrow.feather as paf
import pyarrow.csv as pac
from numpy.typing import NDArray

# ══════════════ 1. CONSTANTES ══════════════
COMMANDS = [  # Lista de comandos de movimiento reconocidos
    "Right Turn", "Front", "Left Turn",
    "Front-Right Diagonal", "Front-Left Diagonal",
    "Back-Left Diagonal", "Back-Right Diagonal", "Back"
]
CMD_CAT = pd.api.types.CategoricalDtype(COMMANDS, ordered=True)  # Define orden para los comandos
FloatArray = NDArray[np.floating]  # Abreviatura para arrays de números decimales

# ══════════════ 2. UTILIDADES DE LECTURA ══════════════
def _strip(txt: str) -> str:
    # Elimina acentos y caracteres especiales del texto
    return "".join(ch for ch in unicodedata.normalize("NFD", txt)
                   if unicodedata.category(ch) != "Mn")

def canon_cmd(raw: str) -> str | None:
    # Convierte cualquier texto a uno de los comandos estándar
    if not isinstance(raw, str): return None  # Si no es texto, retorna None
    
    # Limpia el texto: quita símbolos, normaliza espacios y convierte a minúsculas
    s = re.sub(r"\s+", " ", re.sub(r"[^a-zA-Z ]+", " ", _strip(raw))).strip().lower()
    diag = ("diag" in s) or ("diagonal" in s)  # Detecta si menciona diagonal
    
    # Serie de verificaciones para determinar qué comando es
    if "right" in s and "turn" in s and "front" not in s:  return "Right Turn"
    if "left"  in s and "turn" in s and "front" not in s:  return "Left Turn"
    if s == "front" or ("front" in s and not diag and "turn" not in s): return "Front"
    if "front" in s and "right" in s and diag:  return "Front-Right Diagonal"
    if "front" in s and "left"  in s and diag:  return "Front-Left Diagonal"
    if "back"  in s and "left"  in s and diag:  return "Back-Left Diagonal"
    if "back"  in s and "right" in s and diag:  return "Back-Right Diagonal"
    if s == "back" or ("back" in s and not diag):          return "Back"
    
    return None  # Si no coincide con ningún patrón conocido

def meta(path: Path) -> Tuple[str, str]:
    # Extrae ID de participante y número de etapa de la ruta del archivo
    # Busca "participant_X" en las partes de la ruta
    pid = next((p.split("_")[1] for p in path.parts if p.lower().startswith("participant_")), "unknown")
    
    # Busca "etapaX" en la ruta
    m = re.search(r"etapa[_\-]?(\d)", str(path), re.I)
    
    return pid, (m.group(1) if m else "1")  # Retorna ID y etapa (1 por defecto)

def load_ideal(base: Path) -> pd.DataFrame:
    # Carga archivo con trayectorias ideales para cada comando
    # - base: Carpeta donde buscar
    # - Retorna: DataFrame con datos de trayectorias ideales
    
    # Busca archivos con nombres como "puntos_ideales.csv" o "puntos ideales.xlsx"
    for pat in ["puntos_ideales*.csv", "puntos ideales*.csv",
                "puntos_ideales*.xlsx", "puntos ideales*.xlsx"]:
        for f in base.glob(pat):  # Para cada archivo que coincida
            # Lee según extensión
            d = pd.read_csv(f) if f.suffix==".csv" else pd.read_excel(f)
            d.columns = d.columns.str.strip()  # Limpia espacios en nombres de columnas
            d["Command"] = d["Command"].apply(canon_cmd)  # Estandariza comandos
            
            if d["Command"].isna().any():  # Si hay comandos no reconocidos
                raise ValueError(f"Comando no reconocido en {f.name}")
                
            print(f"[+] Ideal usado: {f.name}")
            return d  # Retorna primer archivo válido
            
    raise FileNotFoundError("No se encontró archivo ideal")  # Error si no hay archivo

def read_csv_file(path: Path) -> pd.DataFrame:
    """
    Lee un archivo CSV de trayectoria real, intentando varias opciones.
    
    Args:
        path: Ruta al archivo CSV
    
    Returns:
        DataFrame con los datos del CSV
    """
    try:
        # Intentar leer con pandas primero
        df = pd.read_csv(path, encoding='utf-8')
        
        # Verificar columnas mínimas requeridas
        required_cols = ["ChairPositionX", "ChairPositionZ", "Command"]
        
        # Normalizar nombres de columnas (quitar espacios, convertir a minúsculas)
        df.columns = [c.strip() for c in df.columns]
        
        # Diccionario para mapear variantes de nombres de columna
        col_variants = {
            "ChairPositionX": ["chairpositionx", "chair_position_x", "chairposition_x", "chair_x", "x"],
            "ChairPositionZ": ["chairpositionz", "chair_position_z", "chairposition_z", "chair_z", "z"],
            "Command": ["command", "cmd", "direction", "action"],
            "Time": ["time", "timestamp", "t"],
            "Participant": ["participant", "subject", "id", "pid"],
            "Attempt": ["attempt", "stage", "trial", "etapa"],
            "Task": ["task", "subtask"]
        }
        
        # Verificar y renombrar columnas si es necesario
        for req_col, variants in col_variants.items():
            if req_col not in df.columns:
                # Buscar variantes en minúsculas
                found = False
                for var in variants:
                    for col in df.columns:
                        if var == col.lower():
                            df.rename(columns={col: req_col}, inplace=True)
                            found = True
                            break
                    if found:
                        break
        
        # Verificar si todas las columnas requeridas existen
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"⚠️ Columnas faltantes en {path}: {missing_cols}")
            
            # Intentar extraer información del nombre del archivo para columnas faltantes
            pid, stage = meta(path)
            
            # Crear columnas faltantes con valores predeterminados
            if "Command" not in df.columns:
                df["Command"] = "Front"  # Comando predeterminado
            if "ChairPositionX" not in df.columns:
                df["ChairPositionX"] = 0.0
            if "ChairPositionZ" not in df.columns:
                df["ChairPositionZ"] = 0.0
            if "Time" not in df.columns:
                df["Time"] = range(len(df))
            if "Participant" not in df.columns:
                df["Participant"] = pid
            if "Attempt" not in df.columns:
                df["Attempt"] = stage
            if "Task" not in df.columns:
                df["Task"] = f"Task {stage}"
        
        # Asegurarse de que Command sea un string y normalizarlo
        if "Command" in df.columns:
            df["Command"] = df["Command"].astype(str).apply(canon_cmd)
        
        return df
    
    except Exception as e:
        print(f"❌ Error leyendo {path}: {e}")
        
        # Crear un DataFrame mínimo con la estructura correcta
        pid, stage = meta(path)
        return pd.DataFrame({
            "Time": [0],
            "Participant": [pid],
            "Attempt": [stage],
            "Task": [f"Task {stage}"],
            "Command": ["Front"],
            "ChairPositionX": [0.0],
            "ChairPositionZ": [0.0]
        })

def adapt_trajectory_length(real_traj: pd.DataFrame, ideal_traj: pd.DataFrame) -> pd.DataFrame:
    """
    Adapta la longitud de una trayectoria real para que coincida con la longitud de una trayectoria ideal.
    
    Args:
        real_traj: DataFrame con la trayectoria real (debe tener ChairPositionX, ChairPositionZ)
        ideal_traj: DataFrame con la trayectoria ideal (debe tener IdealPositionX, IdealPositionZ)
        
    Returns:
        DataFrame con la trayectoria real adaptada a la longitud de la ideal
    """
    if real_traj.empty or ideal_traj.empty:
        return real_traj
    
    real_len = len(real_traj)
    ideal_len = len(ideal_traj)
    
    # Si las longitudes son iguales, no es necesario adaptar
    if real_len == ideal_len:
        return real_traj
    
    # Columnas que necesitamos interpolar
    cols_to_interp = ["ChairPositionX", "ChairPositionZ"]
    
    # Verificar si las columnas existen
    for col in cols_to_interp:
        if col not in real_traj.columns:
            print(f"⚠️ Columna '{col}' no encontrada en trayectoria real")
            real_traj[col] = 0.0  # Valor predeterminado
    
    # Si la trayectoria real es más corta, necesitamos añadir puntos
    if real_len < ideal_len:
        # Crear índices normalizados para interpolación
        real_indices = np.linspace(0, 1, real_len)
        ideal_indices = np.linspace(0, 1, ideal_len)
        
        # Crear nuevo DataFrame para la trayectoria adaptada
        adapted_traj = pd.DataFrame()
        
        # Copiar columnas que no necesitan interpolación
        for col in real_traj.columns:
            if col not in cols_to_interp:
                if col == "Time":
                    # Para tiempo, usar interpolación lineal
                    adapted_traj[col] = np.interp(ideal_indices, real_indices, real_traj[col])
                else:
                    # Para otras columnas, repetir el primer valor
                    adapted_traj[col] = [real_traj[col].iloc[0]] * ideal_len
        
        # Interpolar las coordenadas X y Z
        for col in cols_to_interp:
            if col in real_traj.columns:
                adapted_traj[col] = np.interp(ideal_indices, real_indices, real_traj[col])
        
    # Si la trayectoria real es más larga, necesitamos eliminar puntos
    else:
        # Crear índices normalizados para downsampling
        real_indices = np.linspace(0, 1, real_len)
        ideal_indices = np.linspace(0, 1, ideal_len)
        
        # Encontrar los índices más cercanos en la trayectoria real para cada punto deseado
        selected_indices = []
        for idx in ideal_indices:
            # Encontrar el índice real más cercano
            closest_idx = np.argmin(np.abs(real_indices - idx))
            selected_indices.append(closest_idx)
        
        # Seleccionar los puntos correspondientes
        adapted_traj = real_traj.iloc[selected_indices].reset_index(drop=True)
    
    # Añadir las posiciones ideales al DataFrame adaptado
    adapted_traj["IdealPositionX"] = ideal_traj["IdealPositionX"].values
    adapted_traj["IdealPositionZ"] = ideal_traj["IdealPositionZ"].values
    
    return adapted_traj

def process_file(file_path: Path, ideal_df: pd.DataFrame, output_base: Path) -> None:
    """
    Procesa un archivo CSV, adapta sus trayectorias y guarda el resultado.
    
    Args:
        file_path: Ruta al archivo CSV original
        ideal_df: DataFrame con trayectorias ideales
        output_base: Carpeta base para guardar resultados
    """
    try:
        # Extraer información del archivo
        pid, stage = meta(file_path)
        
        # Leer el archivo CSV
        real_df = read_csv_file(file_path)
        
        if real_df.empty:
            print(f"⚠️ Archivo vacío o no se pudo leer: {file_path}")
            return
        
        # Crear carpetas de salida
        output_dir = output_base / f"participant_{pid}" / f"etapa_{stage}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / f"participant_{pid}_etapa_{stage}_adaptado.csv"
        
        # Agrupar por comandos para procesar cada uno por separado
        results = []
        
        for cmd, group in real_df.groupby("Command"):
            # Filtrar trayectoria ideal para este comando
            cmd_ideal = ideal_df[ideal_df["Command"] == cmd]
            
            if cmd_ideal.empty:
                print(f"⚠️ No hay trayectoria ideal para comando: {cmd}")
                continue
            
            # Adaptar longitud de la trayectoria real
            adapted = adapt_trajectory_length(group, cmd_ideal)
            
            if not adapted.empty:
                results.append(adapted)
        
        if not results:
            print(f"⚠️ No se pudieron adaptar trayectorias para: {file_path}")
            return
            
        # Combinar todas las trayectorias adaptadas
        result_df = pd.concat(results).sort_values("Time").reset_index(drop=True)
        
        # Guardar resultado
        result_df.to_csv(output_file, index=False)
        print(f"✓ Adaptado y guardado: {output_file}")
        
    except Exception as e:
        print(f"❌ Error procesando {file_path}: {e}")
        import traceback
        traceback.print_exc()

def main(base: Path, output_base: Path):
    """
    Función principal que coordina todo el proceso.
    
    Args:
        base: Carpeta base donde buscar archivos
        output_base: Carpeta donde guardar resultados
    """
    try:
        # Cargar trayectorias ideales
        ideal_df = load_ideal(base)
        
        # Buscar archivos CSV en DTW_dataset
        dataset_dir = base / "DTW_dataset"
        
        if not dataset_dir.exists():
            print(f"❌ No se encontró la carpeta DTW_dataset en {base}")
            return
            
        csv_files = list(dataset_dir.rglob("*.csv"))
        
        if not csv_files:
            print(f"❌ No se encontraron archivos CSV en {dataset_dir}")
            return
            
        print(f"🔄 Procesando {len(csv_files)} archivos...")
        
        # Crear carpeta de salida
        output_base.mkdir(parents=True, exist_ok=True)
        
        # Procesar cada archivo
        for file_path in csv_files:
            process_file(file_path, ideal_df, output_base)
            
        print(f"✅ Proceso completado. Resultados en: {output_base}")
        
    except Exception as e:
        print(f"❌ Error en el pipeline principal: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Punto de entrada cuando se ejecuta el script directamente
    ag = argparse.ArgumentParser()
    ag.add_argument("--base", default=".", help="Carpeta raíz (DTW_dataset)")
    ag.add_argument("--output", default="./DTW_dataset_adaptada", help="Carpeta de salida para resultados adaptados")
    A = ag.parse_args()

    t0 = time.perf_counter()  # Inicia cronómetro
    main(Path(A.base).resolve(), Path(A.output).resolve())  # Ejecuta función principal
    print(f"⏱️  Tiempo total: {time.perf_counter()-t0:.2f} s")  # Muestra tiempo total