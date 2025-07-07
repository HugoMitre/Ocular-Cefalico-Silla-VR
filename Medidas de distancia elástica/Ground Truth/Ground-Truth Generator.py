import pandas as pd
import numpy as np
from pathlib import Path
from dtaidistance import dtw
import matplotlib.pyplot as plt

# ──────── Rutas ────────
input_path = Path(r"C:\Users\Manuel Delado\Documents\clasificador tesis\mio\error_medio\puntos_ideales_ajustados_1.csv")
output_dir = input_path.parent / "ground_truth_salida"
output_dir.mkdir(parents=True, exist_ok=True)
csv_output = output_dir / "ground_truth_todo.csv"

# ──────── Leer CSV ────────
df = pd.read_csv(input_path)

# Verificar columnas
required_cols = ["TimeSec", "Command", "IdealPositionX", "IdealPositionZ"]
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Falta columna: {col}")

# ──────── Procesar por comando y combinar ────────
comandos = df["Command"].dropna().unique()
todos = []

for cmd in comandos:
    df_cmd = df[df["Command"] == cmd].reset_index(drop=True)
    if len(df_cmd) < 2:
        print(f"[!] Comando '{cmd}' ignorado por tener menos de 2 puntos.")
        continue

    # Extraer X y Z
    x = df_cmd["IdealPositionX"].to_numpy()
    z = df_cmd["IdealPositionZ"].to_numpy()

    # Duplicado
    df_dup = df_cmd.copy()
    df_dup.columns = [col + "_dup" for col in df_dup.columns]

    # DTW X
    _, cost_x = dtw.warping_paths(x, x)
    path_x = dtw.best_path(cost_x)

    # DTW Z
    _, cost_z = dtw.warping_paths(z, z)
    path_z = dtw.best_path(cost_z)

    # Path DataFrame
    path_df = pd.DataFrame({
        "path_i_x": [i for i, _ in path_x],
        "path_j_x": [j for _, j in path_x],
        "path_i_z": [i for i, _ in path_z],
        "path_j_z": [j for _, j in path_z]
    })

    # Combinar todo en un solo dataframe (sin matriz de costos)
    combined = pd.concat([
        df_cmd.reset_index(drop=True),
        df_dup.reset_index(drop=True),
        path_df.reindex(range(len(df_cmd))).reset_index(drop=True)
    ], axis=1)

    todos.append(combined)

    # Guardar imágenes de la matriz de costos
    cmd_safe = cmd.replace(" ", "_").replace("-", "_")

    # Imagen X
    plt.figure(figsize=(6, 4))
    plt.imshow(cost_x, cmap=plt.cm.binary, interpolation="nearest", origin="lower")
    x_path, y_path = zip(*path_x)
    plt.plot(y_path, x_path)
    plt.xlabel("$y$")
    plt.ylabel("$x$")
    plt.title(f"Cost Matrix X - {cmd}")
    plt.tight_layout()
    plt.savefig(output_dir / f"matriz_costos_{cmd_safe}_X.png")
    plt.close()

    # Imagen Z
    plt.figure(figsize=(6, 4))
    plt.imshow(cost_z, cmap=plt.cm.binary, interpolation="nearest", origin="lower")
    x_path_z, y_path_z = zip(*path_z)
    plt.plot(y_path_z, x_path_z)
    plt.xlabel("$y$")
    plt.ylabel("$x$")
    plt.title(f"Cost Matrix Z - {cmd}")
    plt.tight_layout()
    plt.savefig(output_dir / f"matriz_costos_{cmd_safe}_Z.png")
    plt.close()

# ──────── Guardar CSV final ────────
df_final = pd.concat(todos, axis=0).reset_index(drop=True)
df_final.to_csv(csv_output, index=False, encoding="utf-8")

print("✔ Todo generado exitosamente:")
print(f"  - {csv_output.name} (todos los comandos)")
print(f"  - imágenes .png por cada comando en: {output_dir}")
