import random
import csv
from pathlib import Path
import numpy as np

# ---------------------------------------------------------------------------------
#  CONFIGURACIÓN GENERAL
# ---------------------------------------------------------------------------------

ROOT = Path("DTW_dataset_realistic")
PARTICIPANTS = 30
STAGES = 3

# Secuencia fija de Task / Command
task_command = [
    ("task one",   "right turn"),
    ("two",        "Front"),
    ("Task three", "left turn"),
    ("Task four",  "Front-Right Diagonal"),
    ("Task five",  "Front-Left Diagonal"),
    ("Task Six",   "Back-Left Diagonal"),
    ("Task Seven", "Back-Right Diagonal"),
    ("Task eight", "Back")
]

# ---------------------------------------------------------------------------------
#  PARÁMETROS CINEMÁTICOS (µ, σ) POR DIRECCIÓN
# ---------------------------------------------------------------------------------
_dir = {
    "Front"      : ( 65,10,   50, 8,   18,4,     0, 5,    0, 4,    70,12,   55,10,   65,15),
    "Left"       : (-70, 8,   20, 6,   20,4,   -75,10,    0, 8,   -75,10,   25, 7,   70,12),
    "Right"      : ( 70, 8,   20, 6,   20,4,    75,10,    0, 8,    75,10,   25, 7,   70,12),
    "Up"         : ( 25, 6,   65, 8,   18,5,     0, 6,   50,15,    28, 7,   70,10,   65,14),
    "Down"       : ( 25, 6,  -65, 8,   18,5,     0, 6,  -50,15,    28, 7,  -70,10,   65,14),
    "Up-Left"    : ( 55,10,   60, 8,   20,4,    55,12,   55, 9,    60,10,   65, 9,   70,12),
    "Up-Right"   : ( 55,10,   60, 8,   20,4,    55,12,   55, 9,    60,10,   65, 9,   70,12),
    "Down-Left"  : ( 55,10,  -60, 8,   20,4,    55,12,  -55, 9,    60,10,  -65, 9,   70,12),
    "Down-Right" : ( 55,10,  -60, 8,   20,4,    55,12,  -55, 9,    60,10,  -65, 9,   70,12)
}

# Mapeo de diagonales
for diag in ["Front-Right Diagonal", "Back-Right Diagonal"]:
    _dir[diag] = _dir["Right"]
for diag in ["Front-Left Diagonal", "Back-Left Diagonal"]:
    _dir[diag] = _dir["Left"]

# ---------------------------------------------------------------------------------
#  FACTORES POR ETAPA
# ---------------------------------------------------------------------------------
stage_k = {1: 1.0, 2: 0.6, 3: 0.35}

rnd = lambda µ, σ, st: round(random.gauss(µ, σ*stage_k[st]), 6)

# ---------------------------------------------------------------------------------
#  PUNTOS DE RUTA IDEALES
# ---------------------------------------------------------------------------------
route_points = {
    "right turn": {
        "start": (-22.65709, 6.361619),
        "end": (-22.6605, 4.881082)
    },
    "Front": {
        "start": (-22.6638, 4.703775),
        "end": (-22.55156, -2.09514)
    },
    "left turn": {
        "start": (-22.52902, -2.27449),
        "end": (-20.14235, -5.037046)
    },
    "Front-Right Diagonal": {
        "start": (-14.71175, -4.627835),
        "end": (-10.29645, -6.647438)
    },
    "Front-Left Diagonal": {
        "start": (-10.2066, -6.645674),
        "end": (-6.496493, -4.940638)
    },
    "Back-Left Diagonal": {
        "start": (-4.63133, -1.897745),
        "end": (-9.761788, 0.868068)
    },
    "Back-Right Diagonal": {
        "start": (-9.853524, 0.857745),
        "end": (-15.14106, -1.194558)
    },
    "Back": {
        "start": (-15.23071, -1.217206),
        "end": (-20.27323, -1.715508)
    }
}

# ---------------------------------------------------------------------------------
#  FUNCIONES DE DESVIACIÓN MEJORADAS
# ---------------------------------------------------------------------------------

def calculate_movement_scale(cmd):
    """
    Calcula la escala de movimiento para un comando específico
    """
    if cmd not in route_points:
        return 1.0
    
    start = route_points[cmd]["start"]
    end = route_points[cmd]["end"]
    
    # Calcular la distancia total del movimiento
    distance = np.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
    
    # Calcular el rango de movimiento en cada eje
    range_x = abs(end[0] - start[0])
    range_z = abs(end[1] - start[1])
    
    return {
        "distance": distance,
        "range_x": range_x,
        "range_z": range_z,
        "avg_range": (range_x + range_z) / 2
    }

def adaptive_path_deviation(stg, cmd, task_progress):
    """
    Genera desviaciones adaptativas basadas en:
    - La etapa (stage)
    - El comando actual
    - El progreso en la tarea
    """
    # Obtener la escala de movimiento para este comando
    scale = calculate_movement_scale(cmd)
    
    # Base deviation como porcentaje del movimiento total
    # Usar 5% del rango promedio como desviación base
    base_deviation_percentage = 0.05  # 5% del movimiento
    
    if cmd in route_points:
        # Desviación proporcional al tamaño del movimiento
        base_deviation = scale["avg_range"] * base_deviation_percentage
    else:
        # Valor por defecto muy pequeño
        base_deviation = 0.01
    
    # Factor de etapa
    stage_factor = stage_k[stg]
    
    # Factor de progreso (mayor error al inicio y fin)
    progress_factor = 1.0
    if task_progress is not None:
        # Patrón en U: más error al principio (aprendizaje) y al final (fatiga)
        progress_factor = 1.0 + 0.3 * (1.0 - 4.0 * (task_progress - 0.5)**2)
    
    # Generar desviación con distribución normal
    deviation = random.gauss(0, base_deviation * stage_factor * progress_factor)
    
    return deviation

def generate_smooth_noise(n_points, scale=0.001):
    """
    Genera ruido suave y correlacionado para simular temblor realista
    """
    # Generar ruido blanco
    white_noise = np.random.randn(n_points) * scale
    
    # Aplicar filtro de media móvil para suavizar
    window_size = min(5, n_points // 3)
    if window_size > 1:
        smoothed = np.convolve(white_noise, np.ones(window_size)/window_size, mode='same')
    else:
        smoothed = white_noise
    
    return smoothed

# ---------------------------------------------------------------------------------
#  FILA INDIVIDUAL MEJORADA
# ---------------------------------------------------------------------------------

def make_row(t, pid, stg, task, cmd, task_start_time, task_duration, 
             noise_x=0, noise_z=0, prev_chair_x=None, prev_chair_z=None):
    
    def resolve(cmd):
        if not cmd: return "Front"
        c = cmd.lower()
        if "front-right" in c: return "Up-Right"
        if "front-left"  in c: return "Up-Left"
        if "back-left"   in c: return "Down-Left"
        if "back-right"  in c: return "Down-Right"
        if "front"       in c: return "Front"
        if "back"        in c: return "Down"
        if "left"        in c: return "Left"
        if "right"       in c: return "Right"
        return "Front"

    direc = resolve(cmd)
    stats = _dir[direc]
    (hvxµ, hvxσ, hvyµ, hvyσ, σhµ, σhσ, axµ, axσ, ayµ, ayσ,
     gvxµ, gvxσ, gvyµ, gvyσ, σgµ, σgσ) = stats

    # Calcular posición ideal
    if cmd and cmd in route_points:
        start_x, start_z = route_points[cmd]["start"]
        end_x, end_z = route_points[cmd]["end"]
        
        task_progress = min(1.0, (t - task_start_time) / task_duration)
        
        # Interpolación con función de easing (más realista que lineal)
        # Usar función sigmoide para acelerar al inicio y desacelerar al final
        eased_progress = task_progress  # Por simplicidad, mantener lineal
        
        ideal_x = start_x + (end_x - start_x) * eased_progress
        ideal_z = start_z + (end_z - start_z) * eased_progress
    else:
        ideal_x = -15.0
        ideal_z = 0.0

    # Generar desviaciones adaptativas
    task_progress = (t - task_start_time) / task_duration if task_duration > 0 else 0
    dx = adaptive_path_deviation(stg, cmd, task_progress)
    dz = adaptive_path_deviation(stg, cmd, task_progress)
    
    # Añadir ruido suave de alta frecuencia
    dx += noise_x
    dz += noise_z
    
    # Calcular posición real
    chair_x = ideal_x + dx
    chair_z = ideal_z + dz
    
    # Aplicar suavizado con la posición anterior (filtro de Kalman simple)
    if prev_chair_x is not None and prev_chair_z is not None:
        alpha = 0.8  # Factor de suavizado (0.8 = 80% valor nuevo, 20% valor anterior)
        chair_x = alpha * chair_x + (1 - alpha) * prev_chair_x
        chair_z = alpha * chair_z + (1 - alpha) * prev_chair_z
    
    chair_x = round(chair_x, 6)
    chair_z = round(chair_z, 6)
    
    # Desviación de ruta
    actual_dx = chair_x - ideal_x
    actual_dz = chair_z - ideal_z
    pdev = round(np.sqrt(actual_dx**2 + actual_dz**2), 6)

    # Tiempos de permanencia
    dwell_s = max(0, t-2)
    dwell_e = t if t%5<4 else None

    return {
        "Time": round(t, 3),
        "Participant": pid,
        "Attempt": stg,
        "Task": task,
        "Command": cmd,
        # Cabeza
        "HeadVelocityX": rnd(hvxµ, hvxσ, stg),
        "HeadVelocityY": rnd(hvyµ, hvyσ, stg),
        "HeadVelocityXNorm": rnd(hvxµ/90, hvxσ/90, stg),
        "HeadVelocityYNorm": rnd(hvyµ/90, hvyσ/90, stg),
        "HeadDeviationX": rnd(axµ*0.12, 2, stg),
        "HeadDeviationY": rnd(ayµ*0.12, 2, stg),
        "HeadDeviationXNorm": rnd(axµ/90*0.12, 0.02, stg),
        "HeadDeviationYNorm": rnd(ayµ/90*0.12, 0.02, stg),
        "HeadAngleX": rnd(axµ, axσ, stg),
        "HeadAngleY": rnd(ayµ, ayσ, stg),
        "HeadAngleXNormalized": rnd(axµ/90, axσ/90, stg),
        "HeadAngleYNormalized": rnd(ayµ/90, ayσ/90, stg),
        "HeadDirection": direc,
        "HeadDwellStartTimeSec": dwell_s,
        "HeadDwellEndTimeSec": dwell_e,
        "HeadDwellDurationSec": None if dwell_e is None else round(dwell_e-dwell_s, 3),
        # Mirada
        "GazeVelocityX": rnd(gvxµ, gvxσ, stg),
        "GazeVelocityY": rnd(gvyµ, gvyσ, stg),
        "GazeVelocityXNorm": rnd(gvxµ/90, gvxσ/90, stg),
        "GazeVelocityYNorm": rnd(gvyµ/90, gvyσ/90, stg),
        "GazeDeviationX": rnd(axµ*0.18, 3, stg),
        "GazeDeviationY": rnd(ayµ*0.18, 3, stg),
        "GazeDeviationXNorm": rnd(axµ/90*0.18, 0.03, stg),
        "GazeDeviationYNorm": rnd(ayµ/90*0.18, 0.03, stg),
        "GazeAngleX": rnd(axµ*1.05, axσ*1.1, stg),
        "GazeAngleY": rnd(ayµ*1.05, ayσ*1.1, stg),
        "GazeAngleXNormalized": rnd(axµ/90*1.05, axσ/90*1.1, stg),
        "GazeAngleYNormalized": rnd(ayµ/90*1.05, ayσ/90*1.1, stg),
        "GazeDirection": direc,
        # Ruta
        "ChairPositionX": chair_x,
        "ChairPositionZ": chair_z,
        "IdealPositionX": round(ideal_x, 6),
        "IdealPositionZ": round(ideal_z, 6),
        "PathDeviation": pdev,
        # Joystick
        "JoistickInput_X_Normalized": rnd(actual_dx*10, 0.02, stg),
        "JoistikInput_Y_Normalized": rnd(actual_dz*10, 0.02, stg),
        "JoistickAngle": rnd(random.uniform(-180, 180), 3, stg),
        "Magnitude": round(min(1.0, pdev * 5), 3),
        # Duplicados
        "ChairPosition_X": chair_x,
        "ChairPosition_Z": chair_z,
        "ChairRotation_Y": rnd(axµ, axσ, stg),
        "Action": (cmd.split()[0] if cmd else None),
        "ChairDwellDuration": None if dwell_e is None else round(dwell_e-dwell_s, 3)
    }

# ---------------------------------------------------------------------------------
#  GENERADOR PRINCIPAL MEJORADO
# ---------------------------------------------------------------------------------

def generate_dataset():
    ROOT.mkdir(exist_ok=True)
    header = list(make_row(0, 1, 1, "task", "cmd", 0, 1).keys())

    for pid in range(1, PARTICIPANTS+1):
        for stg in range(1, STAGES+1):
            total_time = random.uniform(60, 80)
            dt = random.uniform(0.08, 0.12)
            
            folder = ROOT / f"participant_{pid:02d}" / f"etapa_{stg}"
            folder.mkdir(parents=True, exist_ok=True)
            file_path = folder / f"participant_{pid:02d}_etapa_{stg}.csv"

            # Distribuir tiempo entre tareas
            task_durations = []
            remaining_time = total_time
            
            for i in range(len(task_command) - 1):
                min_time_per_remaining = 4.0
                max_allocation = remaining_time - (min_time_per_remaining * (len(task_command) - i - 1))
                
                if max_allocation <= min_time_per_remaining:
                    task_dur = min_time_per_remaining
                else:
                    task_dur = random.uniform(min_time_per_remaining, max_allocation)
                
                task_durations.append(task_dur)
                remaining_time -= task_dur
            
            task_durations.append(remaining_time)
            
            # Calcular tiempos de inicio
            task_start_times = [0]
            for i in range(len(task_durations) - 1):
                task_start_times.append(task_start_times[-1] + task_durations[i])

            # Generar ruido suave para toda la secuencia
            n_timesteps = int(total_time/dt)
            noise_scale = 0.001  # Escala muy pequeña para el ruido
            noise_x_seq = generate_smooth_noise(n_timesteps, noise_scale)
            noise_z_seq = generate_smooth_noise(n_timesteps, noise_scale)
            
            # Escribir CSV
            with file_path.open("w", newline="", encoding="utf8") as f:
                w = csv.DictWriter(f, fieldnames=header)
                w.writeheader()

                current_task_idx = 0
                prev_chair_x = None
                prev_chair_z = None
                
                for i, t in enumerate(i*dt for i in range(n_timesteps)):
                    # Determinar tarea actual
                    while (current_task_idx < len(task_command) - 1 and 
                           t >= task_start_times[current_task_idx+1]):
                        current_task_idx += 1
                    
                    task, cmd = task_command[current_task_idx]
                    task_start_time = task_start_times[current_task_idx]
                    task_duration = task_durations[current_task_idx]
                    
                    # Obtener ruido para este timestep
                    noise_x = noise_x_seq[i] if i < len(noise_x_seq) else 0
                    noise_z = noise_z_seq[i] if i < len(noise_z_seq) else 0
                    
                    row = make_row(t, pid, stg, task, cmd, task_start_time, task_duration,
                                   noise_x, noise_z, prev_chair_x, prev_chair_z)
                    w.writerow(row)
                    
                    # Guardar posición para suavizado
                    prev_chair_x = row["ChairPositionX"]
                    prev_chair_z = row["ChairPositionZ"]
            
            print(f"✓ Creado: {file_path} (Participante {pid}, Etapa {stg})")
            
            # Mostrar estadísticas para el primer participante
            if pid == 1 and stg == 1:
                print("\n📊 Estadísticas de ejemplo (Participante 1, Etapa 1):")
                for cmd_name, points in route_points.items():
                    scale = calculate_movement_scale(cmd_name)
                    base_dev = scale["avg_range"] * 0.05  # 5%
                    print(f"  {cmd_name:20s}: Movimiento={scale['distance']:.3f}, "
                          f"Desviación base={base_dev:.4f}")

if __name__ == "__main__":
    print("🔧 Generando dataset con desviaciones realistas...")
    generate_dataset()
    print("\n✅ Dataset generado exitosamente!")