import os
import shutil
from pathlib import Path

def analizar_estructura(ruta_carpeta, nivel=0, max_nivel=10):
    """
    Analiza y muestra la estructura de carpetas de forma recursiva.
    
    Args:
        ruta_carpeta (str): Ruta de la carpeta a analizar
        nivel (int): Nivel actual de profundidad (para indentación)
        max_nivel (int): Máximo nivel de profundidad a mostrar
    
    Returns:
        dict: Diccionario con estadísticas de la estructura
    """
    carpeta = Path(ruta_carpeta)
    estadisticas = {
        'total_carpetas': 0,
        'total_archivos': 0,
        'estructura': {},
        'extensiones': {}
    }
    
    if not carpeta.exists() or not carpeta.is_dir():
        return estadisticas
    
    # Crear indentación visual
    indentacion = "│   " * nivel
    prefijo = "├── " if nivel > 0 else "📁 "
    
    try:
        # Obtener contenido de la carpeta
        contenido = list(carpeta.iterdir())
        carpetas = [item for item in contenido if item.is_dir()]
        archivos = [item for item in contenido if item.is_file()]
        
        # Mostrar nombre de la carpeta actual
        print(f"{indentacion}{prefijo}{carpeta.name}/")
        
        # Contar archivos en esta carpeta
        estadisticas['total_archivos'] += len(archivos)
        
        # Mostrar archivos si hay pocos (para no saturar la pantalla)
        if len(archivos) <= 10 and nivel < 3:
            for archivo in sorted(archivos):
                extension = archivo.suffix.lower()
                estadisticas['extensiones'][extension] = estadisticas['extensiones'].get(extension, 0) + 1
                print(f"{indentacion}│   📄 {archivo.name}")
        elif len(archivos) > 0:
            # Contar extensiones
            for archivo in archivos:
                extension = archivo.suffix.lower()
                estadisticas['extensiones'][extension] = estadisticas['extensiones'].get(extension, 0) + 1
            print(f"{indentacion}│   📄 [{len(archivos)} archivos]")
        
        # Procesar subcarpetas recursivamente
        for carpeta_hija in sorted(carpetas):
            estadisticas['total_carpetas'] += 1
            if nivel < max_nivel:
                sub_stats = analizar_estructura(str(carpeta_hija), nivel + 1, max_nivel)
                # Combinar estadísticas
                estadisticas['total_carpetas'] += sub_stats['total_carpetas']
                estadisticas['total_archivos'] += sub_stats['total_archivos']
                for ext, count in sub_stats['extensiones'].items():
                    estadisticas['extensiones'][ext] = estadisticas['extensiones'].get(ext, 0) + count
            else:
                print(f"{indentacion}│   ├── {carpeta_hija.name}/ [...]")
    
    except PermissionError:
        print(f"{indentacion}│   ❌ Sin permisos para acceder")
    except Exception as e:
        print(f"{indentacion}│   ❌ Error: {e}")
    
    return estadisticas

def mostrar_resumen_estructura(estadisticas):
    """
    Muestra un resumen de las estadísticas de la estructura.
    """
    print("\n" + "="*60)
    print("📊 RESUMEN DE LA ESTRUCTURA")
    print("="*60)
    print(f"📁 Total de carpetas: {estadisticas['total_carpetas']}")
    print(f"📄 Total de archivos: {estadisticas['total_archivos']}")
    
    if estadisticas['extensiones']:
        print(f"\n📋 Tipos de archivos encontrados:")
        for extension, cantidad in sorted(estadisticas['extensiones'].items()):
            ext_display = extension if extension else "(sin extensión)"
            print(f"   {ext_display}: {cantidad} archivo{'s' if cantidad != 1 else ''}")
    
    print("="*60)

def duplicar_carpeta_especifica(carpeta_origen, directorio_destino, nombre_carpeta_duplicada):
    """
    Duplica una carpeta específica en un directorio destino.
    
    Args:
        carpeta_origen (str): Ruta completa de la carpeta que se quiere duplicar
        directorio_destino (str): Directorio donde se guardará la carpeta duplicada
        nombre_carpeta_duplicada (str): Nombre de la nueva carpeta duplicada
    """
    
    # Convertir a Path para manejo más fácil de rutas
    origen = Path(carpeta_origen)
    dir_destino = Path(directorio_destino)
    destino = dir_destino / nombre_carpeta_duplicada
    
    # Verificar que la carpeta origen existe
    if not origen.exists():
        print(f"❌ Error: La carpeta '{carpeta_origen}' no existe.")
        return False
    
    if not origen.is_dir():
        print(f"❌ Error: '{carpeta_origen}' no es una carpeta.")
        return False
    
    # Verificar que el directorio destino existe
    if not dir_destino.exists():
        print(f"❌ Error: El directorio destino '{directorio_destino}' no existe.")
        return False
    
    try:
        # Si la carpeta destino ya existe, preguntamos si la eliminamos
        if destino.exists():
            respuesta = input(f"⚠️  La carpeta '{nombre_carpeta_duplicada}' ya existe en el destino. ¿Deseas reemplazarla? (s/n): ")
            if respuesta.lower() in ['s', 'sí', 'si', 'yes', 'y']:
                shutil.rmtree(destino)
                print(f"🗑️  Carpeta existente eliminada.")
            else:
                print("❌ Operación cancelada.")
                return False
        
        # Copiar toda la estructura de carpetas y archivos
        shutil.copytree(origen, destino)
        
        print(f"✅ ¡Duplicación completada exitosamente!")
        print(f"📁 Carpeta origen: {origen}")
        print(f"📁 Carpeta duplicada: {destino}")
        
        # Mostrar estadísticas
        archivos_copiados = sum([len(files) for r, d, files in os.walk(destino)])
        carpetas_copiadas = sum([len(dirs) for r, dirs, files in os.walk(destino)])
        
        print(f"📊 Estadísticas:")
        print(f"   - Archivos copiados: {archivos_copiados}")
        print(f"   - Carpetas copiadas: {carpetas_copiadas}")
        
        return True
        
    except PermissionError as e:
        print(f"❌ Error de permisos: {e}")
        print("💡 Intenta ejecutar el programa como administrador.")
        return False
    except Exception as e:
        print(f"❌ Error inesperado: {e}")
        return False

def main():
    """Función principal del programa"""
    print("🔄 ANALIZADOR Y DUPLICADOR DE CARPETA DTW_dataset")
    print("=" * 60)
    
    # Rutas específicas para tu caso
    carpeta_origen = r"C:\Users\Manuel Delado\Documents\clasificador tesis\mio\DTW_dataset"
    directorio_destino = r"C:\Users\Manuel Delado\Documents\clasificador tesis\mio"
    
    print(f"📁 Analizando carpeta: {carpeta_origen}")
    print(f"📁 Se guardará en: {directorio_destino}")
    print("\n🔍 ANÁLISIS DE LA ESTRUCTURA:")
    print("-" * 60)
    
    # Analizar y mostrar la estructura
    estadisticas = analizar_estructura(carpeta_origen)
    
    # Mostrar resumen
    mostrar_resumen_estructura(estadisticas)
    
    # Preguntar si continuar con la duplicación
    print(f"\n¿Deseas continuar con la duplicación? (s/n): ", end="")
    continuar = input().strip().lower()
    
    if continuar not in ['s', 'sí', 'si', 'yes', 'y']:
        print("❌ Operación cancelada.")
        input("\nPresiona Enter para salir...")
        return
    
    # Preguntar nombre de carpeta destino
    nombre_destino = input("Nombre para la carpeta duplicada (por defecto 'DTW_dataset_duplicado'): ").strip()
    if not nombre_destino:
        nombre_destino = "DTW_dataset_duplicado"
    
    print(f"\n🚀 Iniciando duplicación...")
    print(f"📂 Desde: {carpeta_origen}")
    print(f"📂 Hacia: {directorio_destino}\\{nombre_destino}")
    print("-" * 60)
    
    # Ejecutar la duplicación
    exito = duplicar_carpeta_especifica(carpeta_origen, directorio_destino, nombre_destino)
    
    if exito:
        print("\n🎉 ¡Proceso completado con éxito!")
    else:
        print("\n💥 El proceso terminó con errores.")
    
    input("\nPresiona Enter para salir...")

if __name__ == "__main__":
    main()
