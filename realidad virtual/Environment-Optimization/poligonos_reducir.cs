using UnityEngine;
using UnityEditor;
using UnityMeshSimplifier;
using System.Collections.Generic;
using System.IO;
using System.Linq;

#if UNITY_EDITOR
[ExecuteInEditMode]
public class SimplificadorMasivo : MonoBehaviour
{
    [Header("Configuración Principal")]
    [Tooltip("Carpetas que contienen todos los modelos a simplificar")]
    [SerializeField] private List<Transform> carpetasPrincipales;

    [Tooltip("Calidad de la malla (1 = original, 0 = máxima simplificación)")]
    [Range(0, 1)]
    [SerializeField] private float calidadMalla = 0.5f; // Aumentado a 0.5 para mejor calidad

    [Header("Configuración de Preservación")]
    [Tooltip("Mantener bordes de UV")]
    [SerializeField] private bool preservarUVs = true;
    [Tooltip("Mantener vértices en los bordes")]
    [SerializeField] private bool preservarBordes = true;

    [Header("Límites y Restricciones")]
    [Tooltip("Número mínimo de triángulos a mantener")]
    [SerializeField] private int triangulosMinimos = 100; // Aumentado para mantener más detalle
    [Tooltip("Ignorar objetos con menos triángulos que este valor")]
    [SerializeField] private int umbralTriangulosMinimos = 300;

    [Header("Guardado y Backup")]
    [SerializeField] private bool guardarMallas = true;
    [SerializeField] private bool crearBackup = true;
    [SerializeField] private string carpetaDestino = "Assets/MallasSimplificadas";
    [SerializeField] private string carpetaBackup = "Assets/MallasBackup";

    [Header("Debug")]
    [SerializeField] private bool habilitarDebugLog = true;
    [SerializeField] private bool mostrarEstadisticas = true;

    // Estadísticas
    private int totalObjetosProcesados;
    private int objetosSimplificados;
    private int erroresEncontrados;
    private float reduccionPromedio;
    private Dictionary<string, MeshStats> estadisticasPorObjeto;

    private class MeshStats
    {
        public int verticesOriginales;
        public int trianglesOriginales;
        public int verticesFinales;
        public int trianglesFinales;
        public float porcentajeReduccion;
    }

    // Lista de nombres a excluir (case-insensitive)
    private readonly string[] nombresExcluidos = new string[]
    {
        "pilar", "columna", "pillar", "column", "beam", "wall"
    };

    public void SimplificarTodos()
    {
        if (!ValidarConfiguracion()) return;

        InicializarEstadisticas();

        foreach (Transform carpeta in carpetasPrincipales)
        {
            MeshFilter[] todasLasMallas = carpeta.GetComponentsInChildren<MeshFilter>(true);

            if (habilitarDebugLog)
                Debug.Log($"Procesando {carpeta.name} con {todasLasMallas.Length} objetos");

            if (crearBackup)
                CrearBackupMallas(todasLasMallas);

            foreach (MeshFilter malla in todasLasMallas)
            {
                if (!DebeExcluirse(malla.gameObject.name))
                {
                    ProcesarMalla(malla);
                }
                else if (habilitarDebugLog)
                {
                    Debug.Log($"Excluyendo objeto: {malla.gameObject.name}");
                }
            }
        }

        MostrarResumenFinal();
        AssetDatabase.SaveAssets();
        AssetDatabase.Refresh();
    }

    private bool DebeExcluirse(string nombreObjeto)
    {
        nombreObjeto = nombreObjeto.ToLower();
        return nombresExcluidos.Any(excluido => nombreObjeto.Contains(excluido.ToLower()));
    }

    private bool ValidarConfiguracion()
    {
        if (carpetasPrincipales == null || carpetasPrincipales.Count == 0)
        {
            Debug.LogError("Error: No se han asignado carpetas principales");
            return false;
        }

        if (guardarMallas && !Directory.Exists(carpetaDestino))
        {
            try
            {
                Directory.CreateDirectory(carpetaDestino);
            }
            catch (System.Exception e)
            {
                Debug.LogError($"Error al crear directorio de destino: {e.Message}");
                return false;
            }
        }

        if (crearBackup && !Directory.Exists(carpetaBackup))
        {
            try
            {
                Directory.CreateDirectory(carpetaBackup);
            }
            catch (System.Exception e)
            {
                Debug.LogError($"Error al crear directorio de backup: {e.Message}");
                return false;
            }
        }

        return true;
    }

    private void InicializarEstadisticas()
    {
        totalObjetosProcesados = 0;
        objetosSimplificados = 0;
        erroresEncontrados = 0;
        reduccionPromedio = 0;
        estadisticasPorObjeto = new Dictionary<string, MeshStats>();
    }

    private void CrearBackupMallas(MeshFilter[] mallas)
    {
        foreach (var malla in mallas)
        {
            if (malla.sharedMesh == null) continue;

            string rutaBackup = Path.Combine(carpetaBackup, $"{malla.gameObject.name}_backup.asset");

            try
            {
                Mesh meshBackup = Instantiate(malla.sharedMesh);
                AssetDatabase.CreateAsset(meshBackup, rutaBackup);
            }
            catch (System.Exception e)
            {
                Debug.LogWarning($"No se pudo crear backup para {malla.gameObject.name}: {e.Message}");
            }
        }
        AssetDatabase.SaveAssets();
    }

    private void ProcesarMalla(MeshFilter malla)
    {
        if (malla == null || malla.sharedMesh == null) return;

        totalObjetosProcesados++;
        string nombreObjeto = malla.gameObject.name;

        try
        {
            Mesh mallaOriginal = malla.sharedMesh;

            if (mallaOriginal.triangles.Length / 3 < umbralTriangulosMinimos)
            {
                if (habilitarDebugLog)
                    Debug.Log($"Omitiendo {nombreObjeto} - muy pocos triángulos ({mallaOriginal.triangles.Length / 3})");
                return;
            }

            var simplificador = new MeshSimplifier();
            simplificador.Initialize(mallaOriginal);

            // Configuración mejorada para preservar calidad
            var opciones = simplificador.SimplificationOptions;
            opciones.PreserveUVSeamEdges = preservarUVs;
            opciones.PreserveBorderEdges = preservarBordes;
            opciones.PreserveUVFoldoverEdges = true;
            opciones.PreserveSurfaceCurvature = true;
            opciones.EnableSmartLink = true;
            opciones.VertexLinkDistance = 0.0001f;
            opciones.MaxIterationCount = 100; // Más iteraciones para mejor calidad

            int trianglesObjetivo = Mathf.Max(
                triangulosMinimos,
                Mathf.RoundToInt(mallaOriginal.triangles.Length / 3 * calidadMalla)
            );

            float calidadCalculada = (float)trianglesObjetivo / (mallaOriginal.triangles.Length / 3);
            simplificador.SimplifyMesh(calidadCalculada);

            Mesh nuevaMalla = simplificador.ToMesh();
            nuevaMalla.name = malla.sharedMesh.name + "_simplified";

            // Recalcular normales y tangentes para mejor calidad visual
            nuevaMalla.RecalculateNormals();
            nuevaMalla.RecalculateTangents();

            if (guardarMallas)
            {
                GuardarMalla(nuevaMalla, nombreObjeto);
            }

            malla.sharedMesh = nuevaMalla;

            RegistrarEstadisticas(nombreObjeto, mallaOriginal, nuevaMalla);
            objetosSimplificados++;

            if (habilitarDebugLog)
                Debug.Log($"Simplificado: {nombreObjeto} - Reducción: {estadisticasPorObjeto[nombreObjeto].porcentajeReduccion:F2}%");
        }
        catch (System.Exception e)
        {
            Debug.LogError($"Error al simplificar {nombreObjeto}: {e.Message}");
            erroresEncontrados++;
        }
    }

    private void GuardarMalla(Mesh malla, string nombreObjeto)
    {
        string rutaCompleta = Path.Combine(carpetaDestino, $"{nombreObjeto}_simplified.asset");
        AssetDatabase.CreateAsset(malla, rutaCompleta);
        AssetDatabase.SaveAssets();
    }

    private void RegistrarEstadisticas(string nombreObjeto, Mesh original, Mesh nueva)
    {
        var stats = new MeshStats
        {
            verticesOriginales = original.vertexCount,
            trianglesOriginales = original.triangles.Length / 3,
            verticesFinales = nueva.vertexCount,
            trianglesFinales = nueva.triangles.Length / 3
        };

        stats.porcentajeReduccion = 100f * (1f - (float)stats.trianglesFinales / stats.trianglesOriginales);
        estadisticasPorObjeto[nombreObjeto] = stats;
        reduccionPromedio = estadisticasPorObjeto.Values.Average(s => s.porcentajeReduccion);
    }

    private void MostrarResumenFinal()
    {
        if (!mostrarEstadisticas) return;

        string resumen = $"\n=== RESUMEN DE SIMPLIFICACIÓN ===\n" +
                        $"Total objetos procesados: {totalObjetosProcesados}\n" +
                        $"Objetos simplificados: {objetosSimplificados}\n" +
                        $"Errores encontrados: {erroresEncontrados}\n" +
                        $"Reducción promedio: {reduccionPromedio:F2}%\n" +
                        "\nDetalles por objeto:\n";

        foreach (var kvp in estadisticasPorObjeto.OrderByDescending(x => x.Value.porcentajeReduccion))
        {
            resumen += $"\n{kvp.Key}:\n" +
                      $"  Vértices: {kvp.Value.verticesOriginales:N0} → {kvp.Value.verticesFinales:N0}\n" +
                      $"  Triángulos: {kvp.Value.trianglesOriginales:N0} → {kvp.Value.trianglesFinales:N0}\n" +
                      $"  Reducción: {kvp.Value.porcentajeReduccion:F2}%";
        }

        Debug.Log(resumen);
    }

    private void OnValidate()
    {
        triangulosMinimos = Mathf.Max(4, triangulosMinimos);
        umbralTriangulosMinimos = Mathf.Max(triangulosMinimos, umbralTriangulosMinimos);
    }
}
#endif