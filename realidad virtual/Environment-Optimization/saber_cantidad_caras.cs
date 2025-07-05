using UnityEngine;
using UnityEditor;
using System.Collections.Generic;
using System.Linq;

#if UNITY_EDITOR
public class AnalizadorMallas : EditorWindow
{
    private List<Transform> carpetasAnalizar = new List<Transform>();
    private Vector2 scrollPosition;
    private List<ObjetoAnalizado> objetosAnalizados = new List<ObjetoAnalizado>();
    private bool mostrarTodo = false;
    private int topObjetos = 10;
    private bool agruparPorCarpeta = false;
    private string filtroNombre = "";

    [MenuItem("Window/Análisis de Mallas")]
    public static void ShowWindow()
    {
        GetWindow<AnalizadorMallas>("Analizador de Mallas");
    }

    private class ObjetoAnalizado
    {
        public string nombre;
        public string rutaCompleta;
        public string carpetaPadre;
        public int vertices;
        public int triangulos;
        public float tamañoMB;
    }

    private void OnGUI()
    {
        GUILayout.Label("Analizador de Mallas", EditorStyles.boldLabel);

        // Sección de configuración
        EditorGUILayout.BeginVertical("box");
        GUILayout.Label("Configuración", EditorStyles.boldLabel);

        // Lista de carpetas
        EditorGUILayout.LabelField("Carpetas a Analizar:");
        for (int i = 0; i < carpetasAnalizar.Count; i++)
        {
            EditorGUILayout.BeginHorizontal();
            carpetasAnalizar[i] = (Transform)EditorGUILayout.ObjectField(
                carpetasAnalizar[i], typeof(Transform), true);
            if (GUILayout.Button("-", GUILayout.Width(20)))
            {
                carpetasAnalizar.RemoveAt(i);
            }
            EditorGUILayout.EndHorizontal();
        }

        if (GUILayout.Button("Añadir Carpeta"))
        {
            carpetasAnalizar.Add(null);
        }

        // Opciones de visualización
        EditorGUILayout.Space();
        mostrarTodo = EditorGUILayout.Toggle("Mostrar Todos los Objetos", mostrarTodo);
        if (!mostrarTodo)
        {
            topObjetos = EditorGUILayout.IntSlider("Top Objetos a Mostrar", topObjetos, 1, 50);
        }
        agruparPorCarpeta = EditorGUILayout.Toggle("Agrupar por Carpeta", agruparPorCarpeta);
        filtroNombre = EditorGUILayout.TextField("Filtrar por Nombre", filtroNombre);

        EditorGUILayout.EndVertical();

        // Botón de análisis
        if (GUILayout.Button("Analizar Mallas"))
        {
            AnalizarMallas();
        }

        // Mostrar resultados
        if (objetosAnalizados.Count > 0)
        {
            MostrarResultados();
        }
    }

    private void AnalizarMallas()
    {
        objetosAnalizados.Clear();

        foreach (var carpeta in carpetasAnalizar)
        {
            if (carpeta == null) continue;

            // Obtener todos los MeshFilters en la carpeta y subcarpetas
            var meshFilters = carpeta.GetComponentsInChildren<MeshFilter>(true);
            foreach (var mf in meshFilters)
            {
                if (mf.sharedMesh != null)
                {
                    var mesh = mf.sharedMesh;
                    objetosAnalizados.Add(new ObjetoAnalizado
                    {
                        nombre = mf.gameObject.name,
                        rutaCompleta = GetGameObjectPath(mf.transform),
                        carpetaPadre = GetParentFolderName(mf.transform),
                        vertices = mesh.vertexCount,
                        triangulos = mesh.triangles.Length / 3,
                        tamañoMB = CalcularTamañoMeshMB(mesh)
                    });
                }
            }

            // Analizar SkinnedMeshRenderers también
            var skinnedMeshes = carpeta.GetComponentsInChildren<SkinnedMeshRenderer>(true);
            foreach (var smr in skinnedMeshes)
            {
                if (smr.sharedMesh != null)
                {
                    var mesh = smr.sharedMesh;
                    objetosAnalizados.Add(new ObjetoAnalizado
                    {
                        nombre = smr.gameObject.name,
                        rutaCompleta = GetGameObjectPath(smr.transform),
                        carpetaPadre = GetParentFolderName(smr.transform),
                        vertices = mesh.vertexCount,
                        triangulos = mesh.triangles.Length / 3,
                        tamañoMB = CalcularTamañoMeshMB(mesh)
                    });
                }
            }
        }
    }

    private void MostrarResultados()
    {
        EditorGUILayout.Space();
        EditorGUILayout.BeginVertical("box");

        // Estadísticas totales
        MostrarEstadisticasTotales();

        // Filtrar y ordenar resultados
        var resultadosFiltrados = objetosAnalizados
            .Where(o => string.IsNullOrEmpty(filtroNombre) ||
                       o.nombre.ToLower().Contains(filtroNombre.ToLower()))
            .OrderByDescending(o => o.triangulos);

        if (agruparPorCarpeta)
        {
            MostrarResultadosAgrupados(resultadosFiltrados);
        }
        else
        {
            MostrarResultadosPlanos(resultadosFiltrados);
        }

        EditorGUILayout.EndVertical();
    }

    private void MostrarEstadisticasTotales()
    {
        var totales = new
        {
            vertices = objetosAnalizados.Sum(o => o.vertices),
            triangulos = objetosAnalizados.Sum(o => o.triangulos),
            tamañoMB = objetosAnalizados.Sum(o => o.tamañoMB)
        };

        EditorGUILayout.LabelField("Estadísticas Totales:", EditorStyles.boldLabel);
        EditorGUILayout.LabelField($"Total Objetos: {objetosAnalizados.Count}");
        EditorGUILayout.LabelField($"Total Vértices: {totales.vertices:N0}");
        EditorGUILayout.LabelField($"Total Triángulos: {totales.triangulos:N0}");
        EditorGUILayout.LabelField($"Tamaño Total: {totales.tamañoMB:F2} MB");
        EditorGUILayout.Space();
    }

    private void MostrarResultadosPlanos(IEnumerable<ObjetoAnalizado> resultados)
    {
        scrollPosition = EditorGUILayout.BeginScrollView(scrollPosition);

        EditorGUILayout.LabelField("Objetos por Cantidad de Triángulos:", EditorStyles.boldLabel);

        var listaFinal = mostrarTodo ? resultados : resultados.Take(topObjetos);

        foreach (var obj in listaFinal)
        {
            EditorGUILayout.BeginVertical("box");
            EditorGUILayout.LabelField($"Nombre: {obj.nombre}");
            EditorGUILayout.LabelField($"Ruta: {obj.rutaCompleta}");
            EditorGUILayout.LabelField($"Vértices: {obj.vertices:N0}");
            EditorGUILayout.LabelField($"Triángulos: {obj.triangulos:N0}");
            EditorGUILayout.LabelField($"Tamaño: {obj.tamañoMB:F2} MB");
            EditorGUILayout.EndVertical();
            EditorGUILayout.Space();
        }

        EditorGUILayout.EndScrollView();
    }

    private void MostrarResultadosAgrupados(IEnumerable<ObjetoAnalizado> resultados)
    {
        scrollPosition = EditorGUILayout.BeginScrollView(scrollPosition);

        var grupos = resultados.GroupBy(o => o.carpetaPadre)
                              .OrderByDescending(g => g.Sum(o => o.triangulos));

        foreach (var grupo in grupos)
        {
            EditorGUILayout.BeginVertical("box");
            EditorGUILayout.LabelField($"Carpeta: {grupo.Key}", EditorStyles.boldLabel);

            var estadisticasGrupo = new
            {
                vertices = grupo.Sum(o => o.vertices),
                triangulos = grupo.Sum(o => o.triangulos),
                tamañoMB = grupo.Sum(o => o.tamañoMB)
            };

            EditorGUILayout.LabelField($"Total Vértices: {estadisticasGrupo.vertices:N0}");
            EditorGUILayout.LabelField($"Total Triángulos: {estadisticasGrupo.triangulos:N0}");
            EditorGUILayout.LabelField($"Tamaño Total: {estadisticasGrupo.tamañoMB:F2} MB");

            var objetosGrupo = mostrarTodo ? grupo : grupo.Take(topObjetos);
            foreach (var obj in objetosGrupo)
            {
                EditorGUILayout.BeginVertical("helpBox");
                EditorGUILayout.LabelField($"Nombre: {obj.nombre}");
                EditorGUILayout.LabelField($"Vértices: {obj.vertices:N0}");
                EditorGUILayout.LabelField($"Triángulos: {obj.triangulos:N0}");
                EditorGUILayout.EndVertical();
            }

            EditorGUILayout.EndVertical();
            EditorGUILayout.Space();
        }

        EditorGUILayout.EndScrollView();
    }

    private string GetGameObjectPath(Transform transform)
    {
        string path = transform.name;
        Transform parent = transform.parent;

        while (parent != null)
        {
            path = parent.name + "/" + path;
            parent = parent.parent;
        }

        return path;
    }

    private string GetParentFolderName(Transform transform)
    {
        if (transform.parent == null) return "Root";
        return transform.parent.name;
    }

    private float CalcularTamañoMeshMB(Mesh mesh)
    {
        float tamaño = 0;

        // Vertices (12 bytes por vértice - 3 floats x 4 bytes)
        tamaño += mesh.vertexCount * 12;

        // Normales (12 bytes por normal)
        if (mesh.normals != null && mesh.normals.Length > 0)
            tamaño += mesh.normals.Length * 12;

        // UVs (8 bytes por UV - 2 floats x 4 bytes)
        if (mesh.uv != null && mesh.uv.Length > 0)
            tamaño += mesh.uv.Length * 8;

        // Triangulos (4 bytes por índice)
        tamaño += mesh.triangles.Length * 4;

        // Convertir a MB
        return tamaño / (1024 * 1024);
    }
}
#endif