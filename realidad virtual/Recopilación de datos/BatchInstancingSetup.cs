using UnityEngine;
using UnityEditor;
using System.Collections.Generic;

public class BatchInstancingSetup : MonoBehaviour
{
    [SerializeField] private GameObject rootFolder;
    [SerializeField] private bool includeInactive = true;
    [SerializeField] private bool showDetailedLog = false;

    private struct MaterialProcessingResult
    {
        public int ProcessedCount;
        public int EnabledCount;
        public int ReadOnlyCount;
        public List<string> ReadOnlyMaterials;

        public MaterialProcessingResult(int processed = 0, int enabled = 0, int readOnly = 0)
        {
            ProcessedCount = processed;
            EnabledCount = enabled;
            ReadOnlyCount = readOnly;
            ReadOnlyMaterials = new List<string>();
        }

        public bool IsValid()
        {
            return ReadOnlyMaterials != null;
        }
    }

    [ContextMenu("Setup GPU Instancing For All Children")]
    public void SetupInstancing()
    {
        if (!ValidateSetup()) return;

        MaterialProcessingResult result = ProcessMaterials();
        if (result.IsValid())
        {
            SafeLogResults(result);
        }
    }

    private bool ValidateSetup()
    {
        if (rootFolder == null)
        {
            rootFolder = this.gameObject;
            Debug.LogWarning($"[BatchInstancingSetup] Root folder no especificado. Usando {gameObject.name}");
        }

        if (!SystemInfo.supportsInstancing)
        {
            Debug.LogError("[BatchInstancingSetup] GPU Instancing no está soportado en este sistema.");
            return false;
        }

        return true;
    }

    private MaterialProcessingResult ProcessMaterials()
    {
        MaterialProcessingResult result = new MaterialProcessingResult();
        HashSet<Material> processedMaterials = new HashSet<Material>();

        try
        {
            result.ReadOnlyMaterials = new List<string>();

            Undo.IncrementCurrentGroup();
            string undoGroupName = "Batch GPU Instancing Setup";
            Undo.SetCurrentGroupName(undoGroupName);

            if (rootFolder != null)
            {
                Renderer[] allRenderers = rootFolder.GetComponentsInChildren<Renderer>(includeInactive);

                if (allRenderers != null && allRenderers.Length > 0)
                {
                    foreach (Renderer renderer in allRenderers)
                    {
                        if (renderer != null)
                        {
                            ProcessRendererMaterials(renderer, processedMaterials, ref result);
                        }
                    }

                    if (result.EnabledCount > 0)
                    {
                        AssetDatabase.SaveAssets();
                    }
                }
                else
                {
                    Debug.LogWarning("[BatchInstancingSetup] No se encontraron renderers en la jerarquía.");
                }
            }
        }
        catch (System.Exception e)
        {
            Debug.LogError($"[BatchInstancingSetup] Error durante el procesamiento: {e.Message}\n{e.StackTrace}");
            Undo.RevertAllDownToGroup(Undo.GetCurrentGroup());
            return new MaterialProcessingResult();
        }

        return result;
    }

    private void ProcessRendererMaterials(Renderer renderer, HashSet<Material> processedMaterials, ref MaterialProcessingResult result)
    {
        if (renderer.sharedMaterials == null) return;

        Undo.RecordObject(renderer, "Enable Instancing on Renderer");

        foreach (Material material in renderer.sharedMaterials)
        {
            if (material == null || processedMaterials.Contains(material)) continue;

            processedMaterials.Add(material);
            result.ProcessedCount++;

            if (!AssetDatabase.IsOpenForEdit(material))
            {
                result.ReadOnlyCount++;
                result.ReadOnlyMaterials.Add(material.name);
                continue;
            }

            if (!material.enableInstancing)
            {
                Undo.RecordObject(material, "Enable Instancing");
                material.enableInstancing = true;
                EditorUtility.SetDirty(material);
                result.EnabledCount++;
            }
        }
    }

    private void SafeLogResults(MaterialProcessingResult result)
    {
        try
        {
            if (result.ProcessedCount == 0)
            {
                Debug.LogWarning("[BatchInstancingSetup] No se procesó ningún material.");
                return;
            }

            string logMessage = $"[BatchInstancingSetup] Proceso completado:\n" +
                              $"- Materiales procesados: {result.ProcessedCount}\n" +
                              $"- Instancing habilitado en: {result.EnabledCount} materiales\n" +
                              $"- Materiales de solo lectura: {result.ReadOnlyCount}";

            if (showDetailedLog && result.ReadOnlyMaterials.Count > 0)
            {
                logMessage += "\n\nMateriales de solo lectura:";
                foreach (string materialName in result.ReadOnlyMaterials)
                {
                    logMessage += $"\n- {materialName}";
                }
            }

            if (result.EnabledCount > 0)
                Debug.Log(logMessage);
            else
                Debug.LogWarning(logMessage);
        }
        catch (System.Exception e)
        {
            Debug.LogError($"[BatchInstancingSetup] Error al mostrar resultados: {e.Message}");
        }
    }

    [ContextMenu("Check Current Instancing Status")]
    public void CheckInstancingStatus()
    {
        if (!ValidateSetup()) return;

        try
        {
            Dictionary<string, (int enabled, int disabled)> shaderStats = new Dictionary<string, (int, int)>();
            HashSet<Material> processedMaterials = new HashSet<Material>();

            Renderer[] allRenderers = rootFolder.GetComponentsInChildren<Renderer>(includeInactive);

            foreach (Renderer renderer in allRenderers)
            {
                if (renderer == null || renderer.sharedMaterials == null) continue;

                foreach (Material material in renderer.sharedMaterials)
                {
                    if (material == null || processedMaterials.Contains(material)) continue;

                    processedMaterials.Add(material);
                    string shaderName = material.shader != null ? material.shader.name : "Unknown Shader";

                    if (!shaderStats.ContainsKey(shaderName))
                        shaderStats[shaderName] = (0, 0);

                    var current = shaderStats[shaderName];
                    if (material.enableInstancing)
                        shaderStats[shaderName] = (current.enabled + 1, current.disabled);
                    else
                        shaderStats[shaderName] = (current.enabled, current.disabled + 1);
                }
            }

            SafeLogInstancingStatus(shaderStats);
        }
        catch (System.Exception e)
        {
            Debug.LogError($"[BatchInstancingSetup] Error al verificar estado: {e.Message}");
        }
    }

    private void SafeLogInstancingStatus(Dictionary<string, (int enabled, int disabled)> shaderStats)
    {
        try
        {
            if (shaderStats.Count == 0)
            {
                Debug.LogWarning("[BatchInstancingSetup] No se encontraron materiales para analizar.");
                return;
            }

            int totalEnabled = 0, totalDisabled = 0;
            string detailedLog = "[BatchInstancingSetup] Estado de Instancing por Shader:\n";

            foreach (var stat in shaderStats)
            {
                totalEnabled += stat.Value.enabled;
                totalDisabled += stat.Value.disabled;

                if (showDetailedLog)
                {
                    detailedLog += $"\n{stat.Key}:\n" +
                                  $"  - Con instancing: {stat.Value.enabled}\n" +
                                  $"  - Sin instancing: {stat.Value.disabled}";
                }
            }

            string summary = $"[BatchInstancingSetup] Resumen:\n" +
                           $"- Total materiales con instancing: {totalEnabled}\n" +
                           $"- Total materiales sin instancing: {totalDisabled}";

            if (showDetailedLog)
                Debug.Log($"{summary}\n\n{detailedLog}");
            else
                Debug.Log(summary);
        }
        catch (System.Exception e)
        {
            Debug.LogError($"[BatchInstancingSetup] Error al mostrar estado: {e.Message}");
        }
    }
}

[CustomEditor(typeof(BatchInstancingSetup))]
public class BatchInstancingSetupEditor : Editor
{
    public override void OnInspectorGUI()
    {
        BatchInstancingSetup script = (BatchInstancingSetup)target;

        // Dibuja los campos por defecto
        DrawDefaultInspector();

        EditorGUILayout.Space(10);

        // Estilo para los botones
        GUIStyle buttonStyle = new GUIStyle(GUI.skin.button);
        buttonStyle.fontStyle = FontStyle.Bold;
        buttonStyle.padding = new RectOffset(10, 10, 6, 6);

        // Botón principal
        GUI.backgroundColor = new Color(0.4f, 0.8f, 0.4f);
        if (GUILayout.Button("INICIAR PROCESO DE GPU INSTANCING", buttonStyle, GUILayout.Height(30)))
        {
            script.SetupInstancing();
        }
        GUI.backgroundColor = Color.white;

        EditorGUILayout.Space(5);

        // Botón de verificación
        GUI.backgroundColor = new Color(0.4f, 0.6f, 0.8f);
        if (GUILayout.Button("Verificar Estado Actual", buttonStyle))
        {
            script.CheckInstancingStatus();
        }
        GUI.backgroundColor = Color.white;

        EditorGUILayout.Space(10);

        // Área informativa
        EditorGUILayout.HelpBox(
            "Este proceso activará GPU Instancing en todos los materiales del objeto seleccionado y sus hijos.\n\n" +
            "1. Asigna el Root Folder (opcional)\n" +
            "2. Ajusta las opciones según necesites\n" +
            "3. Presiona 'INICIAR PROCESO' para comenzar\n" +
            "4. Revisa la consola para ver los resultados",
            MessageType.Info);

        if (script.gameObject != null && script.enabled)
        {
            EditorGUILayout.Space(5);
            EditorGUILayout.HelpBox(
                "Si no asignas un Root Folder, se usará el objeto actual como raíz.",
                MessageType.Info);
        }
    }
}