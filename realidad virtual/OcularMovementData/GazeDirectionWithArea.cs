using UnityEngine;
using System.Collections.Generic;
using System.IO;
using System.Text;

public class GazeDirectionwitharea : MonoBehaviour
{
    public float UltimoAnguloGazeX
    {
        get { return records.Count > 0 ? records[records.Count - 1].angleH : 0f; }
    }

    public float UltimoAnguloGazeY
    {
        get { return records.Count > 0 ? records[records.Count - 1].angleV : 0f; }
    }

    public float UltimoAnguloGazeXNormalizado
    {
        get { return records.Count > 0 ? records[records.Count - 1].normH : 0f; }
    }

    public float UltimoAnguloGazeYNormalizado
    {
        get { return records.Count > 0 ? records[records.Count - 1].normV : 0f; }
    }

    public string UltimaDireccionGaze
    {
        get { return records.Count > 0 ? records[records.Count - 1].direction : "OutOfArea"; }
    }

    [SerializeField] private LineRenderer gazeRayLine;
    [SerializeField] private float sampleInterval = 0.2f;
    [SerializeField] private LayerMask hitLayers;

    [SerializeField] public BoxCollider areaFront;
    [SerializeField] public BoxCollider areaDown;
    [SerializeField] public BoxCollider areaUp;
    [SerializeField] public BoxCollider areaLeft;
    [SerializeField] public BoxCollider areaRight;
    [SerializeField] public BoxCollider areaUpLeft;
    [SerializeField] public BoxCollider areaUpRight;
    [SerializeField] public BoxCollider areaDownLeft;
    [SerializeField] public BoxCollider areaDownRight;
    [SerializeField] public BoxCollider areaRightExtension;
    [SerializeField] public BoxCollider areaDownRightExtension;
    [SerializeField] public BoxCollider areaDownLeftExtension;
    [SerializeField] public BoxCollider areaDownExtension;
    [SerializeField] public BoxCollider areaLeftExtension;

    private float lastSampleTime;
    private float startTime;
    private Dictionary<BoxCollider, string> areaNames;
    private List<(float time, float angleH, float angleV, float normH, float normV, string direction)> records;
    private RaycastHit[] hitBuffer = new RaycastHit[10];

    private readonly Dictionary<string, (float minH, float maxH, float minV, float maxV)> angleRanges =
        new Dictionary<string, (float minH, float maxH, float minV, float maxV)>
    {
        { "Front", (-15f, 15f, -15f, 15f) },
        { "Left", (-112.5f, -15f, -15f, 15f) },
        { "LeftExtension", (-180f, -112.5f, -15f, 15f) },
        { "Right", (15f, 112.5f, -15f, 15f) },
        { "RightExtension", (112.5f, 180f, -15f, 15f) },
        { "Up", (-15f, 15f, 15f, 180f) },
        { "Down", (-15f, 15f, -180f, -15f) },
        { "UpRight", (15f, 112.5f, 15f, 120f) },
        { "UpLeft", (-112.5f, -15f, 15f, 120f) },
        { "DownRight", (15f, 112.5f, -120f, -15f) },
        { "DownLeft", (-112.5f, -15f, -120f, -15f) }
    };

    private void Awake()
    {
        records = new List<(float, float, float, float, float, string)>();
        areaNames = new Dictionary<BoxCollider, string>();
    }

    private void Start()
    {
        startTime = Time.time;

        if (gazeRayLine == null)
        {
            Debug.LogError("LineRenderer no asignado!");
            enabled = false;
            return;
        }

        InitializeAreas();

        if (hitLayers.value == 0)
        {
            hitLayers = LayerMask.GetMask("Default");
        }
    }

    private void InitializeAreas()
    {
        InitializeArea(areaFront, "Front");
        InitializeArea(areaDown, "Down");
        InitializeArea(areaUp, "Up");
        InitializeArea(areaLeft, "Left");
        InitializeArea(areaRight, "Right");
        InitializeArea(areaUpLeft, "UpLeft");
        InitializeArea(areaUpRight, "UpRight");
        InitializeArea(areaDownLeft, "DownLeft");
        InitializeArea(areaDownRight, "DownRight");
        InitializeArea(areaRightExtension, "Right");
        InitializeArea(areaDownRightExtension, "DownRight");
        InitializeArea(areaDownLeftExtension, "DownLeft");
        InitializeArea(areaDownExtension, "Down");
        InitializeArea(areaLeftExtension, "Left");

        if (areaNames.Count == 0)
        {
            Debug.LogError("¡No hay áreas asignadas! Por favor asigne box colliders en el inspector.");
            enabled = false;
        }
    }

    private void InitializeArea(BoxCollider collider, string name)
    {
        if (collider != null)
        {
            areaNames.Add(collider, name);
            collider.isTrigger = true;
        }
    }

    private float NormalizeAngle(float angle)
    {
        angle -= 90f;
        while (angle > 180f) angle -= 360f;
        while (angle < -180f) angle += 360f;
        return angle;
    }

    private float NormalizeToMinusOneOne(float angle, bool isVertical = false)
    {
        if (isVertical)
        {
            // Para ángulos verticales, primero multiplicamos por 2 y luego normalizamos a -1 a 1
            angle = angle * 2f;
            return Mathf.Clamp(angle / 90f, -1f, 1f);
        }
        else
        {
            // Para ángulos horizontales, normalizamos directamente a -1 a 1 usando 90 como base
            return Mathf.Clamp(angle / 90f, -1f, 1f);
        }
    }

    private void Update()
    {
        if (Time.time - lastSampleTime >= sampleInterval)
        {
            lastSampleTime = Time.time;
            RecordGazeDirection();
        }
    }

    private void RecordGazeDirection()
    {
        Vector3[] positions = new Vector3[2];
        gazeRayLine.GetPositions(positions);
        Vector3 worldDir = (positions[1] - positions[0]).normalized;
        Vector3 localDir = transform.InverseTransformDirection(worldDir);

        // Calcula ángulos raw
        float rawAngleH = Mathf.Atan2(localDir.x, localDir.z) * Mathf.Rad2Deg;
        float rawAngleV = Mathf.Atan2(localDir.y, new Vector2(localDir.x, localDir.z).magnitude) * Mathf.Rad2Deg;

        // Normaliza a -180 a 180 (mantenemos este paso para la detección de dirección)
        float normalizedAngleH = NormalizeAngle(rawAngleH);
        float normalizedAngleV = rawAngleV;

        // Para mostrar y normalizar, multiplicamos el ángulo vertical por 2
        float displayAngleV = normalizedAngleV * 2f;

        // Normaliza a -1 a 1 basado en 90 grados
        float normalizedMinusOneOneH = NormalizeToMinusOneOne(normalizedAngleH);
        float normalizedMinusOneOneV = NormalizeToMinusOneOne(normalizedAngleV, true);

        string detectedDirection = DetectDirectionFromColliders(positions[0], worldDir);
        string direction = GetDirectionFromAngles(normalizedAngleH, normalizedAngleV);
        string finalDirection = detectedDirection != "OutOfArea" ? detectedDirection : direction;

        float currentTime = Time.time - startTime;
        records.Add((currentTime, normalizedAngleH, displayAngleV, normalizedMinusOneOneH, normalizedMinusOneOneV, finalDirection));

        Debug.Log($"Time: {currentTime:F2}s, " +
                 $"GazeAngleX: {normalizedAngleH:F2}°, GazeAngleY: {displayAngleV:F2}°, " +
                 $"GazeNormalizedX: {normalizedMinusOneOneH:F2}, GazeNormalizedY: {normalizedMinusOneOneV:F2}, " +
                 $"GazeDirection: {finalDirection}");
    }

    private string DetectDirectionFromColliders(Vector3 origin, Vector3 direction)
    {
        int hitCount = Physics.SphereCastNonAlloc(origin, 0.1f, direction, hitBuffer, 100f, hitLayers);

        for (int i = 0; i < hitCount; i++)
        {
            BoxCollider hitCollider = hitBuffer[i].collider as BoxCollider;
            if (hitCollider != null && areaNames.ContainsKey(hitCollider))
            {
                return areaNames[hitCollider];
            }
        }

        return "OutOfArea";
    }

    private string GetDirectionFromAngles(float horizontalAngle, float verticalAngle)
    {
        foreach (var range in angleRanges)
        {
            if (horizontalAngle >= range.Value.minH &&
                horizontalAngle <= range.Value.maxH &&
                verticalAngle >= range.Value.minV &&
                verticalAngle <= range.Value.maxV)
            {
                return range.Key.Replace("Extension", "");
            }
        }
        return "OutOfArea";
    }

    public void SaveDataToCSV()
    {
        if (records.Count == 0)
        {
            Debug.LogWarning("¡No hay datos para guardar!");
            return;
        }

        StringBuilder csv = new StringBuilder();
        csv.AppendLine("Time,GazeAngleX,GazeAngleY,GazeNormalizedX,GazeNormalizedY,GazeDirection");

        foreach (var record in records)
        {
            csv.AppendLine($"{record.time:F3}," +
                          $"{record.angleH:F6}," +
                          $"{record.angleV:F6}," +   // Ya está multiplicado por 2
                          $"{record.normH:F6}," +
                          $"{record.normV:F6}," +
                          $"{record.direction}");
        }

        string folder = Application.persistentDataPath;
        string prefix = "gaze_tracking";
        string timestamp = System.DateTime.Now.ToString("yyyyMMdd_HHmmss");
        string filePath = Path.Combine(folder, $"{prefix}_{timestamp}.csv");

        try
        {
            File.WriteAllText(filePath, csv.ToString());
            Debug.Log($"Datos guardados en: {filePath}");
        }
        catch (IOException ex)
        {
            Debug.LogError($"Error al guardar datos: {ex.Message}");
        }
    }

    private void OnDisable()
    {
        SaveDataToCSV();
    }
}