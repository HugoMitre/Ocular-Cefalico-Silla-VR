using System.Collections.Generic;
using System.IO;
using UnityEngine;
using System.Text;
using System.Threading;

public class GazeStandardDeviation : MonoBehaviour
{
    // Propiedades públicas para el DataCombiner
    public Vector2 UltimaDesviacionGaze
    {
        get
        {
            if (desviacionesEstandar.Count > 0)
                return desviacionesEstandar[desviacionesEstandar.Count - 1];
            return Vector2.zero;
        }
    }

    public Vector2 UltimaDesviacionGazeNormalizada
    {
        get
        {
            if (desviacionesNormalizadas.Count > 0)
                return desviacionesNormalizadas[desviacionesNormalizadas.Count - 1];
            return Vector2.zero;
        }
    }

    // Referencia al LineRenderer que define el rayo de mirada
    private LineRenderer gazeRayLine;
    private Vector3 currentDirection;
    private Vector2 currentGazeAngles = Vector2.zero;
    private Vector2 previousGazeAngles = Vector2.zero;
    private float deltaTime = 0.2f;
    private float timer = 0f;

    // Cola para almacenar los últimos 10 ángulos de mirada
    private Queue<Vector2> gazeAnglesQueue = new Queue<Vector2>();
    private const int QUEUE_SIZE = 10;

    // Valor máximo para normalización (usamos 90° porque los ángulos se esperan en -90 a 90)
    private const float MAX_DEVIATION = 90f;

    private List<float> tiempos = new List<float>();
    private List<Vector2> desviacionesEstandar = new List<Vector2>();
    private List<Vector2> desviacionesNormalizadas = new List<Vector2>();

    // VARIABLES PARA LAS ÁREAS (ZONAS)
    [Header("Zonas para Gaze (Asignar BoxColliders en el Inspector)")]
    public BoxCollider areaFront;
    public BoxCollider areaDown;
    public BoxCollider areaUp;
    public BoxCollider areaLeft;
    public BoxCollider areaRight;
    public BoxCollider areaUpLeft;
    public BoxCollider areaUpRight;
    public BoxCollider areaDownLeft;
    public BoxCollider areaDownRight;
    public BoxCollider areaRightExtension;
    public BoxCollider areaDownRightExtension;
    public BoxCollider areaDownLeftExtension;
    public BoxCollider areaDownExtension;
    public BoxCollider areaLeftExtension;

    private Dictionary<BoxCollider, string> areaNames;

    // Opcional: rangos para determinar la dirección a partir de los ángulos
    private readonly Dictionary<string, (float minH, float maxH, float minV, float maxV)> angleRanges =
        new Dictionary<string, (float, float, float, float)>
    {
        { "Front", (-15f, 15f, -15f, 15f) },
        { "Left", (-112.5f, -15f, -15f, 15f) },
        { "Right", (15f, 112.5f, -15f, 15f) },
        { "Up", (-15f, 15f, 15f, 180f) },
        { "Down", (-15f, 15f, -180f, -15f) },
        { "UpLeft", (-112.5f, -15f, 15f, 120f) },
        { "UpRight", (15f, 112.5f, 15f, 120f) },
        { "DownLeft", (-112.5f, -15f, -120f, -15f) },
        { "DownRight", (15f, 112.5f, -120f, -15f) }
    };

    void Start()
    {
        // Inicializa el LineRenderer
        gazeRayLine = GetComponent<LineRenderer>();
        if (gazeRayLine == null)
        {
            Debug.LogError("No se encontró el LineRenderer para la mirada!");
            enabled = false;
            return;
        }

        // Inicializa la cola con valores cero
        for (int i = 0; i < QUEUE_SIZE; i++)
        {
            gazeAnglesQueue.Enqueue(Vector2.zero);
        }

        // Inicializa el diccionario de áreas y las zonas
        areaNames = new Dictionary<BoxCollider, string>();
        InitializeAreas();

        // Primera muestra de ángulos de mirada
        previousGazeAngles = RecordGazeDirection();
    }

    void Update()
    {
        if (gazeRayLine == null) return;

        timer += Time.deltaTime;
        if (timer >= deltaTime)
        {
            // Actualiza la dirección y obtiene los ángulos usando la lógica completa con zonas
            currentGazeAngles = RecordGazeDirection();

            // Actualiza la ventana de muestras
            if (gazeAnglesQueue.Count >= QUEUE_SIZE)
                gazeAnglesQueue.Dequeue();
            gazeAnglesQueue.Enqueue(currentGazeAngles);

            // Calcula la variabilidad (desviación simple) a partir de la ventana de muestras
            Vector2 stdDev = CalculateStandardDeviation();
            if (stdDev.magnitude > 0.0001f)
            {
                Vector2 normalizedStdDev = new Vector2(
                    NormalizeValue(stdDev.x),
                    NormalizeValue(stdDev.y)
                );

                tiempos.Add(Time.time);
                desviacionesEstandar.Add(stdDev);
                desviacionesNormalizadas.Add(normalizedStdDev);

                Debug.Log($"Time: {Time.time:F3}s, StdDev: {stdDev}, Normalized: {normalizedStdDev}");
            }

            previousGazeAngles = currentGazeAngles;
            timer = 0f;
        }
    }

    /// <summary>
    /// Inicializa las áreas (zonas) asignadas para la mirada.
    /// </summary>
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
            Debug.LogError("¡No hay áreas asignadas! Por favor asigne BoxColliders en el inspector.");
            enabled = false;
        }
    }

    /// <summary>
    /// Agrega un BoxCollider y su nombre asociado al diccionario de áreas.
    /// </summary>
    private void InitializeArea(BoxCollider collider, string name)
    {
        if (collider != null)
        {
            areaNames.Add(collider, name);
            collider.isTrigger = true;
        }
    }

    /// <summary>
    /// Registra la dirección de la mirada y obtiene los ángulos utilizando la lógica de GazeDirectionwitharea.
    /// Convierte la dirección del rayo a coordenadas locales, calcula:
    /// - rawAngleH mediante Atan2(localDir.x, localDir.z)
    /// - rawAngleV mediante Atan2(localDir.y, horizontalMagnitude)
    /// Luego se normaliza el ángulo horizontal (restándole 90° y ajustándolo a [-180, 180])
    /// y se multiplica el ángulo vertical por 2 para fines de visualización.
    /// También se detecta la zona mediante colisiones (opcional, para registro o debug).
    /// </summary>
    /// <returns>Vector2 (GazeAngleX, GazeAngleY display)</returns>
    private Vector2 RecordGazeDirection()
    {
        // Se obtienen las posiciones definidas en el LineRenderer
        Vector3[] positions = new Vector3[2];
        gazeRayLine.GetPositions(positions);
        currentDirection = (positions[1] - positions[0]).normalized;

        // Convierte la dirección mundial a local para que las zonas tengan sentido
        Vector3 localDir = transform.InverseTransformDirection(currentDirection);

        // Calcula los ángulos raw
        float rawAngleH = Mathf.Atan2(localDir.x, localDir.z) * Mathf.Rad2Deg;
        float rawAngleV = Mathf.Atan2(localDir.y, new Vector2(localDir.x, localDir.z).magnitude) * Mathf.Rad2Deg;

        // Se aplica NormalizeAngle al horizontal (resta 90° y ajusta al rango [-180, 180])
        float normalizedAngleH = NormalizeAngle(rawAngleH);
        // Para el vertical se conserva el valor raw
        float normalizedAngleV = rawAngleV;
        // Se multiplica el ángulo vertical por 2 para visualización
        float displayAngleV = normalizedAngleV * 2f;

        // Detecta la zona usando colisiones (opcional)
        string detectedZone = DetectDirectionFromColliders(positions[0], currentDirection);
        string zoneByAngle = GetDirectionFromAngles(normalizedAngleH, normalizedAngleV);
        string finalZone = detectedZone != "OutOfArea" ? detectedZone : zoneByAngle;

        Debug.Log($"GazeZone: {finalZone}, Angles: ({normalizedAngleH:F2}°, {displayAngleV:F2}°)");

        return new Vector2(normalizedAngleH, displayAngleV);
    }

    /// <summary>
    /// Ajusta el ángulo restando 90° y limitándolo al rango [-180, 180].
    /// </summary>
    private float NormalizeAngle(float angle)
    {
        angle -= 90f;
        while (angle > 180f) angle -= 360f;
        while (angle < -180f) angle += 360f;
        return angle;
    }

    /// <summary>
    /// Normaliza un ángulo a [-1, 1] usando 90° como base. Si es vertical, se multiplica por 2.
    /// </summary>
    private float NormalizeToMinusOneOne(float angle, bool isVertical = false)
    {
        if (isVertical)
        {
            angle = angle * 2f;
            return Mathf.Clamp(angle / 90f, -1f, 1f);
        }
        else
        {
            return Mathf.Clamp(angle / 90f, -1f, 1f);
        }
    }

    /// <summary>
    /// Detecta la zona de la mirada usando un SphereCast y compara el collider impactado con las áreas asignadas.
    /// </summary>
    private string DetectDirectionFromColliders(Vector3 origin, Vector3 direction)
    {
        RaycastHit[] hitBuffer = new RaycastHit[10];
        int hitCount = Physics.SphereCastNonAlloc(origin, 0.1f, direction, hitBuffer, 100f);
        for (int i = 0; i < hitCount; i++)
        {
            BoxCollider hitCollider = hitBuffer[i].collider as BoxCollider;
            if (hitCollider != null && areaNames.ContainsKey(hitCollider))
                return areaNames[hitCollider];
        }
        return "OutOfArea";
    }

    /// <summary>
    /// Determina la dirección de la mirada basada en los ángulos y rangos predefinidos.
    /// </summary>
    private string GetDirectionFromAngles(float horizontalAngle, float verticalAngle)
    {
        foreach (var range in angleRanges)
        {
            if (horizontalAngle >= range.Value.minH &&
                horizontalAngle <= range.Value.maxH &&
                verticalAngle >= range.Value.minV &&
                verticalAngle <= range.Value.maxV)
            {
                return range.Key;
            }
        }
        return "OutOfArea";
    }

    /// <summary>
    /// Calcula la "desviación simple" (cambio angular) a partir de la ventana de muestras.
    /// Se calcula la media, la varianza (ajustada para la naturaleza circular) y se extrae la raíz cuadrada,
    /// asignando un signo según la dirección del movimiento acumulado.
    /// </summary>
    private Vector2 CalculateStandardDeviation()
    {
        if (gazeAnglesQueue.Count == 0)
            return Vector2.zero;

        // Calcular la media de la ventana
        Vector2 mean = Vector2.zero;
        foreach (Vector2 angle in gazeAnglesQueue)
            mean += angle;
        mean /= gazeAnglesQueue.Count;

        Vector2 variance = Vector2.zero;
        Vector2 direction = Vector2.zero; // Para determinar el signo

        foreach (Vector2 angle in gazeAnglesQueue)
        {
            Vector2 diff = angle - mean;
            // Ajuste para diferencias mayores a 180° por la naturaleza circular
            if (Mathf.Abs(diff.x) > 180f)
                diff.x = 360f - Mathf.Abs(diff.x);
            if (Mathf.Abs(diff.y) > 180f)
                diff.y = 360f - Mathf.Abs(diff.y);

            variance.x += diff.x * diff.x;
            variance.y += diff.y * diff.y;
            // Acumula el cambio respecto a la muestra anterior
            direction += (angle - previousGazeAngles);
        }

        if (gazeAnglesQueue.Count > 1)
            variance /= (gazeAnglesQueue.Count - 1);

        return new Vector2(
            -Mathf.Sign(direction.x) * Mathf.Sqrt(variance.x),
             Mathf.Sign(direction.y) * Mathf.Sqrt(variance.y)
        );
    }

    /// <summary>
    /// Normaliza un valor al rango [-1, 1] usando MAX_DEVIATION como referencia.
    /// </summary>
    private float NormalizeValue(float value)
    {
        return Mathf.Clamp(value / MAX_DEVIATION, -1f, 1f);
    }

    /// <summary>
    /// Exporta los datos recopilados a un archivo CSV con la cabecera:
    /// Tiempo,DesviacionEstandar_X,DesviacionEstandar_Y,DesviacionNormalizada_X,DesviacionNormalizada_Y
    /// </summary>
    public void GuardarDatosEnCSV()
    {
        StringBuilder csv = new StringBuilder();
        csv.AppendLine("Tiempo,DesviacionEstandar_X,DesviacionEstandar_Y,DesviacionNormalizada_X,DesviacionNormalizada_Y");

        for (int i = 0; i < desviacionesEstandar.Count; i++)
        {
            csv.AppendLine($"{tiempos[i]:F3},{desviacionesEstandar[i].x:F6},{desviacionesEstandar[i].y:F6}," +
                           $"{desviacionesNormalizadas[i].x:F6},{desviacionesNormalizadas[i].y:F6}");
        }

        string carpeta = @"C:\Users\Manuel Delado\Documents";
        string prefijo = "desviacion_estandar_gaze";
        string extension = ".csv";

        bool archivoGuardado = false;
        int intentos = 0;
        string rutaArchivo = "";

        while (!archivoGuardado && intentos < 5)
        {
            try
            {
                rutaArchivo = ObtenerSiguienteNombreArchivo(carpeta, prefijo, extension);
                File.WriteAllText(rutaArchivo, csv.ToString());
                archivoGuardado = true;
                Debug.Log($"Datos guardados exitosamente en: {rutaArchivo}");
            }
            catch (IOException)
            {
                intentos++;
                Thread.Sleep(100);
            }
        }

        if (!archivoGuardado)
        {
            string fechaHora = System.DateTime.Now.ToString("yyyyMMdd_HHmmss");
            rutaArchivo = Path.Combine(carpeta, $"{prefijo}_{fechaHora}{extension}");
            File.WriteAllText(rutaArchivo, csv.ToString());
            Debug.Log($"Datos guardados con timestamp en: {rutaArchivo}");
        }
    }

    /// <summary>
    /// Genera un nombre de archivo que no exista en la carpeta dada.
    /// </summary>
    private string ObtenerSiguienteNombreArchivo(string carpeta, string prefijo, string extension)
    {
        int numero = 1;
        string nombreArchivo;
        do
        {
            nombreArchivo = Path.Combine(carpeta, $"{prefijo}{numero}{extension}");
            numero++;
        } while (File.Exists(nombreArchivo));
        return nombreArchivo;
    }

    void OnDisable()
    {
        GuardarDatosEnCSV();
    }
}
