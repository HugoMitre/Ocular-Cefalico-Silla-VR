using System.Collections.Generic;
using System.IO;
using UnityEngine;
using System.Text;
using System.Threading;

public class GazeDataLogger : MonoBehaviour
{
    // Propiedades p�blicas para el DataCombiner
    public Vector2 UltimaVelocidadGaze
    {
        get { return velocidades.Count > 0 ? velocidades[velocidades.Count - 1] : Vector2.zero; }
    }

    public Vector2 UltimaVelocidadGazeNormalizada
    {
        get { return velocidadesNormalizadas.Count > 0 ? velocidadesNormalizadas[velocidadesNormalizadas.Count - 1] : Vector2.zero; }
    }

    // Referencia al LineRenderer que define el rayo de mirada
    private LineRenderer gazeRayLine;
    // Para obtener la direcci�n y los �ngulos de mirada (usando la l�gica de GazeDirectionwitharea)
    private Vector3 currentDirection;
    private Vector2 currentGazeAngles = Vector2.zero;
    private Vector2 previousGazeAngles = Vector2.zero;

    private float deltaTime = 0.2f;
    private float timer = 0f;

    // Cola para filtrar velocidades (media m�vil) � tama�o de ventana: 5
    private Queue<Vector2> gazeVelocities = new Queue<Vector2>();
    private const int WINDOW_SIZE = 5;
    private const float MOVEMENT_THRESHOLD = 0.001f;
    // Valores m�ximos para normalizaci�n de la velocidad angular (en �/s)
    private const float MAX_VELOCITY_X = 110f;
    private const float MAX_VELOCITY_Y = 90f;

    // Listas para almacenar datos
    private List<float> tiempos = new List<float>();
    private List<Vector2> velocidades = new List<Vector2>();
    private List<Vector2> velocidadesNormalizadas = new List<Vector2>();

    // VARIABLES PARA LAS �REAS (ZONAS) � utilizadas para obtener los �ngulos de mirada
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
    // Opcional: rangos para determinar la direcci�n a partir de los �ngulos
    private readonly Dictionary<string, (float minH, float maxH, float minV, float maxV)> angleRanges =
        new Dictionary<string, (float, float, float, float)>
        {
            { "Front",     (-15f,   15f,  -15f,   15f) },
            { "Left",      (-112.5f, -15f, -15f,   15f) },
            { "Right",     ( 15f,  112.5f, -15f,   15f) },
            { "Up",        (-15f,   15f,   15f,  180f) },
            { "Down",      (-15f,   15f, -180f,  -15f) },
            { "UpLeft",    (-112.5f, -15f,  15f,  120f) },
            { "UpRight",   ( 15f,  112.5f,  15f,  120f) },
            { "DownLeft",  (-112.5f, -15f, -120f, -15f) },
            { "DownRight", ( 15f,  112.5f, -120f, -15f) }
        };

    void Start()
    {
        gazeRayLine = GetComponent<LineRenderer>();
        if (gazeRayLine == null)
        {
            Debug.LogError("�No se encontr� el LineRenderer para la mirada!");
            enabled = false;
            return;
        }

        // Inicializa la cola de velocidades (para el filtro de media m�vil)
        InitializeVelocityQueue();

        // Inicializa el diccionario de �reas y las zonas
        areaNames = new Dictionary<BoxCollider, string>();
        InitializeAreas();

        // Obt�n la primera muestra de �ngulos de mirada usando la l�gica de GazeDirectionwitharea
        currentGazeAngles = RecordGazeDirection();
        previousGazeAngles = currentGazeAngles;
    }

    /// <summary>
    /// M�todo que asocia cada BoxCollider a un nombre de zona.
    /// Debes asegurarte de que cada BoxCollider est� asignado en el Inspector.
    /// </summary>
    void InitializeAreas()
    {
        if (areaFront != null) areaNames.Add(areaFront, "Front");
        if (areaDown != null) areaNames.Add(areaDown, "Down");
        if (areaUp != null) areaNames.Add(areaUp, "Up");
        if (areaLeft != null) areaNames.Add(areaLeft, "Left");
        if (areaRight != null) areaNames.Add(areaRight, "Right");
        if (areaUpLeft != null) areaNames.Add(areaUpLeft, "UpLeft");
        if (areaUpRight != null) areaNames.Add(areaUpRight, "UpRight");
        if (areaDownLeft != null) areaNames.Add(areaDownLeft, "DownLeft");
        if (areaDownRight != null) areaNames.Add(areaDownRight, "DownRight");
        if (areaRightExtension != null) areaNames.Add(areaRightExtension, "RightExtension");
        if (areaDownRightExtension != null) areaNames.Add(areaDownRightExtension, "DownRightExtension");
        if (areaDownLeftExtension != null) areaNames.Add(areaDownLeftExtension, "DownLeftExtension");
        if (areaDownExtension != null) areaNames.Add(areaDownExtension, "DownExtension");
        if (areaLeftExtension != null) areaNames.Add(areaLeftExtension, "LeftExtension");
    }

    void InitializeVelocityQueue()
    {
        gazeVelocities.Clear();
        for (int i = 0; i < WINDOW_SIZE; i++)
        {
            gazeVelocities.Enqueue(Vector2.zero);
        }
    }

    void Update()
    {
        if (gazeRayLine == null) return;

        timer += Time.deltaTime;
        if (timer >= deltaTime)
        {
            // Obt�n la nueva direcci�n de mirada (con zonas) y sus �ngulos
            currentGazeAngles = RecordGazeDirection();

            // Calcula la velocidad angular como diferencia entre �ngulos (usando Mathf.DeltaAngle) dividido por deltaTime
            float deltaX = Mathf.DeltaAngle(previousGazeAngles.x, currentGazeAngles.x);
            float deltaY = Mathf.DeltaAngle(previousGazeAngles.y, currentGazeAngles.y);
            Vector2 gazeAngularVelocity = new Vector2(deltaX, deltaY) / deltaTime;

            // Aplica el filtro de media m�vil (ventana de 5 muestras)
            Vector2 filteredVelocity = ApplyMovingAverageFilter(gazeVelocities, gazeAngularVelocity);

            // Si la velocidad filtrada supera el umbral en alguno de los ejes, reg�strala
            if (Mathf.Abs(filteredVelocity.x) > MOVEMENT_THRESHOLD || Mathf.Abs(filteredVelocity.y) > MOVEMENT_THRESHOLD)
            {
                Vector2 normalizedVelocity = new Vector2(
                    NormalizeValue(filteredVelocity.x, MAX_VELOCITY_X),
                    NormalizeValue(filteredVelocity.y, MAX_VELOCITY_Y)
                );

                tiempos.Add(Time.time);
                velocidades.Add(filteredVelocity);
                velocidadesNormalizadas.Add(normalizedVelocity);

                Debug.Log($"Time: {Time.time:F3}s, Velocidad Gaze - X: {filteredVelocity.x:F3}�/s, Y: {filteredVelocity.y:F3}�/s, " +
                          $"Normalizada - X: {normalizedVelocity.x:F3}, Y: {normalizedVelocity.y:F3}");
            }

            previousGazeAngles = currentGazeAngles;
            timer = 0f;
        }
    }

    /// <summary>
    /// Registra la direcci�n de la mirada y obtiene los �ngulos usando la l�gica de GazeDirectionwitharea.
    /// Se obtienen las posiciones del LineRenderer, se transforma la direcci�n a coordenadas locales,
    /// y se calculan:
    /// - rawAngleH = Atan2(localDir.x, localDir.z) (convertido a grados)
    /// - rawAngleV = Atan2(localDir.y, horizontalMagnitude) (convertido a grados)
    /// Luego, se normaliza el �ngulo horizontal (rest�ndole 90� y ajust�ndolo a [-180, 180]) y se multiplica el vertical por 2 para display.
    /// Tambi�n se detecta la zona (opcional) usando colisiones con los BoxColliders asignados.
    /// </summary>
    /// <returns>Vector2 (GazeAngleX, GazeAngleY display)</returns>
    private Vector2 RecordGazeDirection()
    {
        Vector3[] positions = new Vector3[2];
        gazeRayLine.GetPositions(positions);
        if (positions[0] == positions[1])
            return previousGazeAngles; // Evita divisi�n por cero si no hay cambio

        currentDirection = (positions[1] - positions[0]).normalized;
        // Convierte la direcci�n mundial a local para que tengan sentido las zonas
        Vector3 localDir = transform.InverseTransformDirection(currentDirection);

        float rawAngleH = Mathf.Atan2(localDir.x, localDir.z) * Mathf.Rad2Deg;
        float rawAngleV = Mathf.Atan2(localDir.y, new Vector2(localDir.x, localDir.z).magnitude) * Mathf.Rad2Deg;

        // Normaliza el �ngulo horizontal (rest�ndole 90� y ajust�ndolo a [-180, 180])
        float normalizedAngleH = NormalizeAngle(rawAngleH);
        // El vertical se mantiene raw y para display se multiplica por 2
        float normalizedAngleV = rawAngleV;
        float displayAngleV = normalizedAngleV * 2f;

        // Opcional: detectar zona mediante colisiones
        string detectedZone = DetectDirectionFromColliders(positions[0], currentDirection);
        string zoneByAngle = GetDirectionFromAngles(normalizedAngleH, normalizedAngleV);
        string finalZone = detectedZone != "OutOfArea" ? detectedZone : zoneByAngle;

        Debug.Log($"GazeZone: {finalZone}, Angles: ({normalizedAngleH:F2}�, {displayAngleV:F2}�)");

        return new Vector2(normalizedAngleH, displayAngleV);
    }

    /// <summary>
    /// Ajusta el �ngulo rest�ndole 90� y limit�ndolo al rango [-180, 180].
    /// </summary>
    private float NormalizeAngle(float angle)
    {
        angle -= 90f;
        while (angle > 180f) angle -= 360f;
        while (angle < -180f) angle += 360f;
        return angle;
    }

    /// <summary>
    /// Detecta la zona de la mirada usando un SphereCast y comparando el collider impactado con las �reas asignadas.
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
    /// Determina la direcci�n de la mirada basada en rangos predefinidos de �ngulos.
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
    /// Aplica un filtro de media m�vil a las velocidades: mantiene una cola de tama�o WINDOW_SIZE,
    /// suma los valores y los promedia para suavizar las fluctuaciones.
    /// </summary>
    private Vector2 ApplyMovingAverageFilter(Queue<Vector2> velocityQueue, Vector2 newVelocity)
    {
        if (velocityQueue.Count >= WINDOW_SIZE)
            velocityQueue.Dequeue();
        velocityQueue.Enqueue(newVelocity);

        Vector2 sum = Vector2.zero;
        foreach (Vector2 vel in velocityQueue)
            sum += vel;
        return sum / velocityQueue.Count;
    }

    /// <summary>
    /// Normaliza un valor al rango [-1, 1] usando maxVelocity como referencia.
    /// </summary>
    private float NormalizeValue(float value, float maxVelocity)
    {
        return Mathf.Clamp(value / maxVelocity, -1f, 1f);
    }

    /// <summary>
    /// Exporta los datos recopilados a un archivo CSV con la cabecera:
    /// Tiempo,VelocidadGaze_X,VelocidadGaze_Y,VelocidadNormalizada_X,VelocidadNormalizada_Y
    /// </summary>
    public void GuardarDatosEnCSV()
    {
        if (tiempos.Count == 0)
        {
            Debug.LogWarning("No hay datos para guardar");
            return;
        }

        StringBuilder csv = new StringBuilder();
        csv.AppendLine("Tiempo,VelocidadGaze_X,VelocidadGaze_Y,VelocidadNormalizada_X,VelocidadNormalizada_Y");

        for (int i = 0; i < velocidades.Count; i++)
        {
            csv.AppendLine($"{tiempos[i]:F3},{velocidades[i].x:F6},{velocidades[i].y:F6}," +
                           $"{velocidadesNormalizadas[i].x:F6},{velocidadesNormalizadas[i].y:F6}");
        }

        string carpeta = @"C:\Users\Manuel Delado\Documents";
        string prefijo = "velocidad_mirada";
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

    // M�todos p�blicos para obtener la velocidad actual (opcional)
    public Vector2 GetCurrentVelocity()
    {
        return velocidades.Count > 0 ? velocidades[velocidades.Count - 1] : Vector2.zero;
    }

    public Vector2 GetCurrentNormalizedVelocity()
    {
        return velocidadesNormalizadas.Count > 0 ? velocidadesNormalizadas[velocidadesNormalizadas.Count - 1] : Vector2.zero;
    }

    // M�todo para limpiar los datos guardados (opcional)
    public void LimpiarDatos()
    {
        tiempos.Clear();
        velocidades.Clear();
        velocidadesNormalizadas.Clear();
        gazeVelocities.Clear();
        InitializeVelocityQueue();
    }
}
