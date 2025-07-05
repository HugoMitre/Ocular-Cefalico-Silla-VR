using System.Collections.Generic;
using System.IO;
using System.Text;
using UnityEngine;

public class AngularVelocityCalculator : MonoBehaviour
{
    // Propiedades públicas para el DataCombiner
    public Vector2 UltimaVelocidad
    {
        get { return velocidades.Count > 0 ? velocidades[velocidades.Count - 1] : Vector2.zero; }
    }

    public Vector2 UltimaVelocidadNormalizada
    {
        get { return velocidadesNormalizadas.Count > 0 ? velocidadesNormalizadas[velocidadesNormalizadas.Count - 1] : Vector2.zero; }
    }

    [Header("Reference Settings")]
    public GameObject frontReference;

    [Header("Sampling Settings")]
    [Range(0.1f, 1.0f)]
    public float samplingInterval = 0.2f;

    [Header("Angular Velocity Normalization")]
    [Tooltip("Velocidad angular máxima esperada para el eje horizontal (°/s)")]
    public float maxAngularVelocityHorizontal = 90f;
    [Tooltip("Velocidad angular máxima esperada para el eje vertical (°/s)")]
    public float maxAngularVelocityVertical = 90f;

    // Variables para almacenar muestras y tiempos
    private float lastSampleTime;
    private Vector2 previousAngles = Vector2.zero;
    private Vector2 currentAngles = Vector2.zero;

    private List<float> tiempos = new List<float>();
    private List<Vector2> velocidades = new List<Vector2>();
    private List<Vector2> velocidadesNormalizadas = new List<Vector2>();

    // Filtro de media móvil con cola
    private Queue<Vector2> velocityQueue = new Queue<Vector2>();
    private const int QUEUE_SIZE = 5;

    private Camera mainCamera;

    void Start()
    {
        if (frontReference == null)
        {
            Debug.LogError("¡Objeto de referencia frontal no asignado!");
            enabled = false;
            return;
        }

        mainCamera = Camera.main;
        if (mainCamera == null)
        {
            Debug.LogError("¡No se encontró la cámara principal!");
            enabled = false;
            return;
        }

        lastSampleTime = Time.time;

        // Inicializamos la cola para el filtro de media móvil
        for (int i = 0; i < QUEUE_SIZE; i++)
        {
            velocityQueue.Enqueue(Vector2.zero);
        }

        // Tomamos la primera muestra de ángulos
        previousAngles = GetHeadAngles();
    }

    void Update()
    {
        if (Time.time - lastSampleTime >= samplingInterval)
        {
            lastSampleTime = Time.time;
            // Obtenemos los ángulos usando el mismo método que en HeadDirectionTracker
            currentAngles = GetHeadAngles();

            // Calculamos la velocidad angular: diferencia de ángulos / intervalo de muestreo
            Vector2 rawVelocity = (currentAngles - previousAngles) / samplingInterval;

            // Aplicamos un filtro de media móvil para suavizar la lectura
            Vector2 filteredVelocity = ApplyMovingAverageFilter(velocityQueue, rawVelocity);

            // Normalizamos la velocidad a rango [-1, 1] usando los máximos configurados
            Vector2 normalizedVelocity = new Vector2(
                Mathf.Clamp(filteredVelocity.x / maxAngularVelocityHorizontal, -1f, 1f),
                Mathf.Clamp(filteredVelocity.y / maxAngularVelocityVertical, -1f, 1f)
            );

            // Almacenamos los datos para análisis y exportación
            tiempos.Add(Time.time);
            velocidades.Add(filteredVelocity);
            velocidadesNormalizadas.Add(normalizedVelocity);

            Debug.Log($"Tiempo: {Time.time:F3}s, Velocidad: {filteredVelocity}, Normalizada: {normalizedVelocity}");

            previousAngles = currentAngles;
        }
    }

    /// <summary>
    /// Calcula los ángulos horizontales y verticales usando el mismo método del HeadDirectionTracker.
    /// - Para el horizontal se compara la proyección en XZ de la dirección de la cámara y la referencia frontal.
    /// - Para el vertical se calcula el ángulo entre la dirección de la cámara y su proyección horizontal,
    ///   se limita a ±90° y se multiplica por 2 para display.
    /// </summary>
    Vector2 GetHeadAngles()
    {
        Vector3 toReference = (frontReference.transform.position - mainCamera.transform.position).normalized;
        Vector3 lookDirection = mainCamera.transform.forward;

        // Cálculo del ángulo horizontal
        Vector3 horizontalToReference = new Vector3(toReference.x, 0, toReference.z).normalized;
        Vector3 horizontalLook = new Vector3(lookDirection.x, 0, lookDirection.z).normalized;
        float horizontalAngle = Vector3.SignedAngle(horizontalToReference, horizontalLook, Vector3.up);
        float clampedHorizontal = ClampAngle(horizontalAngle);

        // Cálculo del ángulo vertical
        float verticalAngle = Vector3.SignedAngle(lookDirection, horizontalLook, mainCamera.transform.right);
        float clampedVertical = ClampAngle(verticalAngle);
        float displayVerticalAngle = clampedVertical * 2f;

        return new Vector2(clampedHorizontal, displayVerticalAngle);
    }

    /// <summary>
    /// Limita un ángulo al rango de -90° a 90°.
    /// </summary>
    float ClampAngle(float angle)
    {
        if (angle > 90f) return 90f;
        if (angle < -90f) return -90f;
        return angle;
    }

    /// <summary>
    /// Aplica un filtro de media móvil usando una cola de tamaño fijo.
    /// </summary>
    Vector2 ApplyMovingAverageFilter(Queue<Vector2> queue, Vector2 newSample)
    {
        if (queue.Count >= QUEUE_SIZE)
            queue.Dequeue();
        queue.Enqueue(newSample);

        Vector2 sum = Vector2.zero;
        foreach (Vector2 sample in queue)
        {
            sum += sample;
        }
        return sum / queue.Count;
    }

    /// <summary>
    /// Exporta los datos recopilados a un archivo CSV con el siguiente formato:
    /// Tiempo,VelocidadAngular_X,VelocidadAngular_Y,VelocidadNormalizada_X,VelocidadNormalizada_Y
    /// </summary>
    public void GuardarDatosEnCSV()
    {
        if (tiempos.Count == 0)
        {
            Debug.LogWarning("No hay datos para guardar");
            return;
        }

        StringBuilder csv = new StringBuilder();
        csv.AppendLine("Tiempo,VelocidadAngular_X,VelocidadAngular_Y,VelocidadNormalizada_X,VelocidadNormalizada_Y");

        for (int i = 0; i < tiempos.Count; i++)
        {
            csv.AppendLine($"{tiempos[i]:F3},{velocidades[i].x:F6},{velocidades[i].y:F6},{velocidadesNormalizadas[i].x:F6},{velocidadesNormalizadas[i].y:F6}");
        }

        string folder = Path.Combine(Application.persistentDataPath, "AngularVelocityData");
        if (!Directory.Exists(folder))
            Directory.CreateDirectory(folder);

        string filename = ObtenerSiguienteNombreArchivo(folder, "angular_velocity", ".csv");

        try
        {
            File.WriteAllText(filename, csv.ToString());
            Debug.Log($"Datos guardados en: {filename}");
        }
        catch (System.Exception e)
        {
            Debug.LogError($"Error al guardar el archivo: {e.Message}");
        }
    }

    string ObtenerSiguienteNombreArchivo(string folder, string prefix, string extension)
    {
        int count = 1;
        string filename;
        do
        {
            filename = Path.Combine(folder, $"{prefix}{count}{extension}");
            count++;
        }
        while (File.Exists(filename));
        return filename;
    }

    void OnDisable()
    {
        GuardarDatosEnCSV();
    }
}
