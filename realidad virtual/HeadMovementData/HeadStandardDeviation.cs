using System.Collections.Generic;
using System.IO;
using System.Text;
using System.Threading;
using UnityEngine;

public class HeadStandardDeviation : MonoBehaviour
{
    // Propiedades públicas para el DataCombiner
    public Vector2 UltimaDesviacion
    {
        get { return desviacionesEstandar.Count > 0 ? desviacionesEstandar[desviacionesEstandar.Count - 1] : Vector2.zero; }
    }

    public Vector2 UltimaDesviacionNormalizada
    {
        get { return desviacionesNormalizadas.Count > 0 ? desviacionesNormalizadas[desviacionesNormalizadas.Count - 1] : Vector2.zero; }
    }

    [Header("Reference Settings")]
    public GameObject frontReference; // Objeto de referencia frontal (igual que en HeadDirectionTracker)

    // Variables para capturar ángulos "reales" usando el método de HeadDirectionTracker
    private Vector2 currentHeadAngles = Vector2.zero;
    private Vector2 previousHeadAngles = Vector2.zero;

    private float deltaTime = 0.2f;
    private float timer = 0f;

    // Cola para almacenar los últimos 10 ángulos
    private Queue<Vector2> headAnglesQueue = new Queue<Vector2>();
    private const int QUEUE_SIZE = 10;

    // Valor máximo para normalización de la desviación (en grados)
    private const float MAX_DEVIATION = 25f;

    private List<float> tiempos = new List<float>();
    private List<Vector2> desviacionesEstandar = new List<Vector2>();
    private List<Vector2> desviacionesNormalizadas = new List<Vector2>();

    private Camera mainCamera;

    void Start()
    {
        Debug.Log("Sistema de cálculo de desviación estándar de cabeza iniciado");

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

        // Inicializamos la cola con valores iniciales (por defecto, 0°)
        for (int i = 0; i < QUEUE_SIZE; i++)
        {
            headAnglesQueue.Enqueue(Vector2.zero);
        }

        // Se toma la primera muestra usando el método de obtención de ángulos
        previousHeadAngles = GetHeadAngles();
    }

    void Update()
    {
        timer += Time.deltaTime;

        if (timer >= deltaTime)
        {
            // Actualizamos los ángulos usando el método de HeadDirectionTracker
            UpdateHeadAngles();
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

                Debug.Log($"Head StdDev: {stdDev}, Normalized: {normalizedStdDev}");
            }

            previousHeadAngles = currentHeadAngles;
            timer = 0f;
        }
    }

    /// <summary>
    /// Obtiene los ángulos de la cabeza usando la metodología de HeadDirectionTracker.
    /// Se calcula el ángulo horizontal (comparando la proyección en XZ) y
    /// el ángulo vertical (multiplicado por 2 para display) respecto a un objeto de referencia frontal.
    /// </summary>
    Vector2 GetHeadAngles()
    {
        // Dirección desde la cámara hasta el objeto de referencia
        Vector3 toReference = (frontReference.transform.position - mainCamera.transform.position).normalized;
        Vector3 lookDirection = mainCamera.transform.forward;

        // Cálculo del ángulo horizontal (en el plano XZ)
        Vector3 horizontalToReference = new Vector3(toReference.x, 0, toReference.z).normalized;
        Vector3 horizontalLook = new Vector3(lookDirection.x, 0, lookDirection.z).normalized;
        float horizontalAngle = Vector3.SignedAngle(horizontalToReference, horizontalLook, Vector3.up);
        float clampedHorizontal = ClampAngle(horizontalAngle);

        // Cálculo del ángulo vertical
        float verticalAngle = Vector3.SignedAngle(lookDirection, horizontalLook, mainCamera.transform.right);
        float clampedVertical = ClampAngle(verticalAngle);
        float displayVerticalAngle = clampedVertical * 2f; // Para display, igual que en HeadDirectionTracker

        return new Vector2(clampedHorizontal, displayVerticalAngle);
    }

    /// <summary>
    /// Actualiza la muestra de ángulos y la agrega a la cola.
    /// </summary>
    void UpdateHeadAngles()
    {
        currentHeadAngles = GetHeadAngles();
        if (headAnglesQueue.Count >= QUEUE_SIZE)
            headAnglesQueue.Dequeue();
        headAnglesQueue.Enqueue(currentHeadAngles);
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
    /// Calcula la desviación estándar de los ángulos en la cola.
    /// Se calcula la media, la varianza (ajustando diferencias mayores a 180°) y se saca la raíz cuadrada.
    /// Además, se aplica un signo basado en la dirección acumulada del movimiento.
    /// </summary>
    Vector2 CalculateStandardDeviation()
    {
        if (headAnglesQueue.Count == 0) return Vector2.zero;

        // Cálculo de la media
        Vector2 mean = Vector2.zero;
        foreach (Vector2 angle in headAnglesQueue)
        {
            mean += angle;
        }
        mean /= headAnglesQueue.Count;

        Vector2 variance = Vector2.zero;
        Vector2 direction = Vector2.zero; // Para determinar el signo de la desviación

        foreach (Vector2 angle in headAnglesQueue)
        {
            Vector2 diff = angle - mean;
            // Ajuste para diferencias mayores a 180° (debido a la naturaleza circular)
            if (Mathf.Abs(diff.x) > 180f)
                diff.x = 360f - Mathf.Abs(diff.x);
            if (Mathf.Abs(diff.y) > 180f)
                diff.y = 360f - Mathf.Abs(diff.y);

            variance.x += diff.x * diff.x;
            variance.y += diff.y * diff.y;
            direction += (angle - previousHeadAngles);
        }

        if (headAnglesQueue.Count > 1)
            variance /= (headAnglesQueue.Count - 1);

        // Calcula la desviación estándar y asigna el signo según la dirección del movimiento
        return new Vector2(
            -Mathf.Sign(direction.x) * Mathf.Sqrt(variance.x),
             Mathf.Sign(direction.y) * Mathf.Sqrt(variance.y)
        );
    }

    /// <summary>
    /// Normaliza un valor al rango [-1, 1] usando MAX_DEVIATION como referencia.
    /// </summary>
    float NormalizeValue(float value)
    {
        return Mathf.Clamp(value / MAX_DEVIATION, -1f, 1f);
    }

    /// <summary>
    /// Exporta los datos registrados a un archivo CSV con la cabecera:
    /// Tiempo,DesviacionEstandar_X,DesviacionEstandar_Y,DesviacionNormalizada_X,DesviacionNormalizada_Y
    /// </summary>
    public void GuardarDatosEnCSV()
    {
        if (tiempos.Count == 0)
        {
            Debug.LogWarning("No hay datos para guardar");
            return;
        }

        StringBuilder csv = new StringBuilder();
        csv.AppendLine("Tiempo,DesviacionEstandar_X,DesviacionEstandar_Y,DesviacionNormalizada_X,DesviacionNormalizada_Y");

        for (int i = 0; i < desviacionesEstandar.Count; i++)
        {
            csv.AppendLine($"{tiempos[i]:F3},{desviacionesEstandar[i].x:F6},{desviacionesEstandar[i].y:F6}," +
                           $"{desviacionesNormalizadas[i].x:F6},{desviacionesNormalizadas[i].y:F6}");
        }

        string carpeta = @"C:\Users\Manuel Delado\Documents";
        string prefijo = "desviacion_estandar_head";
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

    string ObtenerSiguienteNombreArchivo(string carpeta, string prefijo, string extension)
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
