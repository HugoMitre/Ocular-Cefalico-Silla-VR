using UnityEngine;
using System.Collections.Generic;
using System.IO;
using System.Text;
using System.Threading;

public class NormalizedGazeDwellTimeCalculator : MonoBehaviour
{
    [Header("Referencia al GazeDirectionwitharea")]
    [SerializeField] private GazeDirectionwitharea referenceScript;

    [Header("Line Renderer de la Mirada")]
    [SerializeField] private LineRenderer gazeRayLine;

    [Header("Areas de Referencia (colliders opcionales)")]
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

    [Header("Configuracion de Areas y Deteccion")]
    [SerializeField] private LayerMask hitLayers;

    public float angleThreshold = 3.0f;        // Umbral angular en grados
    public float minPermanenciaTime = 0.1f;    // Tiempo minimo (segundos) para considerar permanencia

    // Listas publicas para los resultados
    public List<float> StartTimeEyePermanenceSec = new List<float>();
    public List<float> EndTimeEyePermanenceSec = new List<float>();
    public List<float> DurationEyePermanenceSec = new List<float>();
    public List<float> StartAngleEyePermanenceX = new List<float>();
    public List<float> StartAngleEyePermanenceY = new List<float>();
    public List<float> EndAngleEyePermanenceX = new List<float>();
    public List<float> EndAngleEyePermanenceY = new List<float>();
    public List<float> EyePermanenceNormalization = new List<float>();

    // Estructura interna para cada intervalo de permanencia
    private class PermanenciaData
    {
        public Vector2 startAngle;
        public Vector2 endAngle;
        public string startArea;
        public string endArea;
        public float startTime;
        public float endTime;
    }

    // Lista de permanencias detectadas
    private List<PermanenciaData> permanencias = new List<PermanenciaData>();

    // Control de la permanencia actual
    private bool enPermanencia = false;
    private Vector2 anguloInicialPermanencia;
    private string areaInicialPermanencia;
    private float tiempoInicioPermanencia;

    // Para detectar el area con colliders
    private Dictionary<BoxCollider, string> areaNames;
    private RaycastHit[] hitBuffer = new RaycastHit[10];

    void Start()
    {
        // Verificacion de LineRenderer
        if (gazeRayLine == null)
        {
            Debug.LogError("LineRenderer no asignado");
            enabled = false;
            return;
        }

        // Verificacion de referencia al GazeDirectionwitharea
        if (referenceScript == null)
        {
            Debug.LogError("No se ha asignado la referencia a GazeDirectionwitharea");
            enabled = false;
            return;
        }

        InitializeAreas();

        // Si no se configuro el LayerMask, usar "Default"
        if (hitLayers.value == 0)
        {
            hitLayers = LayerMask.GetMask("Default");
        }
    }

    private void InitializeAreas()
    {
        areaNames = new Dictionary<BoxCollider, string>();
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
    }

    private void InitializeArea(BoxCollider col, string nombre)
    {
        if (col != null)
        {
            areaNames.Add(col, nombre);
            col.isTrigger = true;
        }
    }

    void Update()
    {
        // 1) Leemos los angulos desde GazeDirectionwitharea
        float angleX = referenceScript.UltimoAnguloGazeX;
        float angleY = referenceScript.UltimoAnguloGazeY;
        Vector2 angulosActuales = new Vector2(angleX, angleY);

        // 2) Detectamos el area (usando el LineRenderer como siempre)
        Vector3[] positions = new Vector3[2];
        gazeRayLine.GetPositions(positions);
        string areaActual = DetectDirectionFromColliders(positions[0], positions[1] - positions[0]);

        // 3) Manejo de la permanencia
        if (!enPermanencia)
        {
            IniciarPermanencia(angulosActuales, areaActual);
        }
        else
        {
            float difAngulos = Vector2.Distance(angulosActuales, anguloInicialPermanencia);
            if (difAngulos > angleThreshold || areaActual != areaInicialPermanencia)
            {
                CerrarPermanencia(angulosActuales, areaActual);
                IniciarPermanencia(angulosActuales, areaActual);
            }
        }
    }

    private void IniciarPermanencia(Vector2 angulos, string area)
    {
        enPermanencia = true;
        anguloInicialPermanencia = angulos;
        areaInicialPermanencia = area;
        tiempoInicioPermanencia = Time.time;
    }

    private void CerrarPermanencia(Vector2 angulosFinales, string areaFinal)
    {
        enPermanencia = false;
        float tiempoFin = Time.time;
        float duracion = tiempoFin - tiempoInicioPermanencia;

        if (duracion >= minPermanenciaTime)
        {
            PermanenciaData p = new PermanenciaData();
            p.startAngle = anguloInicialPermanencia;
            p.endAngle = angulosFinales;
            p.startArea = areaInicialPermanencia;
            p.endArea = areaFinal;
            p.startTime = tiempoInicioPermanencia;
            p.endTime = tiempoFin;
            permanencias.Add(p);

            // Calculamos la normalizacion
            float duracionMs = duracion * 1000f;
            float normalizado = NormalizarTiempoYAplicarDireccion(
                duracionMs, anguloInicialPermanencia.x, angulosFinales.x);

            // Guardamos en las listas publicas
            StartTimeEyePermanenceSec.Add(p.startTime);
            EndTimeEyePermanenceSec.Add(p.endTime);
            DurationEyePermanenceSec.Add(duracion);
            StartAngleEyePermanenceX.Add(p.startAngle.x);
            StartAngleEyePermanenceY.Add(p.startAngle.y);
            EndAngleEyePermanenceX.Add(p.endAngle.x);
            EndAngleEyePermanenceY.Add(p.endAngle.y);
            EyePermanenceNormalization.Add(normalizado);

            Debug.Log("Permanencia registrada: dur=" + duracion.ToString("F2") + "s, angles=" +
                      p.startAngle.x.ToString("F2") + "->" + p.endAngle.x.ToString("F2"));
        }
        else
        {
            Debug.Log("Permanencia descartada por ser muy corta (" + duracion.ToString("F3") + "s)");
        }
    }

    private string DetectDirectionFromColliders(Vector3 origin, Vector3 direction)
    {
        if (direction.sqrMagnitude > 0.0001f)
            direction = direction.normalized;
        else
            return "OutOfArea";

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

    private void OnDisable()
    {
        // Si aun hay una permanencia activa, la cerramos
        if (enPermanencia)
        {
            Vector3[] positions = new Vector3[2];
            gazeRayLine.GetPositions(positions);

            // Nuevamente, tomamos los angulos de referenceScript
            float angleX = referenceScript.UltimoAnguloGazeX;
            float angleY = referenceScript.UltimoAnguloGazeY;
            Vector2 angulosActuales = new Vector2(angleX, angleY);

            string areaActual = DetectDirectionFromColliders(positions[0], positions[1] - positions[0]);
            CerrarPermanencia(angulosActuales, areaActual);
        }

        GuardarPermanenciasEnCSV();
    }

    private void GuardarPermanenciasEnCSV()
    {
        if (permanencias.Count == 0)
        {
            Debug.LogWarning("No hay permanencias para guardar");
            return;
        }

        StringBuilder csv = new StringBuilder();
        csv.AppendLine("StartTime(s),EndTime(s),Duration(s),StartAngleX,StartAngleY,EndAngleX,EndAngleY,Normalization");

        foreach (var p in permanencias)
        {
            float duracionSeg = p.endTime - p.startTime;
            float duracionMs = duracionSeg * 1000f;

            float normalizado = NormalizarTiempoYAplicarDireccion(
                duracionMs, p.startAngle.x, p.endAngle.x);

            csv.AppendLine(
                p.startTime.ToString("F3") + "," +
                p.endTime.ToString("F3") + "," +
                duracionSeg.ToString("F3") + "," +
                p.startAngle.x.ToString("F2") + "," + p.startAngle.y.ToString("F2") + "," +
                p.endAngle.x.ToString("F2") + "," + p.endAngle.y.ToString("F2") + "," +
                normalizado.ToString("F2")
            );
        }

        string carpeta = @"C:\Users\Manuel Delado\Documents";
        string prefijo = "threshold_based_permanencias";
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
                Debug.Log("Datos guardados en: " + rutaArchivo);
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
            rutaArchivo = Path.Combine(carpeta, prefijo + "_" + fechaHora + extension);
            File.WriteAllText(rutaArchivo, csv.ToString());
            Debug.Log("Datos guardados con timestamp en: " + rutaArchivo);
        }
    }

    private string ObtenerSiguienteNombreArchivo(string carpeta, string prefijo, string extension)
    {
        int numero = 1;
        string nombreArchivo;
        do
        {
            nombreArchivo = Path.Combine(carpeta, prefijo + numero + extension);
            numero++;
        }
        while (File.Exists(nombreArchivo));

        return nombreArchivo;
    }

    // Normalizacion de tiempo
    private float NormalizarTiempo(float tiempoSegundos)
    {
        if (tiempoSegundos <= 0.15f)
            return 0f;

        if (tiempoSegundos <= 0.30f)
        {
            if (tiempoSegundos <= 0.18f)
                return Mathf.Lerp(0.0f, 0.2f, (tiempoSegundos - 0.15f) / (0.18f - 0.15f));
            else if (tiempoSegundos <= 0.20f)
                return Mathf.Lerp(0.2f, 0.3f, (tiempoSegundos - 0.18f) / (0.20f - 0.18f));
            else if (tiempoSegundos <= 0.25f)
                return Mathf.Lerp(0.3f, 0.45f, (tiempoSegundos - 0.20f) / (0.25f - 0.20f));
            else if (tiempoSegundos <= 0.28f)
                return Mathf.Lerp(0.45f, 0.55f, (tiempoSegundos - 0.25f) / (0.28f - 0.25f));
            else
                return Mathf.Lerp(0.55f, 0.6f, (tiempoSegundos - 0.28f) / (0.30f - 0.28f));
        }

        if (tiempoSegundos <= 0.80f)
        {
            if (tiempoSegundos <= 0.40f)
                return Mathf.Lerp(0.6f, 0.7f, (tiempoSegundos - 0.30f) / (0.40f - 0.30f));
            else if (tiempoSegundos <= 0.50f)
                return Mathf.Lerp(0.7f, 0.8f, (tiempoSegundos - 0.40f) / (0.50f - 0.40f));
            else if (tiempoSegundos <= 0.60f)
                return Mathf.Lerp(0.8f, 0.85f, (tiempoSegundos - 0.50f) / (0.60f - 0.50f));
            else if (tiempoSegundos <= 0.70f)
                return Mathf.Lerp(0.85f, 0.9f, (tiempoSegundos - 0.60f) / (0.70f - 0.60f));
            else
                return Mathf.Lerp(0.9f, 1.0f, (tiempoSegundos - 0.70f) / (0.80f - 0.70f));
        }

        return 1.0f;
    }

    /// <summary>
    /// Aplica la normalizacion del tiempo y asigna signo segun (anguloFinalX - anguloInicialX).
    /// </summary>
    private float NormalizarTiempoYAplicarDireccion(float tiempoMs, float anguloInicialX, float anguloFinalX)
    {
        float seg = tiempoMs / 1000f;
        float baseNorm = NormalizarTiempo(seg);
        float diff = anguloFinalX - anguloInicialX;
        float sign = (diff >= 0f) ? 1f : -1f;
        return sign * baseNorm;
    }
}

