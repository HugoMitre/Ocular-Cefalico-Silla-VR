using UnityEngine;
using System.Collections.Generic;
using System.IO;
using System.Text;
using System.Threading;

public class NormalizedHeadDwellTimeCalculator : MonoBehaviour
{
    [Header("Configuracion de Deteccion")]
    [Tooltip("Umbral angular (en grados) para considerar que la direccion cambio lo suficiente para cerrar el intervalo.")]
    public float angleThreshold = 3.0f;

    [Tooltip("Tiempo minimo de permanencia (en segundos) para registrar un intervalo.")]
    public float minPermanenciaTime = 0.2f; // 200 ms

    [Header("Referencia para Angulos")]
    [Tooltip("Componente HeadDirectionTracker para obtener los angulos de la cabeza.")]
    [SerializeField] private HeadDirectionTracker headTracker;

    // Listas publicas para almacenar datos de cada intervalo
    public List<float> StartTimeHeadPermanenceSec = new List<float>();
    public List<float> EndTimeHeadPermanenceSec = new List<float>();
    public List<float> DurationHeadPermanenceSec = new List<float>();
    public List<float> StartAngleHeadPermanenceX = new List<float>();
    public List<float> StartAngleHeadPermanenceY = new List<float>();
    public List<float> EndAngleHeadPermanenceX = new List<float>();
    public List<float> EndAngleHeadPermanenceY = new List<float>();
    public List<float> HeadPermanenceNormalization = new List<float>();

    // Propiedades publicas para acceder al ultimo intervalo
    public float UltimaStartTimeHeadPermanenceSec
    {
        get
        {
            if (StartTimeHeadPermanenceSec.Count > 0)
                return StartTimeHeadPermanenceSec[StartTimeHeadPermanenceSec.Count - 1];
            return 0f;
        }
    }

    public float UltimaEndTimeHeadPermanenceSec
    {
        get
        {
            if (EndTimeHeadPermanenceSec.Count > 0)
                return EndTimeHeadPermanenceSec[EndTimeHeadPermanenceSec.Count - 1];
            return 0f;
        }
    }

    public float UltimaDurationHeadPermanenceSec
    {
        get
        {
            if (DurationHeadPermanenceSec.Count > 0)
                return DurationHeadPermanenceSec[DurationHeadPermanenceSec.Count - 1];
            return 0f;
        }
    }

    public float UltimaStartAngleHeadPermanenceX
    {
        get
        {
            if (StartAngleHeadPermanenceX.Count > 0)
                return StartAngleHeadPermanenceX[StartAngleHeadPermanenceX.Count - 1];
            return 0f;
        }
    }

    public float UltimaStartAngleHeadPermanenceY
    {
        get
        {
            if (StartAngleHeadPermanenceY.Count > 0)
                return StartAngleHeadPermanenceY[StartAngleHeadPermanenceY.Count - 1];
            return 0f;
        }
    }

    public float UltimaEndAngleHeadPermanenceX
    {
        get
        {
            if (EndAngleHeadPermanenceX.Count > 0)
                return EndAngleHeadPermanenceX[EndAngleHeadPermanenceX.Count - 1];
            return 0f;
        }
    }

    public float UltimaEndAngleHeadPermanenceY
    {
        get
        {
            if (EndAngleHeadPermanenceY.Count > 0)
                return EndAngleHeadPermanenceY[EndAngleHeadPermanenceY.Count - 1];
            return 0f;
        }
    }

    public float UltimaHeadPermanenceNormalization
    {
        get
        {
            if (HeadPermanenceNormalization.Count > 0)
                return HeadPermanenceNormalization[HeadPermanenceNormalization.Count - 1];
            return 0f;
        }
    }

    // Estructura interna para cada intervalo
    private class HeadPermanenceData
    {
        public Vector2 startAngle;
        public Vector2 endAngle;
        public float startTime;
        public float endTime;
    }

    private List<HeadPermanenceData> permanencias = new List<HeadPermanenceData>();

    // Control de la permanencia actual
    private bool enPermanencia = false;
    private Vector2 anguloInicialPermanencia;
    private float tiempoInicioPermanencia;

    void Start()
    {
        // Verificamos que headTracker este asignado
        if (headTracker == null)
        {
            Debug.LogError("HeadTracker no esta asignado!");
            enabled = false;
        }
    }

    void Update()
    {
        // 1) Leer los angulos del HeadDirectionTracker
        float angleX = headTracker.UltimoAnguloHorizontal;
        float angleY = headTracker.UltimoAnguloVertical;
        Vector2 angulosActuales = new Vector2(angleX, angleY);

        // 2) Manejo de permanencia
        if (!enPermanencia)
        {
            IniciarPermanencia(angulosActuales);
        }
        else
        {
            float difAngulos = Vector2.Distance(angulosActuales, anguloInicialPermanencia);
            if (difAngulos > angleThreshold)
            {
                CerrarPermanencia(angulosActuales);
                IniciarPermanencia(angulosActuales);
            }
        }
    }

    private void IniciarPermanencia(Vector2 angulos)
    {
        enPermanencia = true;
        anguloInicialPermanencia = angulos;
        tiempoInicioPermanencia = Time.time;
    }

    private void CerrarPermanencia(Vector2 angulosFinales)
    {
        enPermanencia = false;
        float tiempoFin = Time.time;
        float duracion = tiempoFin - tiempoInicioPermanencia;

        if (duracion >= minPermanenciaTime)
        {
            HeadPermanenceData p = new HeadPermanenceData
            {
                startAngle = anguloInicialPermanencia,
                endAngle = angulosFinales,
                startTime = tiempoInicioPermanencia,
                endTime = tiempoFin
            };
            permanencias.Add(p);

            // Normalizacion del tiempo en ms
            float duracionMs = duracion * 1000f;
            float normalizado = NormalizarTiempoCabeza(duracionMs);

            // Guardamos en las listas publicas
            StartTimeHeadPermanenceSec.Add(p.startTime);
            EndTimeHeadPermanenceSec.Add(p.endTime);
            DurationHeadPermanenceSec.Add(duracion);
            StartAngleHeadPermanenceX.Add(p.startAngle.x);
            StartAngleHeadPermanenceY.Add(p.startAngle.y);
            EndAngleHeadPermanenceX.Add(p.endAngle.x);
            EndAngleHeadPermanenceY.Add(p.endAngle.y);
            HeadPermanenceNormalization.Add(normalizado);

            Debug.Log("Head Permanence registrado: Duracion=" + duracion.ToString("F2")
                      + "s, Normalizado=" + normalizado.ToString("F2"));
        }
        else
        {
            Debug.Log("Head Permanence descartado por ser muy corto (" + duracion.ToString("F3") + "s).");
        }
    }

    private float NormalizarTiempoCabeza(float ms)
    {
        if (ms < 200f) return 0f;
        if (ms >= 1200f) return 1f;

        if (ms <= 250f) return Mathf.Lerp(0.0f, 0.1f, (ms - 200f) / (250f - 200f));
        if (ms <= 300f) return Mathf.Lerp(0.1f, 0.2f, (ms - 250f) / (300f - 250f));
        if (ms <= 350f) return Mathf.Lerp(0.2f, 0.3f, (ms - 300f) / (350f - 300f));
        if (ms <= 400f) return Mathf.Lerp(0.3f, 0.4f, (ms - 350f) / (400f - 350f));
        if (ms <= 450f) return Mathf.Lerp(0.4f, 0.5f, (ms - 400f) / (450f - 400f));
        if (ms <= 500f) return Mathf.Lerp(0.5f, 0.6f, (ms - 450f) / (500f - 450f));

        if (ms <= 600f) return Mathf.Lerp(0.6f, 0.65f, (ms - 500f) / (600f - 500f));
        if (ms <= 700f) return Mathf.Lerp(0.65f, 0.7f, (ms - 600f) / (700f - 600f));
        if (ms <= 800f) return Mathf.Lerp(0.7f, 0.75f, (ms - 700f) / (800f - 700f));
        if (ms <= 900f) return Mathf.Lerp(0.75f, 0.8f, (ms - 800f) / (900f - 800f));
        if (ms <= 1000f) return Mathf.Lerp(0.8f, 0.9f, (ms - 900f) / (1000f - 900f));
        if (ms <= 1100f) return Mathf.Lerp(0.9f, 0.95f, (ms - 1000f) / (1100f - 1000f));
        return Mathf.Lerp(0.95f, 1.0f, (ms - 1100f) / (1200f - 1100f));
    }

    public void GuardarDatosEnCSV()
    {
        if (StartTimeHeadPermanenceSec.Count == 0)
        {
            Debug.LogWarning("No hay datos para guardar");
            return;
        }

        StringBuilder csv = new StringBuilder();
        csv.AppendLine("StartTimeHeadPermanenceSec,EndTimeHeadPermanenceSec,DurationHeadPermanenceSec,"
                       + "StartAngleHeadPermanenceX,StartAngleHeadPermanenceY,"
                       + "EndAngleHeadPermanenceX,EndAngleHeadPermanenceY,HeadPermanenceNormalization");

        for (int i = 0; i < StartTimeHeadPermanenceSec.Count; i++)
        {
            csv.AppendLine(
                StartTimeHeadPermanenceSec[i].ToString("F3") + ","
                + EndTimeHeadPermanenceSec[i].ToString("F3") + ","
                + DurationHeadPermanenceSec[i].ToString("F3") + ","
                + StartAngleHeadPermanenceX[i].ToString("F2") + ","
                + StartAngleHeadPermanenceY[i].ToString("F2") + ","
                + EndAngleHeadPermanenceX[i].ToString("F2") + ","
                + EndAngleHeadPermanenceY[i].ToString("F2") + ","
                + HeadPermanenceNormalization[i].ToString("F2")
            );
        }

        string carpeta = @"C:\Users\Manuel Delado\Documents";
        string prefijo = "permanencia_cabeza";
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
                Debug.Log("Datos guardados exitosamente en: " + rutaArchivo);
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

    private void OnDisable()
    {
        if (enPermanencia)
        {
            float angleX = headTracker.UltimoAnguloHorizontal;
            float angleY = headTracker.UltimoAnguloVertical;
            Vector2 angulosActuales = new Vector2(angleX, angleY);
            CerrarPermanencia(angulosActuales);
        }
        GuardarDatosEnCSV();
    }
}
