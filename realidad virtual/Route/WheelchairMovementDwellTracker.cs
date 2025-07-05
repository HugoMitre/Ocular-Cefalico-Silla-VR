using UnityEngine;
using System.Collections.Generic;
using System.Text;
using System.IO;
using System.Threading;

public class WheelchairMovementDwellTracker : MonoBehaviour
{
    [Header("Configuración de Detección")]
    [Tooltip("Distancia mínima (en unidades de Unity) para considerar que la silla ha cambiado lo suficiente.")]
    public float positionThreshold = 0.5f;

    [Tooltip("Tiempo mínimo (en segundos) para registrar un intervalo de permanencia.")]
    public float minDwellTime = 1f;

    // Estructura para almacenar cada intervalo (inicio, fin, posiciones).
    [System.Serializable]
    public class DwellInterval
    {
        public float startTime;
        public float endTime;
        public Vector3 startPosition;
        public Vector3 endPosition;
    }

    // Lista pública: se pueden ver los intervalos en Inspector o usarlos externamente
    public List<DwellInterval> dwellIntervals = new List<DwellInterval>();

    // === PROPIEDAD Pública para DataCombiner ===
    // Devuelve la duración (segundos) del último intervalo registrado, o 0 si ninguno existe.
    public float DurationChairPermanence
    {
        get
        {
            if (dwellIntervals.Count > 0)
            {
                var ultimo = dwellIntervals[dwellIntervals.Count - 1];
                return ultimo.endTime - ultimo.startTime;
            }
            return 0f;
        }
    }

    // Control de intervalo actual
    private Vector3 referencePosition;
    private float referenceTime;

    void Start()
    {
        referencePosition = transform.position;
        referenceTime = Time.time;
    }

    void Update()
    {
        Vector3 currentPosition = transform.position;
        float distance = Vector3.Distance(currentPosition, referencePosition);

        // Si se mueve más del umbral, se cierra el intervalo
        if (distance > positionThreshold)
        {
            float intervalTime = Time.time - referenceTime;
            // Solo registramos si el tiempo supera el mínimo
            if (intervalTime >= minDwellTime)
            {
                DwellInterval interval = new DwellInterval
                {
                    startTime = referenceTime,
                    endTime = Time.time,
                    startPosition = referencePosition,
                    endPosition = currentPosition
                };
                dwellIntervals.Add(interval);

                Debug.Log($"Intervalo de silla registrado: {intervalTime:F2}s (de {referencePosition} a {currentPosition})");
            }
            // Reiniciamos la referencia para el siguiente intervalo
            referencePosition = currentPosition;
            referenceTime = Time.time;
        }
    }

    /// <summary>
    /// Guarda los intervalos en un CSV separado (opcional, se llama al desactivar).
    /// </summary>
    public void GuardarDatosEnCSV()
    {
        if (dwellIntervals.Count == 0)
        {
            Debug.LogWarning("No hay datos de permanencia de silla para guardar.");
            return;
        }

        StringBuilder csv = new StringBuilder();
        csv.AppendLine("StartTime,EndTime,Duration,StartPosX,StartPosY,StartPosZ,EndPosX,EndPosY,EndPosZ");

        for (int i = 0; i < dwellIntervals.Count; i++)
        {
            var interval = dwellIntervals[i];
            float duration = interval.endTime - interval.startTime;
            csv.AppendLine(
                $"{interval.startTime:F3}," +
                $"{interval.endTime:F3}," +
                $"{duration:F3}," +
                $"{interval.startPosition.x:F2}," +
                $"{interval.startPosition.y:F2}," +
                $"{interval.startPosition.z:F2}," +
                $"{interval.endPosition.x:F2}," +
                $"{interval.endPosition.y:F2}," +
                $"{interval.endPosition.z:F2}"
            );
        }

        string carpeta = @"C:\Users\Manuel Delado\Documents";
        string prefijo = "wheelchair_dwell";
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
                Debug.Log($"Datos de silla guardados en: {rutaArchivo}");
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
            Debug.Log($"Datos de silla guardados con timestamp en: {rutaArchivo}");
        }
    }

    private string ObtenerSiguienteNombreArchivo(string carpeta, string prefijo, string extension)
    {
        int numero = 1;
        string nombreArchivo;
        do
        {
            nombreArchivo = Path.Combine(carpeta, $"{prefijo}{numero}{extension}");
            numero++;
        }
        while (File.Exists(nombreArchivo));
        return nombreArchivo;
    }

    // Al desactivar, guardamos un CSV específico de la silla (opcional).
    private void OnDisable()
    {
        GuardarDatosEnCSV();
    }
}
