using UnityEngine;
using System.Collections;
using System.IO;
using System.Text;
using System;
using System.Threading;


public class CybersicknessRecorder : MonoBehaviour
{
    [Header("Configuración")]
    [SerializeField] private float recordInterval = 0.2f;    // Intervalo de registro en segundos

    // Estado actual
    private bool isRecording = true;
    private bool sicknessState = false; // false=0, true=1
    private float recordTimer = 0f;
    private float sessionStartTime = 0f;

    // Lista interna para almacenar datos
    private class CybersicknessData
    {
        public float time;
        public int state;
    }
    private System.Collections.Generic.List<CybersicknessData> dataPoints = new System.Collections.Generic.List<CybersicknessData>();

    // UI (opcional)
    [SerializeField] private UnityEngine.UI.Text statusText;
    [SerializeField] private UnityEngine.UI.Text stateText;

    /// <summary>
    /// PROPIEDAD para DataCombiner: retorna el estado actual 0 o 1.
    /// </summary>
    public int UltimoEstadoSickness
    {
        get
        {
            // Devuelve el boolean sicknessState como int
            return sicknessState ? 1 : 0;
        }
    }

    void Start()
    {
        // Iniciar grabación automáticamente
        sessionStartTime = Time.time;
        recordTimer = 0f;
        Debug.Log("Grabación Cybersickness iniciada automáticamente.");
        UpdateUI();
    }

    void Update()
    {
        // Alternar con Espacio
        if (Input.GetKeyDown(KeyCode.Space))
        {
            sicknessState = !sicknessState;
            Debug.Log("Estado Cybersickness cambiado a: " + (sicknessState ? 1 : 0));
            UpdateUI();
        }

        // Acumular tiempo
        recordTimer += Time.deltaTime;

        // Registrar cada recordInterval
        if (recordTimer >= recordInterval)
        {
            RecordDataPoint();
            recordTimer = 0f;
        }

        // Guardar con tecla S (CSV propio)
        if (Input.GetKeyDown(KeyCode.S))
        {
            GuardarDatosEnCSV();
            Debug.Log("Datos guardados manualmente en CSV de Cybersickness.");
        }
    }

    /// <summary>
    /// Agrega un data point a la lista local (tiempo, estado).
    /// </summary>
    private void RecordDataPoint()
    {
        float currentTime = Time.time - sessionStartTime;
        int value = sicknessState ? 1 : 0;

        CybersicknessData data = new CybersicknessData();
        data.time = currentTime;
        data.state = value;
        dataPoints.Add(data);
    }

    /// <summary>
    /// Guarda la lista interna en un CSV propio.
    /// </summary>
    private void GuardarDatosEnCSV()
    {
        if (dataPoints.Count == 0)
        {
            Debug.LogWarning("No hay datos de Cybersickness para guardar.");
            return;
        }

        StringBuilder csv = new StringBuilder();
        csv.AppendLine("Time(s),Cybersickness");

        foreach (var data in dataPoints)
        {
            csv.AppendLine($"{data.time:F1},{data.state}");
        }

        string carpeta = @"C:\Users\Manuel Delado\Documents";
        string prefijo = "cybersickness_data";
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
        if (!Directory.Exists(carpeta))
        {
            try
            {
                Directory.CreateDirectory(carpeta);
            }
            catch (Exception e)
            {
                Debug.LogError("No se pudo crear la carpeta: " + e.Message);
                carpeta = Application.persistentDataPath;
            }
        }

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

    private void UpdateUI()
    {
        if (statusText != null)
        {
            statusText.text = "ESTADO: " + (sicknessState ? "1" : "0");
            statusText.color = sicknessState ? Color.yellow : Color.green;
        }
        if (stateText != null)
        {
            stateText.text = "CYBERSICKNESS";
            stateText.color = Color.white;
        }
    }

    private void OnApplicationQuit()
    {
        // Guardar al salir
        if (dataPoints.Count > 0)
        {
            GuardarDatosEnCSV();
            Debug.Log("Aplicación cerrándose. Datos Cybersickness guardados.");
        }
    }

    private void OnDisable()
    {
        // Guardar al desactivarse
        if (dataPoints.Count > 0)
        {
            GuardarDatosEnCSV();
            Debug.Log("CybersicknessRecorder deshabilitado. Datos guardados.");
        }
    }
}
