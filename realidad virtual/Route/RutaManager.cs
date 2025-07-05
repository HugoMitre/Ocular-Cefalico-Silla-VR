using UnityEngine;
using System.Collections.Generic;
using System.IO;
using System.Text;
using System.Linq;

public class RutaManager : MonoBehaviour
{
    [Header("Referencias")]
    public List<CustomCurve> rutasIdeales;
    public LineRenderer rutaReal;
    public Transform sillaDeRuedas;

    // Propiedades públicas para DataCombiner
    public Vector3 UltimaPosicion
    {
        get { return registros.Count > 0 ? registros[registros.Count - 1].posicionReal : Vector3.zero; }
    }

    public float UltimaDesviacion
    {
        get { return registros.Count > 0 ? registros[registros.Count - 1].desviacion : 0f; }
    }

    public float UltimoPuntoIdealX
    {
        get { return registros.Count > 0 ? registros[registros.Count - 1].posicionIdeal.x : 0f; }
    }

    public float UltimoPuntoIdealZ
    {
        get { return registros.Count > 0 ? registros[registros.Count - 1].posicionIdeal.z : 0f; }
    }

    private List<DatosRegistro> registros = new List<DatosRegistro>();
    private float tiempoUltimoRegistro = 0f;
    private float intervaloRegistro = 0.1f;

    public class DatosRegistro
    {
        public float tiempo;
        public Vector3 posicionReal;
        public Vector3 posicionIdeal;
        public float desviacion;

        public DatosRegistro(float t, Vector3 real, Vector3 ideal)
        {
            tiempo = t;
            posicionReal = real;
            posicionIdeal = ideal;
            desviacion = Vector3.Distance(real, ideal);
        }
    }

    void Start()
    {
        SetupLineRenderer();
    }

    void SetupLineRenderer()
    {
        if (rutaReal != null)
        {
            rutaReal.startWidth = 0.05f;
            rutaReal.endWidth = 0.05f;
            rutaReal.material = new Material(Shader.Find("Sprites/Default"));
            rutaReal.startColor = Color.blue;
            rutaReal.endColor = Color.blue;
            rutaReal.useWorldSpace = true;
        }
    }

    void Update()
    {
        if (Time.time - tiempoUltimoRegistro >= intervaloRegistro)
        {
            RegistrarPosicion();
            tiempoUltimoRegistro = Time.time;
        }
        ActualizarVisualizacion();
    }

    void RegistrarPosicion()
    {
        Vector3 posReal = sillaDeRuedas.position;
        Vector3 posIdeal = ObtenerPuntoIdealMasCercano(posReal);

        var registro = new DatosRegistro(Time.time, posReal, posIdeal);
        registros.Add(registro);

        if (Debug.isDebugBuild)
        {
            Debug.Log($"Tiempo: {registro.tiempo:F3} - " +
                     $"Real(X,Y,Z): ({registro.posicionReal.x:F3}, {registro.posicionReal.y:F3}, {registro.posicionReal.z:F3}) - " +
                     $"Ideal(X,Z): ({registro.posicionIdeal.x:F3}, {registro.posicionIdeal.z:F3}) - " +
                     $"Desviación: {registro.desviacion:F3}");
        }
    }

    Vector3 ObtenerPuntoIdealMasCercano(Vector3 posicionActual)
    {
        float distanciaMinima = float.MaxValue;
        Vector3 puntoMasCercano = Vector3.zero;

        foreach (var curva in rutasIdeales)
        {
            Vector3 puntoEnCurva = curva.GetNearestPoint(posicionActual);
            float distancia = Vector3.Distance(posicionActual, puntoEnCurva);

            if (distancia < distanciaMinima)
            {
                distanciaMinima = distancia;
                puntoMasCercano = puntoEnCurva;
            }
        }

        return puntoMasCercano;
    }

    void ActualizarVisualizacion()
    {
        if (registros.Count > 0)
        {
            rutaReal.positionCount = registros.Count;
            Vector3[] posiciones = new Vector3[registros.Count];
            for (int i = 0; i < registros.Count; i++)
            {
                posiciones[i] = registros[i].posicionReal;
            }
            rutaReal.SetPositions(posiciones);
        }
    }

    public void GuardarDatosEnCSV()
    {
        if (registros.Count == 0)
        {
            Debug.LogWarning("No hay datos para guardar");
            return;
        }

        try
        {
            StringBuilder csv = new StringBuilder();
            csv.AppendLine("Tiempo,PosRealX,PosRealY,PosRealZ,PuntoIdealX,PuntoIdealZ,Desviacion");

            foreach (var registro in registros)
            {
                csv.AppendLine($"{registro.tiempo:F3},{registro.posicionReal.x:F6},{registro.posicionReal.y:F6}," +
                             $"{registro.posicionReal.z:F6},{registro.posicionIdeal.x:F6},{registro.posicionIdeal.z:F6}," +
                             $"{registro.desviacion:F6}");
            }

            string carpeta = @"C:\Users\Manuel Delado\Documents";
            string prefijo = "ruta_datos";
            string extension = ".csv";
            SaveFileWithRetry(carpeta, prefijo, extension, csv.ToString());
        }
        catch (System.Exception e)
        {
            Debug.LogError($"Error al guardar datos: {e.Message}");
        }
    }

    private void SaveFileWithRetry(string carpeta, string prefijo, string extension, string content)
    {
        bool archivoGuardado = false;
        int intentos = 0;
        string rutaArchivo = "";

        while (!archivoGuardado && intentos < 5)
        {
            try
            {
                rutaArchivo = ObtenerSiguienteNombreArchivo(carpeta, prefijo, extension);
                File.WriteAllText(rutaArchivo, content);
                archivoGuardado = true;
                Debug.Log($"Datos guardados exitosamente en: {rutaArchivo}");
            }
            catch (IOException)
            {
                intentos++;
                System.Threading.Thread.Sleep(100);
            }
        }

        if (!archivoGuardado)
        {
            string fechaHora = System.DateTime.Now.ToString("yyyyMMdd_HHmmss");
            rutaArchivo = Path.Combine(carpeta, $"{prefijo}_{fechaHora}{extension}");
            File.WriteAllText(rutaArchivo, content);
            Debug.Log($"Datos guardados con timestamp en: {rutaArchivo}");
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
        } while (File.Exists(nombreArchivo));
        return nombreArchivo;
    }

    void OnDisable()
    {
        GuardarDatosEnCSV();
    }

    public void LimpiarRegistros()
    {
        registros.Clear();
        rutaReal.positionCount = 0;
    }

    public void GuardarRegistrosManualmente()
    {
        GuardarDatosEnCSV();
    }

    public DatosRegistro GetUltimoRegistro()
    {
        return registros.Count > 0 ? registros[registros.Count - 1] : null;
    }
}