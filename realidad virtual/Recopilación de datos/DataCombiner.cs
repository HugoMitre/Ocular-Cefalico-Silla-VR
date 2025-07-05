using UnityEngine;
using System;
using System.Collections.Generic;
using System.IO;
using System.Text;

public class DataCombiner : MonoBehaviour
{
    [SerializeField]
    private GameObject cameraOffset;
    [SerializeField]
    private GameObject rutaObject;
    [SerializeField]
    private GameObject gazeRayObject;
    [SerializeField]
    private GameObject wheelchair;
    [SerializeField]
    private GameObject gameManager;

    private AngularVelocityCalculator velocidadScript;
    private HeadStandardDeviation desviacionScript;
    private NormalizedDwellTimeCalculator permanenciaScript;
    private RutaManager rutaScript;
    private NormalizedGazeDwellTimeCalculator gazeScript;
    private GazeStandardDeviation gazeStdDevScript;
    private GazeDataLogger gazeVelocityScript;
    private MyMessageListener wheelchairScript;
    private HeadDirectionTracker headDirectionScript;
    private GazeDirectionwitharea gazeDirectionAreaScript;

    private StringBuilder csvData = new StringBuilder();
    private float deltaTime = 0.2f;
    private float timer = 0f;
    private bool headerWritten = false;
    private string tareaActual = "";
    private string comandoActual = "";

    private int participanteActual;
    private int intentoActual;
    private bool dataStarted = false;

    void Awake()
    {
        participanteActual = PlayerPrefs.GetInt("ParticipanteActual", 1);
        intentoActual = PlayerPrefs.GetInt("IntentoActual", 1);

        Debug.Log($"Iniciando grabación para Participante {participanteActual}, Intento {intentoActual}");

        if (PlayerPrefs.HasKey("ParticipanteActual"))
        {
            Debug.Log("Configuración encontrada en PlayerPrefs");
        }
        else
        {
            Debug.LogWarning("No se encontró configuración previa. Usando valores por defecto.");
        }
    }

    void Start()
    {
        if (cameraOffset != null && rutaObject != null && gazeRayObject != null && wheelchair != null && gameManager != null)
        {
            velocidadScript = cameraOffset.GetComponent<AngularVelocityCalculator>();
            desviacionScript = cameraOffset.GetComponent<HeadStandardDeviation>();
            permanenciaScript = cameraOffset.GetComponent<NormalizedDwellTimeCalculator>();
            rutaScript = rutaObject.GetComponent<RutaManager>();
            gazeScript = gazeRayObject.GetComponent<NormalizedGazeDwellTimeCalculator>();
            gazeStdDevScript = gazeRayObject.GetComponent<GazeStandardDeviation>();
            gazeVelocityScript = gazeRayObject.GetComponent<GazeDataLogger>();
            wheelchairScript = wheelchair.GetComponent<MyMessageListener>();
            headDirectionScript = gameManager.GetComponent<HeadDirectionTracker>();
            gazeDirectionAreaScript = gameManager.GetComponent<GazeDirectionwitharea>();

            Debug.Log($"Camera Offset encontrado: {cameraOffset.name}");
            Debug.Log($"Velocidad Script encontrado: {velocidadScript != null}");
            Debug.Log($"Desviación Script encontrado: {desviacionScript != null}");
            Debug.Log($"Permanencia Script encontrado: {permanenciaScript != null}");
            Debug.Log($"Ruta Script encontrado: {rutaScript != null}");
            Debug.Log($"Gaze Script encontrado: {gazeScript != null}");
            Debug.Log($"Gaze Standard Deviation encontrado: {gazeStdDevScript != null}");
            Debug.Log($"Gaze Velocity Script encontrado: {gazeVelocityScript != null}");
            Debug.Log($"Wheelchair Script encontrado: {wheelchairScript != null}");
            Debug.Log($"Head Direction Script encontrado: {headDirectionScript != null}");
            Debug.Log($"Gaze Direction Area Script encontrado: {gazeDirectionAreaScript != null}");

            csvData.Clear();
            csvData.AppendLine("Time,Task,Command," +
                "HeadVelocityX,HeadVelocityY,HeadNormalizedVelocityX,HeadNormalizedVelocityY," +
                "HeadDeviationX,HeadDeviationY,HeadNormalizedDeviationX,HeadNormalizedDeviationY," +
                "HeadDwellX,HeadDwellY,HeadNormalizedDwellX,HeadNormalizedDwellY," +
                "HeadHorizontalAngle,HeadVerticalAngle,HeadDirection," +
                "GazeDwellX,GazeDwellY,GazeDwellNormalizedX,GazeDwellNormalizedY," +
                "GazeDeviationX,GazeDeviationY,GazeDeviationNormalizedX,GazeDeviationNormalizedY," +
                "GazeVelocityX,GazeVelocityY,GazeVelocityNormalizedX,GazeVelocityNormalizedY," +
                "GazeAngleX,GazeAngleY,GazeDirection," +
                "UserPositionX,UserPositionY,UserPositionZ,IdealPositionX,IdealPositionZ,PathDeviation," +
                "Input_X_Normalized,Input_Y_Normalized,Angle,Magnitude,Position_X,Position_Z,Rotation_Y,Action");

            headerWritten = true;
            dataStarted = true;
            Debug.Log("Header CSV creado correctamente");
        }
        else
        {
            Debug.LogError("¡Faltan asignar objetos necesarios!");
        }
    }
    void Update()
    {
        if (!dataStarted) return;

        timer += Time.deltaTime;

        if (timer >= deltaTime)
        {
            CollectData();
            timer = 0f;
        }

        if (Input.GetKeyDown(KeyCode.D))
        {
            Debug.Log($"=== Debug Data ===");
            Debug.Log($"Cantidad de datos recolectados: {csvData.Length} bytes");
            string[] lines = csvData.ToString().Split('\n');
            if (lines.Length > 1)
            {
                Debug.Log($"Primera línea (header): {lines[0]}");
                Debug.Log($"Segunda línea (datos): {lines[1]}");
            }
            Debug.Log($"Total de líneas: {lines.Length}");
        }

        if (Input.GetKeyDown(KeyCode.S))
        {
            Debug.Log("Tecla S presionada - Iniciando guardado manual");
            GuardarDatosEnCSV();
        }
    }

    void CollectData()
    {
        if (!dataStarted) return;

        StringBuilder line = new StringBuilder();
        line.Append(Time.time.ToString("F3"));
        line.Append($",{tareaActual},{comandoActual}");

        // Datos de velocidad de cabeza
        if (velocidadScript != null)
        {
            Vector2 vel = velocidadScript.UltimaVelocidad;
            Vector2 velNorm = velocidadScript.UltimaVelocidadNormalizada;
            line.Append($",{vel.x:F6},{vel.y:F6}");
            line.Append($",{velNorm.x:F6},{velNorm.y:F6}");
        }
        else
        {
            line.Append(",0,0,0,0");
        }

        // Datos de desviación de cabeza
        if (desviacionScript != null)
        {
            Vector2 dev = desviacionScript.UltimaDesviacion;
            Vector2 devNorm = desviacionScript.UltimaDesviacionNormalizada;
            line.Append($",{dev.x:F6},{dev.y:F6}");
            line.Append($",{devNorm.x:F6},{devNorm.y:F6}");
        }
        else
        {
            line.Append(",0,0,0,0");
        }

        // Datos de permanencia de cabeza
        if (permanenciaScript != null)
        {
            Vector2 perm = permanenciaScript.UltimaPermanencia;
            Vector2 permNorm = permanenciaScript.UltimaPermanenciaNormalizada;
            line.Append($",{perm.x:F6},{perm.y:F6}");
            line.Append($",{permNorm.x:F6},{permNorm.y:F6}");
        }
        else
        {
            line.Append(",0,0,0,0");
        }

        // Datos de dirección de cabeza
        if (headDirectionScript != null)
        {
            float horizontalAngle = headDirectionScript.UltimoAnguloHorizontal;
            float verticalAngle = headDirectionScript.UltimoAnguloVertical;
            string direction = headDirectionScript.UltimaDireccion;
            line.Append($",{horizontalAngle:F6},{verticalAngle:F6},{direction}");
        }
        else
        {
            line.Append(",0,0,Front");
        }

        // Datos de permanencia de mirada
        if (gazeScript != null)
        {
            Vector2 gazeP = gazeScript.UltimaPermanenciaGaze;
            Vector2 gazePNorm = gazeScript.UltimaPermanenciaGazeNormalizada;
            line.Append($",{gazeP.x:F6},{gazeP.y:F6}");
            line.Append($",{gazePNorm.x:F6},{gazePNorm.y:F6}");
        }
        else
        {
            line.Append(",0,0,0,0");
        }// Datos de desviación de mirada
        if (gazeStdDevScript != null)
        {
            Vector2 gazeStdDev = gazeStdDevScript.UltimaDesviacionGaze;
            Vector2 gazeStdDevNorm = gazeStdDevScript.UltimaDesviacionGazeNormalizada;
            line.Append($",{gazeStdDev.x:F6},{gazeStdDev.y:F6}");
            line.Append($",{gazeStdDevNorm.x:F6},{gazeStdDevNorm.y:F6}");
        }
        else
        {
            line.Append(",0,0,0,0");
        }

        // Datos de velocidad de mirada
        if (gazeVelocityScript != null)
        {
            Vector2 gazeVel = gazeVelocityScript.UltimaVelocidadGaze;
            Vector2 gazeVelNorm = gazeVelocityScript.UltimaVelocidadGazeNormalizada;
            line.Append($",{gazeVel.x:F6},{gazeVel.y:F6}");
            line.Append($",{gazeVelNorm.x:F6},{gazeVelNorm.y:F6}");
        }
        else
        {
            line.Append(",0,0,0,0");
        }

        // Datos de dirección de mirada por áreas
        if (gazeDirectionAreaScript != null)
        {
            float angleX = gazeDirectionAreaScript.UltimoAnguloGazeX;
            float angleY = gazeDirectionAreaScript.UltimoAnguloGazeY;
            string direction = gazeDirectionAreaScript.UltimaDireccionGaze;
            line.Append($",{angleX:F6},{angleY:F6},{direction}");
        }
        else
        {
            line.Append(",0,0,OutOfArea");
        }

        // Datos de ruta y posición
        if (rutaScript != null)
        {
            Vector3 pos = rutaScript.UltimaPosicion;
            Vector3 idealPos = new Vector3(rutaScript.UltimoPuntoIdealX, 0, rutaScript.UltimoPuntoIdealZ);
            float desv = rutaScript.UltimaDesviacion;
            line.Append($",{pos.x:F6},{pos.y:F6},{pos.z:F6}");
            line.Append($",{idealPos.x:F6},{idealPos.z:F6}");
            line.Append($",{desv:F6}");
        }
        else
        {
            line.Append(",0,0,0,0,0,0");
        }

        // Datos del joystick y wheelchair
        if (wheelchairScript != null)
        {
            Vector2 input = wheelchairScript.UltimoInput;
            Vector3 pos = wheelchairScript.UltimaPosicionSilla;
            float rot = wheelchairScript.UltimaRotacionSilla;
            float angle = Mathf.Atan2(input.y, input.x) * Mathf.Rad2Deg;
            float magnitude = input.magnitude;

            line.Append($",{input.x:F6},{input.y:F6}");
            line.Append($",{angle:F6},{magnitude:F6}");
            line.Append($",{pos.x:F6},{pos.z:F6}");
            line.Append($",{rot:F6}");
            line.Append($",{wheelchairScript.UltimaAccion}");
        }
        else
        {
            line.Append(",0,0,0,0,0,0,0,None");
        }

        csvData.AppendLine(line.ToString());
    }
    public void ActualizarTareaComando(string tarea, string comando)
    {
        tareaActual = tarea;
        comandoActual = comando;
        Debug.Log($"Actualizando tarea: {tarea}, comando: {comando}");
    }

    void GuardarDatosEnCSV()
    {
        if (csvData.Length == 0)
        {
            Debug.LogWarning("No hay datos para guardar");
            return;
        }

        try
        {
            participanteActual = PlayerPrefs.GetInt("ParticipanteActual", 1);
            intentoActual = PlayerPrefs.GetInt("IntentoActual", 1);

            string carpetaBase = @"C:\Users\Manuel Delado\Documents\VR_Study";
            string carpetaParticipante = Path.Combine(carpetaBase, $"Participante_{participanteActual}");
            string carpetaIntento = Path.Combine(carpetaParticipante, $"Intento_{intentoActual}");

            if (!Directory.Exists(carpetaBase)) Directory.CreateDirectory(carpetaBase);
            if (!Directory.Exists(carpetaParticipante)) Directory.CreateDirectory(carpetaParticipante);
            if (!Directory.Exists(carpetaIntento)) Directory.CreateDirectory(carpetaIntento);

            string fechaHora = DateTime.Now.ToString("yyyyMMdd_HHmmss");
            string nombreArchivo = $"P{participanteActual}_I{intentoActual}_{fechaHora}.csv";
            string rutaCompleta = Path.Combine(carpetaIntento, nombreArchivo);

            File.WriteAllText(rutaCompleta, csvData.ToString());

            Debug.Log($"=== Guardado de CSV Exitoso ===\n" +
                     $"Participante: {participanteActual}\n" +
                     $"Intento: {intentoActual}\n" +
                     $"Ruta: {rutaCompleta}\n" +
                     $"Tamaño: {csvData.Length} bytes");
        }
        catch (Exception e)
        {
            Debug.LogError($"Error al guardar archivo: {e.Message}\nStackTrace: {e.StackTrace}");
        }
    }

    void OnDisable()
    {
        if (dataStarted) GuardarDatosEnCSV();
    }

    void OnDestroy()
    {
        if (dataStarted) GuardarDatosEnCSV();
    }
}