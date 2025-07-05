using UnityEngine;
using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using WebSocketSharp;
using System.Linq;

public class MyMessageListener : MonoBehaviour
{
    // Propiedades para el DataCombiner
    public Vector2 UltimoInput
    {
        get { return inputsJoystick.Count > 0 ? inputsJoystick.Last() : Vector2.zero; }
    }

    public Vector3 UltimaPosicionSilla
    {
        get
        {
            return wheelchair != null ?
                new Vector3(
                    posicionX.Count > 0 ? posicionX.Last() : 0f,
                    0f,
                    posicionZ.Count > 0 ? posicionZ.Last() : 0f
                ) : Vector3.zero;
        }
    }

    public float UltimaRotacionSilla
    {
        get { return rotacionY.Count > 0 ? rotacionY.Last() : 0f; }
    }

    public string UltimaAccion
    {
        get { return acciones.Count > 0 ? acciones.Last() : "None"; }
    }

    [Header("Referencias de Objetos")]
    public GameObject wheelchair;
    public Camera mainCamera;

    [Header("Configuración WebSocket")]
    public string websocketURL = "ws://localhost:8080";
    public float reconnectInterval = 2f;

    [Header("Configuración de Movimiento")]
    public float movementSpeed = 5.0f;
    public float rotationSpeed = 100.0f;
    public float deadzone = 0.1f;

    [Header("Configuración de Ejes")]
    public bool invertX = false;
    public bool invertY = false;

    [Header("Diagnóstico")]
    public bool enableDebugLogging = true;
    public bool showGizmos = true;

    [Header("Configuración de Registro")]
    public string carpetaGuardado = @"C:\Users\Manuel Delado\Documents";
    public string prefijoArchivo = "datos_joystick";

    // Buffers de datos con capacidad fija
    private const int DATA_BUFFER_SIZE = 1000;
    private readonly Queue<float> tiempos = new Queue<float>(DATA_BUFFER_SIZE);
    private readonly Queue<Vector2> inputsJoystick = new Queue<Vector2>(DATA_BUFFER_SIZE);
    private readonly Queue<float> angulos = new Queue<float>(DATA_BUFFER_SIZE);
    private readonly Queue<float> magnitudes = new Queue<float>(DATA_BUFFER_SIZE);
    private readonly Queue<float> posicionX = new Queue<float>(DATA_BUFFER_SIZE);
    private readonly Queue<float> posicionZ = new Queue<float>(DATA_BUFFER_SIZE);
    private readonly Queue<float> rotacionY = new Queue<float>(DATA_BUFFER_SIZE);
    private readonly Queue<string> acciones = new Queue<string>(DATA_BUFFER_SIZE);

    // Variables de WebSocket y control
    private WebSocket websocket;
    private Vector2 currentInput;
    private bool isConnected;
    private const float MESSAGE_TIMEOUT = 3.0f;
    private float lastMessageTime;
    private float lastReconnectAttempt;
    private float tiempoInicio;

    // Buffer para suavizado de movimiento
    private Vector2[] inputBuffer = new Vector2[3];
    private int bufferIndex = 0;

    // Intervalo de recolección de datos
    private float intervaloRecoleccion = 0.2f;
    private float ultimoTiempoRecoleccion = 0f;


    void Start()
    {
        if (!ValidateComponents()) return;
        tiempoInicio = Time.time;
        InitializeDebugLog();
        InitializeWebSocket();
    }

    bool ValidateComponents()
    {
        if (!wheelchair)
        {
            Debug.LogError("¡Error: No se ha asignado el GameObject de la silla de ruedas!");
            enabled = false;
            return false;
        }

        if (!mainCamera)
        {
            mainCamera = Camera.main;
            if (!mainCamera)
            {
                Debug.LogError("¡Error: No se encontró ninguna cámara!");
                enabled = false;
                return false;
            }
        }

        if (!UnityMainThreadDispatcher.Instance())
        {
            Debug.LogError("¡Error: No se encontró UnityMainThreadDispatcher en la escena!");
            enabled = false;
            return false;
        }

        return true;
    }

    void InitializeWebSocket()
    {
        try
        {
            if (websocket != null)
            {
                websocket.Close();
                websocket = null;
            }

            websocket = new WebSocket(websocketURL);

            websocket.OnMessage += (sender, e) => {
                UnityMainThreadDispatcher.Instance().Enqueue(() => {
                    ProcessWebSocketMessage(e.Data);
                });
            };

            websocket.OnOpen += (sender, e) => {
                UnityMainThreadDispatcher.Instance().Enqueue(() => {
                    isConnected = true;
                    Debug.Log("WebSocket conectado");
                    lastMessageTime = Time.time;
                });
            };

            websocket.OnClose += (sender, e) => {
                UnityMainThreadDispatcher.Instance().Enqueue(() => {
                    isConnected = false;
                    Debug.Log("WebSocket desconectado");
                });
            };

            websocket.OnError += (sender, e) => {
                UnityMainThreadDispatcher.Instance().Enqueue(() => {
                    Debug.LogError($"Error WebSocket: {e.Message}");
                    isConnected = false;
                });
            };

            websocket.Connect();
            lastReconnectAttempt = Time.time;
        }
        catch (Exception e)
        {
            Debug.LogError($"Error al inicializar WebSocket: {e.Message}");
            isConnected = false;
        }
    }

    void ProcessWebSocketMessage(string message)
    {
        try
        {
            string[] data = message.Split(',');
            if (data.Length == 2)
            {
                float angle = float.Parse(data[0]);
                float magnitude = float.Parse(data[1]) / 100f;

                if (invertX) angle = 360 - angle;
                if (invertY) magnitude *= -1;

                float rad = angle * Mathf.Deg2Rad;
                Vector2 input = new Vector2(
                    Mathf.Cos(rad) * magnitude,
                    Mathf.Sin(rad) * magnitude
                );

                // Actualizar buffer para suavizado
                inputBuffer[bufferIndex] = input;
                bufferIndex = (bufferIndex + 1) % inputBuffer.Length;

                // Calcular promedio para suavizar el movimiento
                currentInput = Vector2.zero;
                for (int i = 0; i < inputBuffer.Length; i++)
                {
                    currentInput += inputBuffer[i];
                }
                currentInput /= inputBuffer.Length;

                lastMessageTime = Time.time;
            }
        }
        catch (Exception e)
        {
            Debug.LogWarning($"Error procesando mensaje: {e.Message}");
        }
    }

    void Update()
    {
        // Verificar el estado de la conexión
        CheckConnectionStatus();

        // Aplicar movimiento si está sobre el deadzone
        if (currentInput.magnitude > deadzone && isConnected)
        {
            ApplyMovement();
        }

        // Recolección de datos en intervalos
        if (Time.time - ultimoTiempoRecoleccion >= intervaloRecoleccion)
        {
            RegistrarDatosEnIntervalo(currentInput);
            ultimoTiempoRecoleccion = Time.time;
        }
    }

    void CheckConnectionStatus()
    {
        if (!isConnected || Time.time - lastMessageTime > MESSAGE_TIMEOUT)
        {
            if (Time.time - lastReconnectAttempt >= reconnectInterval)
            {
                Debug.Log("Intentando reconectar...");
                InitializeWebSocket();
            }

            if (currentInput.magnitude > 0)
            {
                currentInput = Vector2.zero;
                for (int i = 0; i < inputBuffer.Length; i++)
                {
                    inputBuffer[i] = Vector2.zero;
                }
            }
        }
    }

    void ApplyMovement()
    {
        // Rotación
        float rotationAmount = currentInput.x * rotationSpeed * Time.deltaTime;
        wheelchair.transform.Rotate(0, -rotationAmount, 0);

        // Movimiento
        float moveAmount = currentInput.y * movementSpeed * Time.deltaTime;
        Vector3 movement = wheelchair.transform.right * (-moveAmount);
        wheelchair.transform.position += movement;
    }

    void RegistrarDatosEnIntervalo(Vector2 input)
    {
        if (!wheelchair) return;

        // Mantener el buffer en un tamaño fijo
        if (tiempos.Count >= DATA_BUFFER_SIZE)
        {
            tiempos.Dequeue();
            inputsJoystick.Dequeue();
            angulos.Dequeue();
            magnitudes.Dequeue();
            posicionX.Dequeue();
            posicionZ.Dequeue();
            rotacionY.Dequeue();
            acciones.Dequeue();
        }

        tiempos.Enqueue(Time.time - tiempoInicio);
        inputsJoystick.Enqueue(input);
        angulos.Enqueue(Mathf.Atan2(input.y, input.x) * Mathf.Rad2Deg);
        magnitudes.Enqueue(input.magnitude);
        posicionX.Enqueue(wheelchair.transform.position.x);
        posicionZ.Enqueue(wheelchair.transform.position.z);
        rotacionY.Enqueue(wheelchair.transform.eulerAngles.y);
        acciones.Enqueue(DeterminarAccion(input));
    }

    string DeterminarAccion(Vector2 input)
    {
        if (input.magnitude <= deadzone) return "Detenido";

        List<string> accionesActuales = new List<string>();

        if (input.y > deadzone) accionesActuales.Add("Avanzando");
        else if (input.y < -deadzone) accionesActuales.Add("Retrocediendo");

        if (input.x > deadzone) accionesActuales.Add("Girando Izquierda");
        else if (input.x < -deadzone) accionesActuales.Add("Girando Derecha");

        return string.Join(" + ", accionesActuales);
    }

    public void GuardarDatosEnCSV()
    {
        StringBuilder csv = new StringBuilder();
        csv.AppendLine("Time,Input_X_Normalized,Input_Y_Normalized,Angle,Magnitude,Position_X,Position_Z,Rotation_Y,Action");

        var tiemposArray = tiempos.ToArray();
        var inputsArray = inputsJoystick.ToArray();
        var angulosArray = angulos.ToArray();
        var magnitudesArray = magnitudes.ToArray();
        var posicionXArray = posicionX.ToArray();
        var posicionZArray = posicionZ.ToArray();
        var rotacionYArray = rotacionY.ToArray();
        var accionesArray = acciones.ToArray();

        for (int i = 0; i < tiempos.Count; i++)
        {
            float inputXInverted = -inputsArray[i].x;

            csv.AppendLine(
                $"{tiemposArray[i]:F3}," +
                $"{inputXInverted:F6},{inputsArray[i].y:F6}," +
                $"{angulosArray[i]:F2},{magnitudesArray[i]:F2}," +
                $"{posicionXArray[i]:F6},{posicionZArray[i]:F6}," +
                $"{rotacionYArray[i]:F2}," +
                $"{accionesArray[i]}"
            );
        }

        GuardarArchivo(csv.ToString());
    }

    void GuardarArchivo(string contenido)
    {
        try
        {
            string timestamp = DateTime.Now.ToString("yyyyMMdd_HHmmss");
            string rutaArchivo = Path.Combine(carpetaGuardado, $"{prefijoArchivo}_{timestamp}.csv");
            File.WriteAllText(rutaArchivo, contenido);
            Debug.Log($"Datos guardados exitosamente en: {rutaArchivo}");
        }
        catch (Exception e)
        {
            Debug.LogError($"Error al guardar archivo: {e.Message}");
        }
    }

    void InitializeDebugLog()
    {
        try
        {
            string logDirectory = Path.Combine(Application.persistentDataPath, "Logs");
            if (!Directory.Exists(logDirectory))
            {
                Directory.CreateDirectory(logDirectory);
            }
            Debug.Log($"Log directory initialized at: {logDirectory}");
        }
        catch (Exception e)
        {
            Debug.LogError($"Error initializing log directory: {e.Message}");
            enableDebugLogging = false;
        }
    }

    void OnDrawGizmos()
    {
        if (!showGizmos || !wheelchair) return;

        Gizmos.color = Color.blue;
        Vector3 position = wheelchair.transform.position;
        Gizmos.DrawLine(position, position + wheelchair.transform.forward * 2);

        if (currentInput.magnitude > deadzone)
        {
            Gizmos.color = Color.green;
            Vector3 inputDirection = new Vector3(currentInput.x, 0, currentInput.y);
            Gizmos.DrawLine(position, position + inputDirection * 2);
        }
    }

    void OnDisable()
    {
        if (websocket != null)
        {
            if (websocket.ReadyState == WebSocketState.Open)
            {
                websocket.Close();
            }
            websocket = null;
        }
        GuardarDatosEnCSV();
    }

    void OnApplicationQuit()
    {
        GuardarDatosEnCSV();
    }
}