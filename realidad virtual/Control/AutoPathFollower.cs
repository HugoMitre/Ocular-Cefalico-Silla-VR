using UnityEngine;
using System.Collections;

public class SimpleConstantMovement : MonoBehaviour
{
    public Transform sillaDeRuedas;       // Transform de la silla
    public Transform xrRig;               // Transform del XR Rig
    public CustomCurve rutaIdeal;         // Tu CustomCurve

    [Header("Movimiento")]
    public float velocidad = 0.7f;        // Velocidad constante
    public float tiempoInicio = 1f;       // Tiempo antes de iniciar
    public bool iniciarAlAwake = true;    // Iniciar automáticamente
    public bool movimientoActivado = false; // Estado del movimiento

    [Header("Rotación")]
    public float anguloOffset = 90f;      // Ángulo perpendicular
    public float suavizadoRotacion = 5f;  // Suavizado rotación

    // Variables privadas
    private float tiempoActual = 0f;      // Tiempo transcurrido
    private Vector3[] puntosRuta;         // Puntos de la ruta
    private float duracionRuta;           // Duración estimada del recorrido
    private bool rutaPreparada = false;   // Si la ruta está lista

    void Start()
    {
        // Referencias
        if (sillaDeRuedas == null) sillaDeRuedas = transform;

        // Preparar la ruta
        if (rutaIdeal != null)
        {
            PrepararRuta();
        }

        // Iniciar automáticamente
        if (iniciarAlAwake)
        {
            Invoke("IniciarMovimiento", tiempoInicio);
        }
    }

    private bool PrepararRuta()
    {
        // Verificar la ruta
        if (rutaIdeal == null || rutaIdeal.pathRenderer == null)
        {
            Debug.LogError("Ruta invalida");
            return false;
        }

        int numPuntos = rutaIdeal.pathRenderer.positionCount;
        if (numPuntos < 2)
        {
            Debug.LogError("La ruta necesita al menos 2 puntos");
            return false;
        }

        // Copiar los puntos
        puntosRuta = new Vector3[numPuntos];
        rutaIdeal.pathRenderer.GetPositions(puntosRuta);

        // Calcular duración aproximada
        float distanciaTotal = 0f;
        for (int i = 0; i < numPuntos - 1; i++)
        {
            distanciaTotal += Vector3.Distance(puntosRuta[i], puntosRuta[i + 1]);
        }

        duracionRuta = distanciaTotal / velocidad;
        Debug.Log($"Ruta preparada. Distancia: {distanciaTotal}, Duración: {duracionRuta}");

        rutaPreparada = true;
        return true;
    }

    public void IniciarMovimiento()
    {
        // Verificar ruta
        if (!rutaPreparada && !PrepararRuta())
        {
            Debug.LogError("No se pudo iniciar el movimiento");
            return;
        }

        // Colocar en posición inicial
        tiempoActual = 0f;
        sillaDeRuedas.position = puntosRuta[0];

        // Iniciar
        movimientoActivado = true;
        Debug.Log("Movimiento iniciado");
    }

    public void DetenerMovimiento()
    {
        movimientoActivado = false;
    }

    public void ReiniciarMovimiento()
    {
        tiempoActual = 0f;
        movimientoActivado = true;
    }

    void Update()
    {
        if (!movimientoActivado || !rutaPreparada) return;

        // Actualizar tiempo
        tiempoActual += Time.deltaTime;

        // Calcular progreso (0-1)
        float progreso = tiempoActual / duracionRuta;

        // Si llegamos al final
        if (progreso >= 1.0f)
        {
            sillaDeRuedas.position = puntosRuta[puntosRuta.Length - 1];
            movimientoActivado = false;
            Debug.Log("Movimiento completado");
            return;
        }

        // Calcular posición
        Vector3 nuevaPosicion = EvaluarRuta(progreso);

        // Calcular dirección
        float progresoAdelantado = Mathf.Min(progreso + 0.01f, 1.0f);
        Vector3 puntoAdelante = EvaluarRuta(progresoAdelantado);
        Vector3 direccion = (puntoAdelante - nuevaPosicion).normalized;

        // Aplicar posición
        sillaDeRuedas.position = nuevaPosicion;

        // Aplicar rotación
        if (direccion.magnitude > 0.001f)
        {
            // Aplanar dirección (ignorar Y)
            Vector3 direccionHorizontal = direccion;
            direccionHorizontal.y = 0;

            if (direccionHorizontal.magnitude > 0.001f)
            {
                // Rotación base + offset
                Quaternion rotBase = Quaternion.LookRotation(direccionHorizontal);
                Quaternion rotOffset = Quaternion.Euler(0, anguloOffset, 0);
                Quaternion rotFinal = rotBase * rotOffset;

                // Aplicar con suavizado
                sillaDeRuedas.rotation = Quaternion.Slerp(
                    sillaDeRuedas.rotation,
                    rotFinal,
                    suavizadoRotacion * Time.deltaTime
                );
            }
        }
    }

    // Método simplificado para evaluar la ruta en un punto t (0-1)
    private Vector3 EvaluarRuta(float t)
    {
        // Asegurar que t está entre 0 y 1
        t = Mathf.Clamp01(t);

        // Convertir t a índice en la curva
        float puntoExacto = t * (puntosRuta.Length - 1);
        int indiceInferior = Mathf.FloorToInt(puntoExacto);
        float fraccion = puntoExacto - indiceInferior;

        // Límites de seguridad
        indiceInferior = Mathf.Clamp(indiceInferior, 0, puntosRuta.Length - 2);

        // Interpolar linealmente entre puntos
        return Vector3.Lerp(
            puntosRuta[indiceInferior],
            puntosRuta[indiceInferior + 1],
            fraccion
        );
    }
}