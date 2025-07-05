using UnityEngine;

public class zonas_visual : MonoBehaviour
{
    // Variables para configurar en el Inspector
    public string taskName;          // Nombre de la tarea para esta zona
    public string command;           // Comando para esta zona
    public string zoneID;            // Identificador único de la zona
    public bool esZonaVerde = false; // Marcar si es zona verde
    public bool esZonaRoja = false;  // Marcar si es zona roja

    [SerializeField]
    private DataCombiner dataCombiner; // Referencia al objeto que maneja el CSV

    void Start()
    {
        // Verificar que se haya asignado el DataCombiner
        if (dataCombiner == null)
        {
            Debug.LogError($"Zona {zoneID}: No se asignó el DataCombiner. Por favor, arrastra el objeto con DataCombiner al campo en el Inspector.");
        }
    }

    private void OnTriggerEnter(Collider other)
    {
        // Verificar si es la silla de ruedas
        if (other.CompareTag("silla de ruedas"))
        {
            // Si es zona verde, actualizar el CSV
            if (esZonaVerde)
            {
                dataCombiner.ActualizarTareaComando(taskName, command);
                Debug.Log($"Zona {zoneID}: Actualizando tarea {taskName} y comando {command}");
            }
        }
    }

    private void OnTriggerExit(Collider other)
    {
        if (other.CompareTag("silla de ruedas"))
        {
            Debug.Log($"Saliendo de zona {zoneID}");
        }
    }
}