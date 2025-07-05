using UnityEngine;

public class jugador_zona : MonoBehaviour
{
    private string currentTask = "";
    private string currentCommand = "";

    public string GetCurrentTask() => currentTask;
    public string GetCurrentCommand() => currentCommand;

    public void SetZoneData(string taskName, string command)
    {
        currentTask = taskName;
        currentCommand = command;
        Debug.Log($"Estableciendo datos: Tarea={taskName}, Comando={command}");
    }

    public void ClearZoneData()
    {
        currentTask = "";
        currentCommand = "";
        Debug.Log("Limpiando datos de zona");
    }
}
