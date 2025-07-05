using UnityEngine;
using UnityEditor;

[CustomEditor(typeof(SimplificadorMasivo))]
public class SimplificadorMasivoEditor : Editor
{
    public override void OnInspectorGUI()
    {
        DrawDefaultInspector();

        SimplificadorMasivo simplificador = (SimplificadorMasivo)target;

        EditorGUILayout.Space(10);

        if (GUILayout.Button("Iniciar Reducción de Polígonos", GUILayout.Height(30)))
        {
            if (EditorUtility.DisplayDialog("Confirmar Reducción",
                "¿Estás seguro de que quieres iniciar el proceso de reducción de polígonos?\n\nSe mantendrán sin cambios:\n- Pilares\n- Columnas\n- Paredes\n- Vigas",
                "Sí, iniciar", "Cancelar"))
            {
                simplificador.SimplificarTodos();
            }
        }
    }
}
