using UnityEngine;
using System.Collections;

public class InstructionalAudioZone.cs : MonoBehaviour
{
    [Header("Clips de Audio")]
    [Tooltip("Audio con instrucciones completas para la Etapa 1")]
    [SerializeField] private AudioClip audioEtapa1;

    [Tooltip("Audio con instrucciones breves para la Etapa 2")]
    [SerializeField] private AudioClip audioEtapa2;

    [Tooltip("Audio con instrucciones breves para la Etapa 3")]
    [SerializeField] private AudioClip audioEtapa3;

    [Header("Configuración")]
    [Tooltip("Segundos de espera antes de reproducir el audio")]
    [SerializeField] private float delayInicial = 2.0f;
    [SerializeField] private AudioSource audioSource;

    private void Awake()
    {
        // Asegurarse de que tengamos un AudioSource
        if (audioSource == null)
            audioSource = GetComponent<AudioSource>();

        if (audioSource == null)
            audioSource = gameObject.AddComponent<AudioSource>();
    }

    private void Start()
    {
        // Reproducir con un pequeño delay inicial
        StartCoroutine(ReproducirAudioEtapa());
    }

    private IEnumerator ReproducirAudioEtapa()
    {
        yield return new WaitForSeconds(delayInicial);

        // Obtener la etapa actual desde PlayerPrefs
        int etapaActual = PlayerPrefs.GetInt("EtapaActual", 1);
        AudioClip audioAReproducir = null;

        // Seleccionar el audio según la etapa
        switch (etapaActual)
        {
            case 1:
                audioAReproducir = audioEtapa1;
                break;
            case 2:
                audioAReproducir = audioEtapa2;
                break;
            case 3:
                audioAReproducir = audioEtapa3;
                break;
            default:
                Debug.LogWarning($"Etapa no reconocida: {etapaActual}, usando audio de Etapa 1");
                audioAReproducir = audioEtapa1;
                break;
        }

        // Reproducir el audio seleccionado
        if (audioAReproducir != null)
        {
            audioSource.clip = audioAReproducir;
            audioSource.Play();
            Debug.Log($"Reproduciendo audio para Etapa {etapaActual}");
        }
        else
        {
            Debug.LogError($"No se ha asignado audio para la Etapa {etapaActual}");
        }
    }
}
