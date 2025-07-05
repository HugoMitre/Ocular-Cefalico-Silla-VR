Descripción 

Se implementa un sistema para combinar y registrar múltiples datos de seguimiento en Unity. Incluye la recopilación de información sobre la dirección de la cabeza, velocidad angular, desviación, permanencia, dirección de la mirada y posición de la silla de ruedas en tiempo real. Los datos se almacenan en archivos CSV organizados por participante e intento, facilitando su análisis posterior.

Requisitos
Unity 2021.3 o superior

C#

Biblioteca Camera.main

Sistema de almacenamiento en archivos CSV

Editor de Unity con acceso a PlayerPrefs

Instalación
utiliza los scripts .cs que se encuentran en este repositorio, importante, el participante delector al ir en la parte superior como menu, debe ir el programa en una carpeta llamda editor es muy importante que tenga este nombre.

Uso del Proyecto
Ejecuta la escena principal en Unity. Configura los objetos en el Inspector para enlazar la cámara, la silla de ruedas y los scripts de seguimiento. en mi caso son estos, estos pueden cambiar. ¿por que son estos? basicamente porque en estos objetos es en donde se encuentran los scripts de los cuales se quiere recopilar los datos, si puedes observar en todos los proyectos tebemos las variables publicas , estas son publicas debido a que se necesita que se acceda a la inofrmacion de etas . El sistema comenzará a recopilar datos en tiempo real sobre la dirección de la cabeza, la mirada y la posición del usuario. Los datos se registrarán automáticamente en archivos CSV dentro de la carpeta C:\Users\Manuel Delgado\Documents\VR_Study. cambiar esto a combeniencia esta es la direccion original Se puede iniciar un guardado manual presionando la tecla S. En la consola de Unity, puedes presionar D para visualizar un resumen de los datos recopilados.

Scripts Principales
4.1. DataCombiner.cs

Actúa como el núcleo del sistema, recopilando y combinando datos de múltiples scripts.

Obtiene información de seguimiento de cabeza, mirada y silla de ruedas. Organiza los datos en una estructura CSV con columnas detalladas Permite la actualización de la tarea y el comando en ejecución. Gestiona la creación de carpetas para organizar los archivos de datos.

4.2. ParticipantSelector.cs

Agrega una ventana en el editor de Unity para seleccionar el número de participante e intento. Guarda la configuración utilizando PlayerPrefs. Genera la estructura de carpetas automáticamente para almacenar los datos.

4.3. CybersicknessRecorder.cs

Registra el estado de mareo cibernético del participante mediante una simple pulsación de tecla. Alterna entre estados (0=sin mareo, 1=con mareo) presionando la tecla espaciadora, proporcionando feedback visual en pantalla. Captura datos a intervalos regulares configurables (por defecto cada 0.2 segundos). Almacena una serie temporal que incluye tiempo transcurrido y estado binario de mareo. Se integra con DataCombiner a través de la propiedad UltimoEstadoSickness para análisis correlacionados. Guarda automáticamente los datos en formato CSV al finalizar la sesión o manualmente presionando la tecla S. Facilita la identificación de patrones entre movimientos en el entorno virtual y la aparición de síntomas de mareo cibernético.

Guardado de Datos en CSV
Los datos de velocidad angular, desviación, permanencia y posición se registran automáticamente. Se almacenan en archivos CSV dentro de C:\Users\Manuel Delgado\Documents\VR_Study. cambiar esto a combeniencia esta es la direccion original Los datos se organizan en carpetas por Participante e Intento, permitiendo una fácil gestión. Los archivos CSV incluyen información sobre:

Tiempo

Ángulo Horizontal y Vertical

Dirección de la cabeza y mirada Velocidad angular normalizada Desviación estándar Tiempo de permanencia Posición real y posición ideal de la silla de ruedas Entrada de control de la silla de ruedas Acciones realizadas por el usuario

Consideraciones Finales
Configura el intervalo de muestreo en DataCombiner.cs para ajustar la frecuencia de recolección de datos. Modifica los scripts de ParticipantSelector.cs si deseas personalizar la estructura de carpetas. Usa Debug.Log() para verificar los valores en la consola de Unity si necesitas depuración en tiempo real. Puedes visualizar el contenido del CSV presionando D en la consola de Unity.
