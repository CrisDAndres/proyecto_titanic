# ANÁLISIS DEL DATASET DEL TITANIC

![Imagen Titanic](https://cdn.pixabay.com/photo/2023/10/06/17/14/ship-8298749_1280.png) 
*Fuente de la imagen: https://pixabay.com/*

Este proyecto tiene como objetivo cargar, preprocesar y analizar el conjunto de datos de los pasajeros del Titanic. El set de datos contiene información sobre pasajeros de este famoso viaje, incluyendo detalles como edad, género, clase, puerto de embarque o estado de supervivencia.

### Características del proyecto

- **Datos**:

    - El conjunto de datos se encuentra en formato CSV y está disponible en dentro de la carpeta ``Data`` con el nombre ``titanic.csv``.

    - En la misma carpeta también se encuentra el CSV de los datos preprocesados (``titanic_preprocessed.csv``) para su uso en la app de Streamlit.

- **Código**: El código usado puede visualizarse en la carpeta ``notebooks``, que incluye 2 notebooks de Jupyter con diferentes secciones:
    1. ``Preprocessing_EDA.ipynb``:

    - Carga de librerías y lectura del dataset
    - Información del dataset y búsqueda de valores nulos
    - Pre-procesamiento de los datos: reparación valores nulos y atípicos
    - Exploración, análisis y visualización de los datos (EDA)
    - Conclusiones
    2. ``ML_forecasting.ipynb``:   
    - Implementación de modelos de aprendizaje automático para la predicción de la supervivencia (modelos de clasificación)

- **Aplicación de Streamlit**: Se ha desarrollado una aplicación en Streamlit para la exploración de los datos analizados de una manera visual e interactiva. Está disponible en https://titanic-project-00.streamlit.app/


### Instrucciones de Ejecución 💻

Para ejecutar este proyecto en tu máquina local, sigue los siguientes pasos:

1. Clona este repositorio en tu máquina local.
2. Instala las dependencias necesarias ejecutando ``pip install -r requirements.txt``.
3. Ejecuta el script ``app_titanic.py`` el .csv, y asegúrate de tener las carpetas ``Data``, ``outputs`` y ``models`` en el mismo entorno. El código para ejecutar el script es ``streamlit run app_titanic.py``.
4. Se abrirá el local host ``http://localhost:8501/`` y aparecerá la aplicación.

### Contacto 📧

Si tienes alguna pregunta o sugerencia sobre este proyecto, no dudes en contactar conmigo.
