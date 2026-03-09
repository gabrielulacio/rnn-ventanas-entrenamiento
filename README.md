# Predicción de Criptomonedas con Redes Neuronales Recurrentes (RNN)

Este proyecto implementa una Red Neuronal Recurrente (RNN) simple utilizando TensorFlow/Keras para predecir el precio histórico de Bitcoin (BTC). El objetivo principal es comparar el rendimiento del modelo utilizando diferentes tamaños de ventanas de tiempo (3, 7, 14, 30 y 60 días) para identificar cuál ofrece una mejor capacidad predictiva.

## Estructura del Proyecto

*   `descargar_datos.py`: Script para descargar datos históricos de Yahoo Finance y guardarlos en un archivo CSV.
*   `preprocesamiento.py`: Contiene funciones para cargar los datos, escalarlos (Min-Max Scaling) y crear las ventanas temporales para el entrenamiento.
*   `rnn_modelo.py`: Define la arquitectura de la RNN, realiza el entrenamiento para diferentes ventanas y muestra una comparación de los errores (MSE).
*   `requirements.txt`: Lista de dependencias necesarias para ejecutar el proyecto.
*   `data/`: Carpeta (creada automáticamente) donde se almacena el archivo `btc_historico.csv`.

## Requisitos

*   Python 3.8+
*   Pip (gestor de paquetes de Python)

## Instalación

1.  **Clonar el repositorio** (o descargar los archivos):
    ```bash
    git clone <url-del-repositorio>
    cd rnn-ventanas-entrenamiento
    ```

2.  **Crear un entorno virtual** (recomendado):
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # En Linux/macOS
    # o
    .venv\Scripts\activate     # En Windows
    ```

3.  **Instalar las dependencias**:
    ```bash
    pip install -r requirements.txt
    ```

## Ejecución

El proyecto debe ejecutarse en el siguiente orden:

1.  **Descargar los datos**:
    Este script descargará los últimos 5 años de datos de BTC-USD.
    ```bash
    python descargar_datos.py
    ```

2.  **Entrenar y comparar modelos**:
    Este script procesará los datos, entrenará la RNN con diferentes ventanas de tiempo y mostrará los resultados del Error Cuadrático Medio (MSE) para cada una.
    ```bash
    python rnn_modelo.py
    ```

## Resultados

Al finalizar la ejecución de `rnn_modelo.py`, verás un resumen en la terminal con la tasa de error (MSE) para cada ventana de tiempo probada:

```text
RESUMEN DE TASAS DE ERROR (MSE) PARA DISCUSIÓN:
--------------------------------------------------
Ventana de  3 días : MSE = 0.XXXXXX
Ventana de  7 días : MSE = 0.XXXXXX
...
```

Este resumen permite analizar cómo influye la cantidad de datos históricos inmediatos en la precisión de la predicción de la red neuronal.
