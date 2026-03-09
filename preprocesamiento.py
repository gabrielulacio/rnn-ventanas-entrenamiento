import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path

def cargar_y_escalar_datos(ruta_archivo, columna='Close'):
    """
    Carga el CSV y escala la columna seleccionada entre 0 y 1.
    Por defecto usa el precio de cierre ('Close').
    """
    df = pd.read_csv(ruta_archivo)
    df[columna] = pd.to_numeric(df[columna], errors='coerce')
    df = df.dropna(subset=[columna])
    
    #Extraer la columna deseada como un array 2D para el escalador
    datos = df[[columna]].values
    
    # Inicializa y aplica el escalador Min-Max
    escalador = MinMaxScaler(feature_range=(0, 1))
    datos_escalados = escalador.fit_transform(datos)
    
    return datos_escalados, escalador

def crear_secuencias(datos, tamano_ventana):
    """
    Crea las matrices X (ventanas de entrenamiento) y y (valores a predecir).
    """
    X, y = [], []
    for i in range(len(datos) - tamano_ventana):
        X.append(datos[i:(i + tamano_ventana), 0])
        y.append(datos[i + tamano_ventana, 0])
        
    return np.array(X), np.array(y)

# Bloque de prueba para verificar que funciona al ejecutar este archivo directamente
if __name__ == "__main__":
    ruta = Path("data/btc_historico.csv")
    
    if ruta.exists():
        print(f"Procesando datos desde: {ruta}")
        datos_escalados, escalador = cargar_y_escalar_datos(ruta)
        
        # Probaremos con una ventana de 30 días
        ventana_prueba = 30 
        X, y = crear_secuencias(datos_escalados, tamano_ventana=ventana_prueba)
        
        print("\n--- Resultados del Preprocesamiento ---")
        print(f"Total de registros originales: {len(datos_escalados)}")
        print(f"Forma de la matriz X (Entradas): {X.shape}")
        print(f"Forma del vector y (Salidas): {y.shape}")
        print(f"\nEjemplo del primer valor a predecir (y[0] escalado): {y[0]:.4f}")
        
        # Invertir el escalado solo para comprobar el valor real
        precio_real = escalador.inverse_transform([[y[0]]])
        print(f"Valor real aproximado de y[0] en USD: ${precio_real[0][0]:.2f}")
    else:
        print(f"Error: No se encontró el archivo en {ruta}.")
        print("Asegúrate de haber ejecutado descargar_datos.py primero.")