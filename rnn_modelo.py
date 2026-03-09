import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, SimpleRNN, Dense
from preprocesamiento import cargar_y_escalar_datos, crear_secuencias
from pathlib import Path

def construir_y_entrenar_rnn(X_train, y_train, X_val, y_val, tamano_ventana):
    """
    Construye una red neuronal recurrente simple, la entrena y retorna el error.
    """

    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_val = np.reshape(X_val, (X_val.shape[0], X_val.shape[1], 1))
    
    # Definir la arquitectura de la red
    modelo = Sequential([
        Input(shape=(tamano_ventana, 1)),
        SimpleRNN(50, activation='relu'),
        Dense(1)
    ])
    
    modelo.compile(optimizer='adam', loss='mse')
    
    # Entrenamiento del modelo
    historia = modelo.fit(X_train, y_train, epochs=20, batch_size=32, 
                          validation_data=(X_val, y_val), verbose=0)
    
    return historia.history['val_loss'][-1]

if __name__ == "__main__":
    ruta_csv = Path("data/btc_historico.csv")
    
    if not ruta_csv.exists():
        print("Error: Dataset no encontrado. Ejecuta descargar_datos.py primero.")
    else:
        # Cargar datos preprocesados
        datos_escalados, escalador = cargar_y_escalar_datos(ruta_csv)
        
        # Ventanas que exige la actividad para la comparación
        ventanas_a_probar = [3, 7, 14, 30, 60]
        errores_por_ventana = {}
        
        print("Iniciando entrenamiento. Esto puede tomar un par de minutos...\n")
        
        for ventana in ventanas_a_probar:
            X, y = crear_secuencias(datos_escalados, ventana)
            
            # Dividir los datos: 80% para entrenar, 20% para validar (pruebas)
            limite = int(len(X) * 0.8)
            X_train, X_val = X[:limite], X[limite:]
            y_train, y_val = y[:limite], y[limite:]
            
            print(f"Entrenando RNN con ventana de {ventana} días...")
            error_mse = construir_y_entrenar_rnn(X_train, y_train, X_val, y_val, ventana)
            errores_por_ventana[ventana] = error_mse
            print(f"-> Tasa de error (MSE): {error_mse:.6f}\n")
            
        print("-" * 50)
        print("RESUMEN DE TASAS DE ERROR (MSE) PARA DISCUSIÓN:")
        print("-" * 50)
        for ventana, error in errores_por_ventana.items():
            print(f"Ventana de {ventana:2d} días : MSE = {error:.6f}")