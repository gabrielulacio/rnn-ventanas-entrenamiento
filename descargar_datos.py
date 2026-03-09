import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

def descargar_datos(ticker, anios=5):
    """
    Descarga el historico de una criptomoneda, guardandolo en un CSV.
    Ejemplos de ticker: 'BTC-USD', 'ETH-USD', 'ADA-USD', etc.
    
    :param ticker: Ticker de la criptomoneda.
    :param anios: Numero de años de historico a descargar.
    """
    #Definiendo la ruta de guardado
    directorio_datos = Path("data")
    directorio_datos.mkdir(parents=True, exist_ok=True)

    fecha_fin = datetime.now()
    fecha_inicio = fecha_fin - timedelta(days=anios*365)

    print(f"Descargando datos de {ticker} desde {fecha_inicio.strftime('%Y-%m-%d')} hasta {fecha_fin.strftime('%Y-%m-%d')}...")

    #Descarga de datos usando yfinance
    datos = yf.download(ticker, start=fecha_inicio.strftime('%Y-%m-%d'), end=fecha_fin.strftime('%Y-%m-%d'))

    # Limpieza y estructuración del DataFrame
    datos.reset_index(inplace=True)
    nombre_archivo = f"{ticker.split('-')[0].lower()}_historico.csv"

    #Guardando a CSV
    ruta_completa = directorio_datos / nombre_archivo
    datos.to_csv(ruta_completa, index=False)
    print(f"Datos guardados en {ruta_completa} con {len(datos)} registros.")


if __name__ == "__main__":
    descargar_datos('BTC-USD', anios=5)