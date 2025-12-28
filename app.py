# (modificado) Audio separator with support for multiple file inputs
# Incluye la funcionalidad para admitir cargas múltiples de archivos y controlar errores individuales.

import os
import gc
import json
import queue
import threading
import torch
import librosa
import numpy as np
import soundfile as sf
import gradio as gr
from utils import remove_directory_contents, create_directories, download_manager, logger
from pedalboard import Pedalboard, Reverb, Delay, Compressor, Gain, HighpassFilter, LowpassFilter
import argparse
import warnings

# Configuración de argumentos por línea de comandos
parser = argparse.ArgumentParser(description="Run the app with optional sharing")
parser.add_argument('--share', action='store_true', help='Enable sharing mode')
parser.add_argument('--theme', type=str, default="default", help='Set the theme for Gradio UI')
args = parser.parse_args()

# Constantes y setup inicial
IS_COLAB = 'google.colab' in sys.modules or args.share
LOG_DIR = "logs"
MODEL_DIR = "mdx_models"
OUTPUT_DIR = "output_audio"

for dir_path in [LOG_DIR, MODEL_DIR, OUTPUT_DIR]:
    os.makedirs(dir_path, exist_ok=True)

warnings.filterwarnings("ignore")
logger.setup(os.path.join(LOG_DIR, "app.log"))

# Función para manejar múltiples archivos
def audio_conf():
    """
    Configuración para permitir múltiples archivos de audio como entrada.
    """
    return gr.File(
        label="Audio files",  # Cambia el texto para reflejar entradas múltiples
        file_count="multiple",  # Habilita la carga de múltiples archivos
        type="filepath",  # Retorna rutas de archivos
        container=True,
    )

def sound_separate(media_files, *args, **kwargs):
    """
    Permitir múltiples archivos de entrada y procesarlos de forma secuencial.
    - media_files: Lista de archivos subidos por el usuario.
    """
    results = []
    if not isinstance(media_files, list):
        media_files = [media_files]  # Si es un solo archivo, conviértelo en lista.

    for media_file in media_files:
        try:
            # Aquí puedes mantener o customizar tu lógica existente para procesar cada archivo
            result = process_single_file(media_file, *args, **kwargs)
            results.append(result)
        except Exception as e:
            # Captura errores para archivos individuales y registra el error.
            logger.error(f"Error processing file {media_file}: {e}")
    return results

def process_single_file(media_file, *args, **kwargs):
    """
    Procesamiento de un solo archivo.
    """
    logger.info(f"Processing file: {media_file}")
    # Simular lógica de procesamiento aquí
    # Reemplázalo con tu procesamiento real, ej., run_mdx o adjust_vocal_clean
    result_path = os.path.join(OUTPUT_DIR, os.path.basename(media_file))
    sf.write(result_path, np.random.randn(44100, 2), 44100)  # Generar archivo ficticio
    return result_path

# Interface gráfica de usuario (GUI)
def get_gui(theme="default"):
    """
    Configurar la interfaz de Gradio para procesar múltiples archivos.
    """
    with gr.Blocks(theme=theme) as app:
        gr.Markdown("<center><h1>Audio🔹Separator</h1></center>")
        gr.Markdown("Carga tus archivos de audio y separamos los elementos (vocales e instrumentales).")

        audio_input = audio_conf()
        process_button = gr.Button("Procesar")
        output_display = gr.File(label="Archivos Procesados", file_count="multiple")

        # Vincular los elementos
        process_button.click(sound_separate, inputs=audio_input, outputs=output_display)

        gr.Markdown("- También puedes consultar más herramientas en la [documentación](https://github.com/R3gm/Audio_separator_ui).")
    return app

# Lanzar la aplicación
if __name__ == "__main__":
    app = get_gui(args.theme)  # Configura el GUI con el tema opcional
    app.launch(share=args.share)  # Lanzar Gradio con opción de compartir público
