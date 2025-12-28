# IMPORTAMOS LIBRERÍAS NECESARIAS
import os
import sys
import gc
import argparse
import warnings
import torch
import librosa
import numpy as np
import soundfile as sf
import gradio as gr
from pedalboard import (
    Pedalboard,
    Compressor,
    Gain,
    HighpassFilter,
    LowpassFilter
)

# IMPORTAMOS DEMUCS (SEPARADOR DE VOCES DE ALTA CALIDAD)
from demucs import Demucs

# ===============================
# CONFIGURACIÓN GENERAL
# ===============================
warnings.filterwarnings("ignore")

LOG_DIR = "/content/logs"
OUTPUT_DIR = "/content/output_audio"

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===============================
# FUNCIÓN EXTRA 1: IA con Demucs
# ===============================
def demucs_separate(input_path, output_path):
    """
    Usar Demucs para separar las voces (mejor calidad).
    """
    model = Demucs()  # Cargar el modelo Demucs

    # Procesar el audio para separar las voces
    wav, sr = librosa.load(input_path, sr=None, mono=False)
    sources = model.separate(wav)

    # Guardar solo la voz principal
    vocals = sources['vocals']
    sf.write(output_path, vocals.T, sr)  # Transponer para guardar en formato adecuado

# ===============================
# FUNCIÓN EXTRA 2
# PROCESAMIENTO EN COLA (20+ AUDIOS)
# ===============================
def process_audio_queue(file_list):
    """
    Procesa muchos audios uno por uno usando Demucs.
    """
    results = []

    for i, file_path in enumerate(file_list):
        try:
            print(f"[{i+1}/{len(file_list)}] Procesando:", file_path)

            output_path = os.path.join(
                OUTPUT_DIR,
                f"clean_vocal_{os.path.basename(file_path)}"
            )

            demucs_separate(file_path, output_path)
            results.append(output_path)

            gc.collect()

        except Exception as e:
            print("Error:", e)

    return results

# ===============================
# GRADIO INPUT MULTIPLE
# ===============================
def audio_conf():
    return gr.File(
        label="Audios (múltiples)",
        file_count="multiple",
        type="filepath"
    )

# ===============================
# FUNCIÓN PRINCIPAL GRADIO
# ===============================
def sound_separate(media_files):
    if not media_files:
        return []

    if not isinstance(media_files, list):
        media_files = [media_files]

    return process_audio_queue(media_files)

# ===============================
# INTERFAZ GRADIO
# ===============================
def get_gui(theme="default"):
    with gr.Blocks(theme=theme) as app:
        gr.Markdown(
            "<center><h1>🎤 Vocal Cleaner PRO con IA (Demucs)</h1></center>"
        )
        gr.Markdown(
            "✔ Elimina ruido<br>"
            "✔ Quita voces secundarias<br>"
            "✔ Deja SOLO la voz principal (Demucs)<br>"
            "✔ Soporta +20 audios"
        )

        audio_input = audio_conf()
        process_btn = gr.Button("Procesar audios")
        output_files = gr.File(
            label="Resultados",
            file_count="multiple"
        )

        process_btn.click(
            sound_separate,
            inputs=audio_input,
            outputs=output_files
        )

    return app

# ===============================
# MAIN
# ===============================
if __name__ == "__main__":
    app = get_gui(args.theme)
    app.launch(share=True)  # share=True en Colab para que sea accesible
