# =========================
# Audio Vocal Extractor - Google Colab Stable
# =========================

import os
import gc
import sys
import torch
import librosa
import numpy as np
import soundfile as sf
import gradio as gr
import warnings

warnings.filterwarnings("ignore")

# =========================
# DIRECTORIOS
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output_audio")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# UTILIDADES
# =========================
def clean_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# =========================
# FUNCIÓN 1: EXTRACCIÓN EXTREMA DE VOZ PRINCIPAL
# =========================
def extract_main_vocal(audio_path, output_path):
    import noisereduce as nr
    import scipy.signal as sps

    # Cargar audio
    y, sr = librosa.load(audio_path, sr=44100, mono=True)

    # Normalizar entrada
    y = librosa.util.normalize(y)

    # Reducción de ruido (agresiva pero estable)
    y = nr.reduce_noise(
        y=y,
        sr=sr,
        stationary=False,
        prop_decrease=0.95
    )

    # Band-pass voz humana (lead vocal)
    sos = sps.butter(
        8,
        [90, 11000],
        btype="bandpass",
        fs=sr,
        output="sos"
    )
    y = sps.sosfilt(sos, y)

    # Gate dinámico (elimina voces lejanas y fondo)
    gate = np.mean(np.abs(y)) * 2.2
    y[np.abs(y) < gate] = 0.0

    # Compresión ligera (mejora claridad)
    rms = librosa.feature.rms(y=y)[0]
    rms = np.maximum(rms, 1e-6)
    y = y / np.mean(rms)

    # Normalización final
    y = librosa.util.normalize(y)

    sf.write(output_path, y, sr, subtype="PCM_16")
    return output_path

# =========================
# FUNCIÓN 2: PROCESAMIENTO SECUENCIAL (+20 AUDIOS)
# =========================
def process_files_sequential(files):
    results = []

    for idx, file in enumerate(files, 1):
        print(f"[{idx}/{len(files)}] Procesando: {os.path.basename(file)}")

        out_path = os.path.join(
            OUTPUT_DIR,
            "VOCAL_" + os.path.basename(file)
        )

        try:
            extract_main_vocal(file, out_path)
            results.append(out_path)
        except Exception as e:
            print("❌ Error en:", file, e)

        clean_memory()

    return results

# =========================
# FUNCIÓN GRADIO PRINCIPAL
# =========================
def sound_separate(media_files):
    if not media_files:
        return []

    if not isinstance(media_files, list):
        media_files = [media_files]

    return process_files_sequential(media_files)

# =========================
# INTERFAZ GRADIO (COLAB FRIENDLY)
# =========================
def launch_ui():
    with gr.Blocks(title="Vocal Extractor - Colab") as app:
        gr.Markdown(
            """
            # 🎤 Vocal Extractor (Colab)
            ✔ Solo voz principal  
            ✔ Sin ruido / voces secundarias  
            ✔ Procesa +20 audios poco a poco  
            ✔ Ideal para RVC / IA
            """
        )

        audio_input = gr.File(
            label="Sube tus audios",
            file_count="multiple",
            type="filepath"
        )

        process_btn = gr.Button("Procesar audios")

        output_files = gr.File(
            label="Resultados",
            file_count="multiple"
        )

        process_btn.click(
            fn=sound_separate,
            inputs=audio_input,
            outputs=output_files
        )

    return app

# =========================
# MAIN (COLAB)
# =========================
if __name__ == "__main__":
    app = launch_ui()
    app.launch(share=True, debug=False)
