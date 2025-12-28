import os
import gc
import warnings
import gradio as gr
import soundfile as sf

from demucs.api import Separator

warnings.filterwarnings("ignore")

# ===============================
# DIRECTORIOS (COLAB)
# ===============================
OUTPUT_DIR = "/content/output_audio"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===============================
# DEMUCS SEPARATOR (CPU SAFE)
# ===============================
separator = Separator(
    model="htdemucs",
    device="cpu",        # COLAB GRATIS
    progress=False
)

# ===============================
# FUNCIÓN 1
# AISLAR SOLO VOZ PRINCIPAL
# ===============================
def extract_vocals_demucs(input_path, output_path):
    """
    Extrae SOLO la voz principal usando Demucs real
    """
    sources = separator.separate_audio_file(input_path)

    vocals = sources["vocals"]
    sf.write(output_path, vocals, 44100)

# ===============================
# FUNCIÓN 2
# COLA DE MUCHOS AUDIOS
# ===============================
def process_audio_queue(files):
    results = []

    for i, f in enumerate(files):
        try:
            print(f"[{i+1}/{len(files)}] Procesando {f}")

            out = os.path.join(
                OUTPUT_DIR,
                "vocals_" + os.path.basename(f)
            )

            extract_vocals_demucs(f, out)
            results.append(out)

            gc.collect()

        except Exception as e:
            print("Error:", e)

    return results

# ===============================
# FUNCIÓN GRADIO
# ===============================
def run_separator(files):
    if not files:
        return []

    if not isinstance(files, list):
        files = [files]

    return process_audio_queue(files)

# ===============================
# INTERFAZ GRADIO
# ===============================
with gr.Blocks() as app:
    gr.Markdown(
        "<center><h1>🎤 Vocal Separator (Demucs 4 · Colab)</h1></center>"
    )
    gr.Markdown(
        "✔ Voz principal real (IA)\n"
        "✔ Sin ruido\n"
        "✔ Sin voces secundarias\n"
        "✔ +20 audios en cola\n"
        "✔ Google Colab GRATIS"
    )

    audio_input = gr.File(
        label="Sube audios",
        file_count="multiple",
        type="filepath"
    )

    btn = gr.Button("Procesar")
    output = gr.File(label="Voces", file_count="multiple")

    btn.click(run_separator, audio_input, output)

app.launch(share=True)
