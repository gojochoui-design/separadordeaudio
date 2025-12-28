import os
import gc
import subprocess
import gradio as gr

# =========================
# RUTAS
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output_audio")
TMP_DIR = os.path.join(BASE_DIR, "tmp_demucs")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TMP_DIR, exist_ok=True)

# =========================
# LIMPIEZA
# =========================
def clean_mem():
    gc.collect()

# =========================
# DEMUCS (VOCALS ONLY)
# =========================
def run_demucs(audio_path):
    """
    Ejecuta Demucs y devuelve la ruta del vocal.wav
    """
    cmd = [
        "demucs",
        "--two-stems", "vocals",
        "-n", "htdemucs",
        "-o", TMP_DIR,
        audio_path
    ]

    subprocess.run(cmd, check=True)

    name = os.path.splitext(os.path.basename(audio_path))[0]
    vocal_path = os.path.join(
        TMP_DIR,
        "htdemucs",
        name,
        "vocals.wav"
    )

    final_out = os.path.join(
        OUTPUT_DIR,
        f"VOCAL_{name}.wav"
    )

    os.replace(vocal_path, final_out)
    return final_out

# =========================
# PROCESAR MUCHOS AUDIOS
# =========================
def process_files(files):
    outputs = []

    for i, f in enumerate(files, 1):
        print(f"[{i}/{len(files)}] Procesando:", os.path.basename(f))
        try:
            out = run_demucs(f)
            outputs.append(out)
        except Exception as e:
            print("❌ Error:", e)

        clean_mem()

    return outputs

# =========================
# FUNCIÓN GRADIO
# =========================
def run(media_files):
    if not media_files:
        return []

    if not isinstance(media_files, list):
        media_files = [media_files]

    return process_files(media_files)

# =========================
# INTERFAZ
# =========================
with gr.Blocks(title="Demucs Vocal Extractor") as app:
    gr.Markdown(
        """
        # 🎤 Demucs Vocal Extractor
        - Separación REAL
        - Voz principal (vocals)
        - Google Colab
        - Múltiples audios
        """
    )

    audio_input = gr.File(
        label="Sube tus audios",
        file_count="multiple",
        type="filepath"
    )

    btn = gr.Button("Procesar")

    output_files = gr.File(
        label="Voces separadas",
        file_count="multiple"
    )

    btn.click(
        fn=run,
        inputs=audio_input,
        outputs=output_files
    )

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    app.launch(share=True)
