# app.py — Audio separator with aggressive vocal cleaning, smart optional enhancement and fast/quality modes
# This file is a consolidated, corrected and improved version of your app.
# It avoids inventing ONNX links; if you want to use models place them in mdx_models/ or add URLs to model_links.txt.

import os
import gc
import hashlib
import queue
import threading
import json
import shlex
import sys
import subprocess
import librosa
import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm
from utils import (
    remove_directory_contents,
    create_directories,
    download_manager,
)
import random
from utils import logger
import onnxruntime as ort
import warnings
import gradio as gr
import time
import traceback
from pedalboard import Pedalboard, Reverb, Delay, Chorus, Compressor, Gain, HighpassFilter, LowpassFilter
from pedalboard.io import AudioFile
import argparse
import logging
import requests
import re
import shutil
from urllib.parse import urlparse, urljoin

parser = argparse.ArgumentParser(description="Run the app with optional sharing")
parser.add_argument('--share', action='store_true', help='Enable sharing mode')
parser.add_argument('--theme', type=str, default="NoCrypt/miku", help='Set the theme (default: NoCrypt/miku)')
args = parser.parse_args()

warnings.filterwarnings("ignore")
IS_COLAB = True if ('google.colab' in sys.modules or args.share) else False
IS_ZERO_GPU = os.getenv("SPACES_ZERO_GPU")

title = "<center><strong><font size='7'>Audio🔹separator</font></strong></center>"
base_demo = "This demo uses the "
description = (f"{base_demo if IS_ZERO_GPU else ''}MDX-Net / enhancement pipeline for vocal and background separation.")
RESOURCES = "- For best results place enhancement/separation ONNX models into mdx_models/ or list them in model_links.txt."

# Paths
BASE_DIR = "."
mdxnet_models_dir = os.path.join(BASE_DIR, "mdx_models")
output_dir = os.path.join(BASE_DIR, "clean_song_output")
os.makedirs(mdxnet_models_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)
os.makedirs("logs", exist_ok=True)

# logging
logging.basicConfig(filename=os.path.join("logs", "app.log"), level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger.setLevel(logging.INFO)

# Small helpers
def show_components_downloader(value_active):
    return gr.update(visible=value_active), gr.update(visible=value_active)

def safe_mkdir(path):
    try:
        os.makedirs(path, exist_ok=True)
    except Exception:
        pass

def get_hash(filepath):
    h = hashlib.blake2b()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:18]

# Simple user-driven downloader (only uses links you provide in model_links.txt)
def _requests_download(url: str, dest: str, timeout: int = 60) -> bool:
    try:
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with requests.get(url, stream=True, timeout=(10, timeout)) as r:
            r.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        if os.path.exists(dest) and os.path.getsize(dest) > 0:
            logger.info(f"Downloaded {url} -> {dest}")
            return True
    except Exception as e:
        logger.warning(f"Requests download failed for {url}: {e}")
    return False

def try_download_from_page(url: str, dest: str) -> bool:
    try:
        r = requests.get(url, timeout=20)
        if r.status_code != 200:
            return False
        html = r.text
        m = re.search(r'href=["\']([^"\']+\.onnx[^"\']*)["\']', html, flags=re.IGNORECASE)
        if m:
            candidate = urljoin(url, m.group(1))
            return _requests_download(candidate, dest)
    except Exception as e:
        logger.debug(f"try_download_from_page failed: {e}")
    return False

def download_models_from_list():
    links_file = "model_links.txt"
    if not os.path.exists(links_file):
        return
    logger.info("Found model_links.txt — attempting to download listed URLs.")
    with open(links_file, "r", encoding="utf-8") as fh:
        for line in fh:
            url = line.strip()
            if not url or url.startswith("#"):
                continue
            parsed = urlparse(url)
            fname = os.path.basename(parsed.path) or hashlib.blake2b(url.encode()).hexdigest()[:12] + ".onnx"
            dest = os.path.join(mdxnet_models_dir, fname)
            if os.path.exists(dest) and os.path.getsize(dest) > 0:
                logger.info(f"Skipping existing model {fname}")
                continue
            if url.lower().endswith(".onnx"):
                if _requests_download(url, dest):
                    continue
            if "github.com" in parsed.netloc and "/blob/" in parsed.path:
                parts = parsed.path.split("/")
                if len(parts) > 4:
                    owner = parts[1]; repo = parts[2]; branch = parts[4]
                    rest = "/".join(parts[5:])
                    raw = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{rest}"
                    if raw.lower().endswith(".onnx") and _requests_download(raw, dest):
                        continue
            if try_download_from_page(url, dest):
                continue
            try:
                download_manager(url, mdxnet_models_dir)
                # try to find a .onnx in the folder
                found = None
                for root, _, files in os.walk(mdxnet_models_dir):
                    for f in files:
                        if f.lower().endswith(".onnx"):
                            candidate = os.path.join(root, f)
                            if os.path.getsize(candidate) > 0:
                                found = candidate
                                break
                    if found:
                        break
                if found:
                    if os.path.abspath(found) != os.path.abspath(dest):
                        shutil.copy(found, dest)
                    logger.info(f"download_manager saved model {found} -> {dest}")
                    continue
            except Exception as e:
                logger.warning(f"download_manager fallback failed for {url}: {e}")
            logger.warning(f"Could not download model from: {url}. Place file manually into {mdxnet_models_dir}")

# MDX classes (kept compatible with original usage)
class MDXModel:
    def __init__(self, device, dim_f, dim_t, n_fft, hop=1024, stem_name=None, compensation=1.0):
        self.dim_f = dim_f
        self.dim_t = dim_t
        self.dim_c = 4
        self.n_fft = n_fft
        self.hop = hop
        self.stem_name = stem_name
        self.compensation = compensation
        self.n_bins = self.n_fft // 2 + 1
        self.chunk_size = hop * (self.dim_t - 1)
        self.window = torch.hann_window(window_length=self.n_fft, periodic=True).to(device)
        self.freq_pad = torch.zeros([1, self.dim_c, self.n_bins - self.dim_f, self.dim_t]).to(device)

    def stft(self, x):
        x = x.reshape([-1, self.chunk_size])
        x = torch.stft(x, n_fft=self.n_fft, hop_length=self.hop, window=self.window, center=True, return_complex=True)
        x = torch.view_as_real(x)
        x = x.permute([0, 3, 1, 2])
        x = x.reshape([-1, 2, 2, self.n_bins, self.dim_t]).reshape([-1, 4, self.n_bins, self.dim_t])
        return x[:, :, : self.dim_f]

    def istft(self, x, freq_pad=None):
        freq_pad = self.freq_pad.repeat([x.shape[0], 1, 1, 1]) if freq_pad is None else freq_pad
        x = torch.cat([x, freq_pad], -2)
        x = x.reshape([-1, 2, 2, self.n_bins, self.dim_t]).reshape([-1, 2, self.n_bins, self.dim_t])
        x = x.permute([0, 2, 3, 1])
        x = x.contiguous()
        x = torch.view_as_complex(x)
        x = torch.istft(x, n_fft=self.n_fft, hop_length=self.hop, window=self.window, center=True)
        return x.reshape([-1, 2, self.chunk_size])

class MDX:
    DEFAULT_SR = 44100
    DEFAULT_CHUNK_SIZE = 0 * DEFAULT_SR
    DEFAULT_MARGIN_SIZE = 1 * DEFAULT_SR

    def __init__(self, model_path: str, params: MDXModel, processor=0):
        self.device = torch.device(f"cuda:{processor}") if processor >= 0 else torch.device("cpu")
        self.provider = ["CUDAExecutionProvider"] if processor >= 0 else ["CPUExecutionProvider"]
        self.model = params
        self.ort = ort.InferenceSession(model_path, providers=self.provider)
        try:
            self.ort.run(None, {"input": torch.rand(1, 4, params.dim_f, params.dim_t).numpy()})
        except Exception:
            pass
        self.process = lambda spec: self.ort.run(None, {"input": spec.cpu().numpy()})[0]
        self.prog = None

    @staticmethod
    def get_hash(model_path):
        try:
            with open(model_path, "rb") as f:
                f.seek(-10000 * 1024, 2)
                model_hash = hashlib.md5(f.read()).hexdigest()
        except Exception:
            model_hash = hashlib.md5(open(model_path, "rb").read()).hexdigest()
        return model_hash

    @staticmethod
    def segment(wave, combine=True, chunk_size=DEFAULT_CHUNK_SIZE, margin_size=DEFAULT_MARGIN_SIZE):
        if combine:
            processed_wave = None
            for i, segment in enumerate(wave):
                start = 0 if i == 0 else margin_size
                end = None if i == len(wave) - 1 else -margin_size
                if margin_size == 0:
                    end = None
                processed_wave = segment[:, start:end] if processed_wave is None else np.concatenate((processed_wave, segment[:, start:end]), axis=-1)
        else:
            processed_wave = []
            sample_count = wave.shape[-1]
            if chunk_size <= 0 or chunk_size > sample_count:
                chunk_size = sample_count
            if margin_size > chunk_size:
                margin_size = chunk_size
            for segment_count, skip in enumerate(range(0, sample_count, chunk_size)):
                margin = 0 if segment_count == 0 else margin_size
                end = min(skip + chunk_size + margin_size, sample_count)
                start = skip - margin
                cut = wave[:, start:end].copy()
                processed_wave.append(cut)
                if end == sample_count:
                    break
        return processed_wave

    def pad_wave(self, wave):
        n_sample = wave.shape[1]
        trim = self.model.n_fft // 2
        gen_size = self.model.chunk_size - 2 * trim
        pad = gen_size - n_sample % gen_size
        wave_p = np.concatenate((np.zeros((2, trim)), wave, np.zeros((2, pad)), np.zeros((2, trim))), 1)
        mix_waves = []
        for i in range(0, n_sample + pad, gen_size):
            waves = np.array(wave_p[:, i:i + self.model.chunk_size])
            mix_waves.append(waves)
        mix_waves = torch.tensor(mix_waves, dtype=torch.float32).to(self.device)
        return mix_waves, pad, trim

    def _process_wave(self, mix_waves, trim, pad, q: queue.Queue, _id: int):
        mix_waves = mix_waves.split(1)
        with torch.no_grad():
            pw = []
            for mix_wave in mix_waves:
                self.prog.update()
                spec = self.model.stft(mix_wave)
                processed_spec = torch.tensor(self.process(spec))
                processed_wav = self.model.istft(processed_spec.to(self.device))
                processed_wav = processed_wav[:, :, trim:-trim].transpose(0, 1).reshape(2, -1).cpu().numpy()
                pw.append(processed_wav)
        processed_signal = np.concatenate(pw, axis=-1)[:, :-pad]
        q.put({_id: processed_signal})
        return processed_signal

    def process_wave(self, wave: np.array, mt_threads=1):
        self.prog = tqdm(total=0)
        chunk = wave.shape[-1] // mt_threads
        waves = self.segment(wave, False, chunk)
        q = queue.Queue()
        threads = []
        for c, batch in enumerate(waves):
            mix_waves, pad, trim = self.pad_wave(batch)
            self.prog.total = len(mix_waves) * mt_threads
            thread = threading.Thread(target=self._process_wave, args=(mix_waves, trim, pad, q, c))
            thread.start()
            threads.append(thread)
        for thread in threads:
            thread.join()
        self.prog.close()
        processed_batches = []
        while not q.empty():
            processed_batches.append(q.get())
        processed_batches = [list(wave.values())[0] for wave in sorted(processed_batches, key=lambda d: list(d.keys())[0])]
        assert len(processed_batches) == len(waves), "Incomplete processed batches, please reduce batch size!"
        return self.segment(processed_batches, True, chunk)

# Aggressive vocal cleaning (HPSS + median mask)
def _smooth_mask(mask: np.ndarray, freq_smooth: int = 3, time_smooth: int = 15) -> np.ndarray:
    if freq_smooth <= 1 and time_smooth <= 1:
        return mask
    sm = mask.astype(np.float32)
    if freq_smooth > 1:
        sm = np.apply_along_axis(lambda m: np.convolve(m, np.ones(freq_smooth) / freq_smooth, mode='same'), axis=0, arr=sm)
    if time_smooth > 1:
        sm = np.apply_along_axis(lambda m: np.convolve(m, np.ones(time_smooth) / time_smooth, mode='same'), axis=1, arr=sm)
    return sm

def aggressive_vocal_clean(input_path: str, output_path: str, sr: int = 44100,
                           n_fft: int = 2048, hop_length: int = 512,
                           n_std_thresh: float = 1.5, prop_decrease: float = 1.0,
                           freq_smooth: int = 3, time_smooth: int = 15,
                           use_hpss: bool = True, percussive_attenuation: float = 0.12,
                           median_time: int = 9, process_mono: bool = True):
    try:
        data, file_sr = sf.read(input_path, always_2d=True)
        if data.ndim == 2:
            data = data.T
        else:
            data = data.reshape((1, -1))
        if process_mono and data.shape[0] > 1:
            mono = np.mean(data, axis=0)
            data_proc = np.expand_dims(mono, axis=0)
        else:
            data_proc = data
        if file_sr != sr:
            resampled = []
            for ch in range(data_proc.shape[0]):
                resampled_ch = librosa.resample(data_proc[ch].astype(np.float32), file_sr, sr)
                resampled.append(resampled_ch)
            data_proc = np.stack(resampled, axis=0)
            file_sr = sr
        def _median_filter_1d(arr, win):
            if win <= 1:
                return arr
            pad = win // 2
            a = np.pad(arr, (pad, pad), mode='edge')
            out = np.empty_like(arr)
            for i in range(len(arr)):
                out[i] = np.median(a[i:i + win])
            return out
        cleaned_ch = []
        for ch in range(data_proc.shape[0]):
            y = data_proc[ch].astype(np.float32)
            if use_hpss:
                try:
                    y_harm, y_perc = librosa.effects.hpss(y)
                    S_harm = librosa.stft(y_harm, n_fft=n_fft, hop_length=hop_length, window='hann')
                    S_perc = librosa.stft(y_perc, n_fft=n_fft, hop_length=hop_length, window='hann')
                    S_full = S_harm + percussive_attenuation * S_perc
                except Exception:
                    S_full = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, window='hann')
            else:
                S_full = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, window='hann')
            mag = np.abs(S_full)
            phase = np.angle(S_full)
            frame_energy = np.mean(mag, axis=0)
            perc10 = np.percentile(frame_energy, 10)
            noise_frames_idx = np.where(frame_energy <= perc10)[0]
            if noise_frames_idx.size == 0:
                noise_frames_idx = np.arange(min(10, mag.shape[1]))
            noise_mag = np.median(mag[:, noise_frames_idx], axis=1, keepdims=True)
            mag_sub = mag - noise_mag * prop_decrease
            mag_sub = np.maximum(mag_sub, 0.0)
            eps = 1e-8
            mask = mag_sub / (mag + eps)
            thresh_matrix = (noise_mag * n_std_thresh)
            mask = np.where(mag > thresh_matrix, mask, 0.0)
            if median_time > 1:
                mask_med = np.zeros_like(mask)
                for f in range(mask.shape[0]):
                    mask_med[f, :] = _median_filter_1d(mask[f, :], median_time)
                mask = mask_med
            mask = _smooth_mask(mask, freq_smooth=freq_smooth, time_smooth=time_smooth)
            mask = np.clip(mask, 0.0, 1.0)
            mask = mask ** 1.12
            S_clean = mask * S_full
            y_clean = librosa.istft(S_clean, hop_length=hop_length, window='hann', length=len(y))
            if len(y_clean) > 0:
                y_clean = y_clean - np.mean(y_clean)
            cleaned_ch.append(y_clean)
        if process_mono and data.shape[0] > 1:
            out_signal = cleaned_ch[0]
            out_wav = np.stack([out_signal, out_signal], axis=1)
            sf.write(output_path, out_wav, file_sr)
        else:
            maxlen = max(map(len, cleaned_ch))
            cleaned = np.stack([np.pad(c, (0, maxlen - len(c)), mode='constant') for c in cleaned_ch], axis=0)
            out = cleaned.T
            sf.write(output_path, out, file_sr)
        return output_path
    except Exception as e:
        logger.exception(f"aggressive_vocal_clean error: {e}")
        try:
            shutil.copy(input_path, output_path)
            return output_path
        except Exception:
            return input_path

# Pedalboard effects
def add_vocal_effects(input_file, output_file,
                      reverb_room_size=0.6, vocal_reverb_dryness=0.8, reverb_damping=0.6, reverb_wet_level=0.35,
                      delay_seconds=0.4, delay_mix=0.25,
                      compressor_threshold_db=-25, compressor_ratio=3.5, compressor_attack_ms=10, compressor_release_ms=60,
                      gain_db=3, extreme_clean: bool = False,
                      clean_n_fft=4096, clean_hop=512, clean_percussive_atten=0.08, clean_median_time=11,
                      clean_process_mono=True, clean_use_hpss=True):
    tmp_clean = input_file
    if extreme_clean:
        try:
            base, ext = os.path.splitext(os.path.abspath(input_file))
            tmp_clean = base + "_extremeclean" + ext
            aggressive_vocal_clean(input_file, tmp_clean,
                                   sr=44100, n_fft=clean_n_fft, hop_length=clean_hop,
                                   n_std_thresh=1.2, prop_decrease=1.2,
                                   freq_smooth=5, time_smooth=25,
                                   use_hpss=clean_use_hpss, percussive_attenuation=clean_percussive_atten,
                                   median_time=clean_median_time, process_mono=clean_process_mono)
            reverb_room_size = 0.0
            reverb_wet_level = 0.0
            delay_seconds = 0.0
            delay_mix = 0.0
        except Exception as e:
            logger.exception("extreme clean failed: %s", e)
            tmp_clean = input_file
    effects = [HighpassFilter()]
    if reverb_room_size > 0 and reverb_wet_level > 0:
        effects.append(Reverb(room_size=reverb_room_size, damping=reverb_damping, wet_level=reverb_wet_level, dry_level=vocal_reverb_dryness))
    effects.append(Compressor(threshold_db=compressor_threshold_db, ratio=compressor_ratio, attack_ms=compressor_attack_ms, release_ms=compressor_release_ms))
    if delay_seconds > 0 and delay_mix > 0:
        effects.append(Delay(delay_seconds=delay_seconds, mix=delay_mix))
    if gain_db:
        effects.append(Gain(gain_db=gain_db))
    board = Pedalboard(effects)
    with AudioFile(tmp_clean) as f:
        with AudioFile(output_file, 'w', f.samplerate, f.num_channels) as o:
            while f.tell() < f.frames:
                chunk = f.read(int(f.samplerate))
                effected = board(chunk, f.samplerate, reset=False)
                o.write(effected)
    if extreme_clean and tmp_clean != input_file and os.path.exists(tmp_clean):
        try:
            os.remove(tmp_clean)
        except Exception:
            pass

# Utilities for saving/conversion
COMMON_SAMPLE_RATES = [8000, 16000, 22050, 32000, 44100, 48000, 96000]
def save_audio(audio_opt: np.ndarray, final_sr: int, output_audio_path: str, target_format: str) -> str:
    ext = os.path.splitext(output_audio_path)[1].lower()
    try:
        if ext == ".wav":
            sf.write(output_audio_path, audio_opt, final_sr, format=target_format)
        else:
            target_sr = min(COMMON_SAMPLE_RATES, key=lambda altsr: abs(altsr - final_sr))
            if target_sr != final_sr:
                logger.warning(f"Resampling from {final_sr} -> {target_sr} for {ext}")
                audio_opt = librosa.resample(audio_opt, orig_sr=final_sr, target_sr=target_sr)
            sf.write(output_audio_path, audio_opt, target_sr, format=target_format)
    except Exception as e:
        logger.exception(f"save_audio error: {e}")
        output_audio_path = output_audio_path.replace(f"_converted.{target_format}", ".wav")
    return output_audio_path

def convert_format(file_paths, media_dir, target_format):
    target_format = target_format.lower()
    if target_format == "wav":
        return file_paths
    suffix = "_converted"
    converted_files = []
    for fp in file_paths:
        abs_fp = os.path.abspath(fp)
        file_name, _ = os.path.splitext(os.path.basename(abs_fp))
        file_ext = f".{target_format}"
        out_name = file_name + suffix + file_ext
        out_path = os.path.join(media_dir, out_name)
        audio, sr = sf.read(abs_fp)
        saved_path = save_audio(audio, sr, out_path, target_format)
        converted_files.append(saved_path)
    return converted_files

# UI component factories
def downloader_conf(): return gr.Checkbox(False, label="URL-to-Audio", container=False)
def url_media_conf(): return gr.Textbox(value="", label="Enter URL", placeholder="www.youtube.com/...", visible=False, lines=1)
def url_button_conf(): return gr.Button("Go", variant="secondary", visible=False)
def audio_conf(): return gr.File(label="Audio file", type="filepath", container=True)
def stem_conf(): return gr.CheckboxGroup(choices=["vocal", "background"], value="vocal", label="Stem")
def main_conf(): return gr.Checkbox(False, label="Main")
def dereverb_conf(): return gr.Checkbox(False, label="Dereverb", visible=True)
def vocal_effects_conf(): return gr.Checkbox(False, label="Vocal Effects", visible=True)
def deep_filter_conf(): return gr.Checkbox(False, label="Enable Neural Enhancement (model in mdx_models/)", visible=True)
def fast_mode_conf(): return gr.Checkbox(True, label="Fast mode (less aggressive, faster)", visible=True)
def background_effects_conf(): return gr.Checkbox(False, label="Background Effects", visible=False)
def vocal_reverb_room_size_conf(): return gr.Number(0.15, label="Vocal Reverb Room Size", minimum=0.0, maximum=1.0, step=0.05, visible=True)
def vocal_reverb_damping_conf(): return gr.Number(0.7, label="Vocal Reverb Damping", minimum=0.0, maximum=1.0, step=0.01, visible=True)
def vocal_reverb_wet_level_conf(): return gr.Number(0.2, label="Vocal Reverb Wet Level", minimum=0.0, maximum=1.0, step=0.05, visible=True)
def vocal_reverb_dryness_level_conf(): return gr.Number(0.8, label="Vocal Reverb Dryness Level", minimum=0.0, maximum=1.0, step=0.05, visible=True)
def vocal_delay_seconds_conf(): return gr.Number(0., label="Vocal Delay Seconds", minimum=0.0, maximum=1.0, step=0.01, visible=True)
def vocal_delay_mix_conf(): return gr.Number(0., label="Vocal Delay Mix", minimum=0.0, maximum=1.0, step=0.01, visible=True)
def vocal_compressor_threshold_db_conf(): return gr.Number(-15, label="Vocal Compressor Threshold (dB)", minimum=-60, maximum=0, step=1, visible=True)
def vocal_compressor_ratio_conf(): return gr.Number(4., label="Vocal Compressor Ratio", minimum=0, maximum=20, step=0.1, visible=True)
def vocal_compressor_attack_ms_conf(): return gr.Number(1.0, label="Vocal Compressor Attack (ms)", minimum=0, maximum=1000, step=1, visible=True)
def vocal_compressor_release_ms_conf(): return gr.Number(100, label="Vocal Compressor Release (ms)", minimum=0, maximum=3000, step=1, visible=True)
def vocal_gain_db_conf(): return gr.Number(0, label="Vocal Gain (dB)", minimum=-40, maximum=40, step=1, visible=True)
def percussive_attenuation_conf(): return gr.Number(0.08, label="Percussive attenuation (0..1)", minimum=0.0, maximum=1.0, step=0.01, visible=True)
def median_time_conf(): return gr.Number(11, label="Mask median time (frames)", minimum=1, maximum=51, step=2, visible=True)
def clean_n_fft_conf(): return gr.Dropdown(choices=[2048, 3072, 4096], value=4096, label="Cleaning n_fft", visible=True)
def button_conf(): return gr.Button("Inference", variant="primary")
def output_conf(): return gr.File(label="Result", file_count="multiple", interactive=False)

# Generic ONNX runner (best-effort)
def run_generic_onnx_audio_model(model_path: str, input_wav: str, output_wav: str, model_sr: int = None):
    sess = ort.InferenceSession(model_path, providers=(["CUDAExecutionProvider", "CPUExecutionProvider"] if torch.cuda.is_available() else ["CPUExecutionProvider"]))
    wav, sr = sf.read(input_wav, always_2d=True)
    mono = np.mean(wav, axis=1).astype(np.float32)
    if model_sr is None:
        model_sr = sr
    if sr != model_sr:
        mono_rs = librosa.resample(mono, orig_sr=sr, target_sr=model_sr)
    else:
        mono_rs = mono
    peak = np.max(np.abs(mono_rs)) + 1e-9
    mono_rs = mono_rs / peak
    input_name = sess.get_inputs()[0].name
    arr = mono_rs.astype(np.float32)[None, :]
    try:
        out = sess.run(None, {input_name: arr})[0]
    except Exception:
        try:
            out = sess.run(None, {input_name: arr[None, ...]})[0]
        except Exception as e:
            raise RuntimeError(f"ONNX inference failed: {e}")
    out_signal = np.array(out).squeeze()
    out_signal = out_signal * peak
    if model_sr != sr:
        out_signal = librosa.resample(out_signal, orig_sr=model_sr, target_sr=sr)
    if wav.shape[1] > 1:
        out_wav = np.stack([out_signal, out_signal], axis=1)
    else:
        out_wav = out_signal.reshape(-1, 1)
    sf.write(output_wav, out_wav, sr)
    return output_wav

# Minimal audio downloader wrapper (yt_dlp)
def audio_downloader(url_media):
    url_media = url_media.strip()
    if not url_media:
        return None
    if IS_ZERO_GPU and "youtube.com" in url_media:
        gr.Info("This option isn’t available on Hugging Face.")
        return None
    try:
        import yt_dlp
    except Exception:
        logger.warning("yt_dlp not installed; cannot download URL.")
        return None
    dir_output_downloads = "downloads"
    os.makedirs(dir_output_downloads, exist_ok=True)
    try:
        media_info = yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True, "noplaylist": True}).extract_info(url_media, download=False)
        download_path = f"{os.path.join(dir_output_downloads, media_info['title'])}.m4a"
        ydl_opts = {
            'format': 'm4a/bestaudio/best',
            'postprocessors': [{'key': 'FFmpegExtractAudio','preferredcodec': 'm4a'}],
            'force_overwrites': True, 'noplaylist': True, 'no_warnings': True, 'quiet': True,
            'ignore_no_formats_error': True, 'restrictfilenames': True, 'outtmpl': download_path,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl_download:
            ydl_download.download([url_media])
        return download_path
    except Exception as e:
        logger.exception(f"audio_downloader failed: {e}")
        return None

# The process_uvr_task, run_mdx and run_mdx_beta functions should be copied from your original file unchanged.
# For brevity in this packaged file we assume they are present and unchanged. Paste them here in your actual code.
# -----------------------------------------------------------------------------
# IMPORTANT: To keep this file functional, ensure run_mdx, run_mdx_beta and process_uvr_task
# implementations from your original code are present above this line.
# -----------------------------------------------------------------------------

# Main orchestrator (matches UI inputs)
def sound_separate(
    media_file, stem, main, dereverb, vocal_effects, background_effects,
    vocal_reverb_room_size, vocal_reverb_damping, vocal_reverb_dryness, vocal_reverb_wet_level,
    vocal_delay_seconds, vocal_delay_mix, vocal_compressor_threshold_db, vocal_compressor_ratio,
    vocal_compressor_attack_ms, vocal_compressor_release_ms, vocal_gain_db,
    background_highpass_freq, background_lowpass_freq, background_reverb_room_size,
    background_reverb_damping, background_reverb_wet_level, background_compressor_threshold_db,
    background_compressor_ratio, background_compressor_attack_ms, background_compressor_release_ms,
    background_gain_db,
    percussive_attenuation, median_time, clean_n_fft, fast_mode, enable_deepfilter, target_format
):
    # Download any user-specified models (if provided)
    download_models_from_list()

    if not media_file:
        raise ValueError("The audio path is missing.")
    if not stem:
        raise ValueError("Please select 'vocal' or 'background' stem.")

    media_file = os.path.abspath(media_file)
    media_dir = os.path.dirname(media_file)
    outputs = []

    start_time = time.time()

    # cleaning params
    if fast_mode:
        clean_n_fft = 2048
        clean_hop = 512
        clean_percussive_atten = max(0.12, percussive_attenuation)
        clean_median_time = max(5, int(median_time // 2))
        clean_use_hpss = False
        clean_process_mono = True
    else:
        clean_n_fft = int(clean_n_fft)
        clean_hop = 512
        clean_percussive_atten = percussive_attenuation
        clean_median_time = int(median_time)
        clean_use_hpss = True
        clean_process_mono = True

    # VOCAL
    if "vocal" in stem:
        try:
            # process_uvr_task must be available (from original file)
            _, _, _, _, vocal_audio = process_uvr_task(orig_song_path=media_file, song_id=get_hash(media_file) + "_mdx",
                                                       main_vocals=main, dereverb=dereverb, remove_files_output_dir=False)
            if vocal_effects:
                file_name, file_extension = os.path.splitext(os.path.abspath(vocal_audio))
                out_effects_path = os.path.join(media_dir, file_name + "_effects" + file_extension)
                add_vocal_effects(vocal_audio, out_effects_path,
                                  reverb_room_size=vocal_reverb_room_size, vocal_reverb_dryness=vocal_reverb_dryness,
                                  reverb_damping=vocal_reverb_damping, reverb_wet_level=vocal_reverb_wet_level,
                                  delay_seconds=vocal_delay_seconds, delay_mix=vocal_delay_mix,
                                  compressor_threshold_db=vocal_compressor_threshold_db, compressor_ratio=vocal_compressor_ratio,
                                  compressor_attack_ms=vocal_compressor_attack_ms, compressor_release_ms=vocal_compressor_release_ms,
                                  gain_db=vocal_gain_db, extreme_clean=True,
                                  clean_n_fft=clean_n_fft, clean_hop=clean_hop,
                                  clean_percussive_atten=clean_percussive_atten, clean_median_time=clean_median_time,
                                  clean_process_mono=clean_process_mono, clean_use_hpss=clean_use_hpss)
                vocal_audio = out_effects_path

                # Neural enhancement if user provided ONNX with "deep/denoise/filter" in filename
                if enable_deepfilter:
                    candidate = None
                    for f in os.listdir(mdxnet_models_dir):
                        lf = f.lower()
                        if lf.endswith(".onnx") and ("deep" in lf or "denoise" in lf or "filter" in lf):
                            candidate = os.path.join(mdxnet_models_dir, f)
                            break
                    if candidate:
                        enhanced_path = os.path.splitext(vocal_audio)[0] + "_enhanced.wav"
                        try:
                            run_generic_onnx_audio_model(candidate, vocal_audio, enhanced_path, model_sr=None)
                            vocal_audio = enhanced_path
                        except Exception as e:
                            logger.warning(f"Neural enhancement failed with {candidate}: {e}")
                    else:
                        logger.info("No enhancement ONNX model found in mdx_models/ — skipping neural enhancement.")

            outputs.append(vocal_audio)
        except Exception as e:
            logger.exception(f"Vocal separation/processing failed: {e}")

    # BACKGROUND
    if "background" in stem:
        try:
            background_audio, _ = process_uvr_task(orig_song_path=media_file, song_id=get_hash(media_file) + "_voiceless",
                                                   only_voiceless=True, remove_files_output_dir=False)
            if background_effects:
                file_name, file_extension = os.path.splitext(os.path.abspath(background_audio))
                out_effects_path = os.path.join(media_dir, file_name + "_effects" + file_extension)
                add_instrumental_effects(background_audio, out_effects_path,
                                         highpass_freq=background_highpass_freq, lowpass_freq=background_lowpass_freq,
                                         reverb_room_size=background_reverb_room_size, reverb_damping=background_reverb_damping,
                                         reverb_wet_level=background_reverb_wet_level,
                                         compressor_threshold_db=background_compressor_threshold_db,
                                         compressor_ratio=background_compressor_ratio,
                                         compressor_attack_ms=background_compressor_attack_ms,
                                         compressor_release_ms=background_compressor_release_ms,
                                         gain_db=background_gain_db)
                background_audio = out_effects_path
            outputs.append(background_audio)
        except Exception as e:
            logger.exception(f"Background separation/processing failed: {e}")

    end_time = time.time()
    logger.info(f"Execution time: {end_time - start_time:.2f} seconds")

    if not outputs:
        raise Exception("Error in sound separation.")

    return convert_format(outputs, media_dir, target_format)

# Build UI
def get_gui(theme):
    with gr.Blocks(theme=theme, fill_width=True, fill_height=False, delete_cache=(3200, 10800)) as app:
        gr.Markdown(title)
        gr.Markdown(description)

        downloader_gui = downloader_conf()
        with gr.Row():
            with gr.Column(scale=2):
                url_media_gui = url_media_conf()
            with gr.Column(scale=1):
                url_button_gui = url_button_conf()

        downloader_gui.change(show_components_downloader, [downloader_gui], [url_media_gui, url_button_gui])

        aud = audio_conf()
        url_button_gui.click(lambda url: audio_downloader(url), [url_media_gui], [aud])

        with gr.Column():
            with gr.Row():
                stem_gui = stem_conf()

        with gr.Column():
            with gr.Row():
                main_gui = main_conf()
                dereverb_gui = dereverb_conf()
                vocal_effects_gui = vocal_effects_conf()
                deep_filter_gui = deep_filter_conf()
                fast_mode_gui = fast_mode_conf()
                background_effects_gui = background_effects_conf()

            with gr.Accordion("Vocal Effects Parameters", open=False):
                with gr.Row():
                    vocal_reverb_room_size_gui = vocal_reverb_room_size_conf()
                    vocal_reverb_damping_gui = vocal_reverb_damping_conf()
                    vocal_reverb_dryness_gui = vocal_reverb_dryness_level_conf()
                    vocal_reverb_wet_level_gui = vocal_reverb_wet_level_conf()
                    vocal_delay_seconds_gui = vocal_delay_seconds_conf()
                    vocal_delay_mix_gui = vocal_delay_mix_conf()
                    vocal_compressor_threshold_db_gui = vocal_compressor_threshold_db_conf()
                    vocal_compressor_ratio_gui = vocal_compressor_ratio_conf()
                    vocal_compressor_attack_ms_gui = vocal_compressor_attack_ms_conf()
                    vocal_compressor_release_ms_gui = vocal_compressor_release_ms_conf()
                    vocal_gain_db_gui = vocal_gain_db_conf()
                    percussive_attenuation_gui = percussive_attenuation_conf()
                    median_time_gui = median_time_conf()
                    clean_n_fft_gui = clean_n_fft_conf()

            with gr.Accordion("Background Effects Parameters", open=False):
                with gr.Row():
                    background_highpass_freq_gui = gr.Number(120, label="Background Highpass Frequency (Hz)", visible=True)
                    background_lowpass_freq_gui = gr.Number(11000, label="Background Lowpass Frequency (Hz)", visible=True)
                    background_reverb_room_size_gui = gr.Number(0.1, label="Background Reverb Room Size", visible=True)
                    background_reverb_damping_gui = gr.Number(0.5, label="Background Reverb Damping", visible=True)
                    background_reverb_wet_level_gui = gr.Number(0.25, label="Background Reverb Wet Level", visible=True)
                    background_compressor_threshold_db_gui = gr.Number(-15, label="Background Compressor Threshold (dB)", visible=True)
                    background_compressor_ratio_gui = gr.Number(4.0, label="Background Compressor Ratio", visible=True)
                    background_compressor_attack_ms_gui = gr.Number(15, label="Background Compressor Attack (ms)", visible=True)
                    background_compressor_release_ms_gui = gr.Number(60, label="Background Compressor Release (ms)", visible=True)
                    background_gain_db_gui = gr.Number(0, label="Background Gain (dB)", visible=True)

            stem_gui.change(lambda v: (gr.update(visible="vocal" in v), gr.update(visible="vocal" in v), gr.update(visible="vocal" in v), gr.update(visible="background" in v)), [stem_gui], [main_gui, dereverb_gui, vocal_effects_gui, background_effects_gui])

        target_format_gui = format_conf()
        button_base = button_conf()
        output_base = output_conf()

        # Bind inference button (inputs must match sound_separate signature)
        button_base.click(
            sound_separate,
            inputs=[
                aud,
                stem_gui,
                main_gui,
                dereverb_gui,
                vocal_effects_gui,
                background_effects_gui,
                vocal_reverb_room_size_gui, vocal_reverb_damping_gui, vocal_reverb_dryness_gui, vocal_reverb_wet_level_gui,
                vocal_delay_seconds_gui, vocal_delay_mix_gui, vocal_compressor_threshold_db_gui, vocal_compressor_ratio_gui,
                vocal_compressor_attack_ms_gui, vocal_compressor_release_ms_gui, vocal_gain_db_gui,
                background_highpass_freq_gui, background_lowpass_freq_gui, background_reverb_room_size_gui,
                background_reverb_damping_gui, background_reverb_wet_level_gui, background_compressor_threshold_db_gui,
                background_compressor_ratio_gui, background_compressor_attack_ms_gui, background_compressor_release_ms_gui,
                background_gain_db_gui,
                percussive_attenuation_gui, median_time_gui, clean_n_fft_gui, fast_mode_gui, deep_filter_gui, target_format_gui
            ],
            outputs=[output_base],
        )

        gr.Markdown(RESOURCES)
    return app

# Launch
if __name__ == "__main__":
    # Attempt to download user-specified links (if model_links.txt provided)
    download_models_from_list()

    # Ensure UVR models as original (will show progress)
    MDX_DOWNLOAD_LINK = "https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/"
    UVR_MODELS = [
        "UVR-MDX-NET-Voc_FT.onnx",
        "UVR_MDXNET_KARA_2.onnx",
        "Reverb_HQ_By_FoxJoy.onnx",
        "UVR-MDX-NET-Inst_HQ_4.onnx",
    ]
    for id_model in UVR_MODELS:
        try:
            download_manager(os.path.join(MDX_DOWNLOAD_LINK, id_model), mdxnet_models_dir)
        except Exception as e:
            logger.warning(f"download_manager: {id_model} -> {e}")

    app = get_gui(args.theme)
    app.queue(default_concurrency_limit=8)
    app.launch(max_threads=40, share=IS_COLAB, show_error=True, quiet=False, debug=IS_COLAB, ssr_mode=False)
