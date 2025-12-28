def aggressive_vocal_clean(input_path: str, output_path: str, sr: int = 44100,
                           n_fft: int = 2048, hop_length: int = 512,
                           n_std_thresh: float = 1.5, prop_decrease: float = 1.0,
                           freq_smooth: int = 3, time_smooth: int = 15,
                           use_hpss: bool = True, percussive_attenuation: float = 0.12,
                           median_time: int = 9):
    """
    Aggressive spectral cleaning improved to reduce short transient noises (chairs, people, golpes).
    - use_hpss: aplica separación harmonic/percussive y atenúa la parte percutiva.
    - percussive_attenuation: factor [0..1] para mantener la parte percutiva (0 = eliminarla).
    - median_time: ventana (en frames) de mediana temporal aplicada a la máscara para suprimir transitorios.
    Otros parámetros igual que antes.
    """
    try:
        data, file_sr = sf.read(input_path, always_2d=True)
        # Normalize/resample if needed
        if file_sr != sr:
            # data shape: (frames, channels)
            data = data.T
            resampled = []
            for ch in range(data.shape[0]):
                d = data[ch].astype(np.float32)
                resampled_ch = librosa.resample(d, file_sr, sr)
                resampled.append(resampled_ch)
            data = np.stack(resampled, axis=0)
            file_sr = sr
        else:
            # ensure (channels, samples)
            if data.ndim == 2:
                data = data.T
            else:
                data = data.reshape((1, -1))

        def _median_filter_1d(arr, win):
            if win <= 1:
                return arr
            pad = win // 2
            a = np.pad(arr, (pad, pad), mode='edge')
            out = np.empty_like(arr)
            # sliding median
            for i in range(len(arr)):
                out[i] = np.median(a[i:i + win])
            return out

        cleaned_ch = []
        for ch in range(data.shape[0]):
            y = data[ch].astype(np.float32)

            # Optional HPSS to separate harmonic (vocal) and percussive (transients)
            if use_hpss:
                try:
                    y_harm, y_perc = librosa.effects.hpss(y)
                    S_harm = librosa.stft(y_harm, n_fft=n_fft, hop_length=hop_length, window='hann')
                    S_perc = librosa.stft(y_perc, n_fft=n_fft, hop_length=hop_length, window='hann')
                    S_full = S_harm + percussive_attenuation * S_perc
                except Exception:
                    # fallback to direct STFT
                    S_full = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, window='hann')
            else:
                S_full = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, window='hann')

            mag = np.abs(S_full)
            phase = np.angle(S_full)

            # Noise estimation: pick quiet frames by frame energy (10th percentile)
            frame_energy = np.mean(mag, axis=0)
            perc10 = np.percentile(frame_energy, 10)
            noise_frames_idx = np.where(frame_energy <= perc10)[0]
            if noise_frames_idx.size == 0:
                noise_frames_idx = np.arange(min(10, mag.shape[1]))
            # per-frequency noise mag (median across quiet frames)
            noise_mag = np.median(mag[:, noise_frames_idx], axis=1, keepdims=True)

            # Spectral subtraction
            mag_sub = mag - noise_mag * prop_decrease
            mag_sub = np.maximum(mag_sub, 0.0)

            # Soft mask
            eps = 1e-8
            mask = mag_sub / (mag + eps)

            # Thresholding by SNR-like rule per-frequency
            thresh_matrix = (noise_mag * n_std_thresh)
            mask = np.where(mag > thresh_matrix, mask, 0.0)

            # Apply temporal median filter on mask to suppress very short spikes (transients)
            if median_time > 1:
                # mask shape: (freq_bins, frames) -> apply median along time axis for each freq
                mask_med = np.zeros_like(mask)
                for f in range(mask.shape[0]):
                    mask_med[f, :] = _median_filter_1d(mask[f, :], median_time)
                mask = mask_med

            # Frequency/time smoothing as before (light smoothing to reduce musical noise)
            mask = _smooth_mask(mask, freq_smooth=freq_smooth, time_smooth=time_smooth)
            mask = np.clip(mask, 0.0, 1.0)

            # Power-shape the mask a bit to favor stronger suppression of low-energy parts
            mask = mask ** 1.15

            # Apply mask to original complex STFT (preserves harmonic/percussive recombination)
            S_clean = mask * S_full

            # ISTFT - preserve original length
            y_clean = librosa.istft(S_clean, hop_length=hop_length, window='hann', length=len(y))

            # Small post processing: DC remove and gentle highpass around 70-80Hz to remove rumble
            y_clean = y_clean - np.mean(y_clean)
            cleaned_ch.append(y_clean)

        # Align lengths and write
        maxlen = max(map(len, cleaned_ch))
        cleaned = np.stack([np.pad(c, (0, maxlen - len(c)), mode='constant') for c in cleaned_ch], axis=0)
        out = cleaned.T  # (samples, channels)
        sf.write(output_path, out, file_sr)
        return output_path

    except Exception as e:
        logger.error(f"Aggressive clean failed (enhanced): {e}")
        # fallback: return original
        try:
            out_data, _ = sf.read(input_path)
            sf.write(output_path, out_data, sr)
            return output_path
        except Exception:
            return input_path
