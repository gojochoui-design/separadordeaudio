# Patch: enable multi-file input (accept >20 files) and batch processing without breaking.
# Replace the existing audio_conf, sound_separate and the button binding in get_gui with these versions.
# Paste these functions into your app.py replacing the originals.

def audio_conf():
    """
    Allow multiple audio files to be selected and uploaded.
    Returns file paths (type='filepath') as a list when multiple files are chosen.
    """
    return gr.File(
        label="Audio file",
        file_count="multiple",   # <-- allow selecting many files
        type="filepath",
        container=True,
    )


def _process_single_file(
    media_file,
    stem,
    main,
    dereverb,
    vocal_effects,
    background_effects,
    vocal_reverb_room_size,
    vocal_reverb_damping,
    vocal_reverb_dryness,
    vocal_reverb_wet_level,
    vocal_delay_seconds,
    vocal_delay_mix,
    vocal_compressor_threshold_db,
    vocal_compressor_ratio,
    vocal_compressor_attack_ms,
    vocal_compressor_release_ms,
    vocal_gain_db,
    background_highpass_freq,
    background_lowpass_freq,
    background_reverb_room_size,
    background_reverb_damping,
    background_reverb_wet_level,
    background_compressor_threshold_db,
    background_compressor_ratio,
    background_compressor_attack_ms,
    background_compressor_release_ms,
    background_gain_db,
    target_format,
):
    """
    Process a single audio file through the existing pipeline.
    Returns a list of output file paths (one or more) for this input audio.
    Errors are caught per-file and logged — a failing file won't break the batch.
    """
    outputs = []
    try:
        # keep same behavior as before but isolated per-file
        hash_audio = str(get_hash(media_file))
        media_dir = os.path.dirname(media_file)

        # Call separation pipeline: process_uvr_task returns several paths
        _, _, _, _, vocal_audio = process_uvr_task(
            orig_song_path=media_file,
            song_id=hash_audio + "mdx",
            main_vocals=main,
            dereverb=dereverb,
            remove_files_output_dir=False,
        )

        # Vocal path handling
        if "vocal" in stem and vocal_audio:
            current_vocal = vocal_audio
            if vocal_effects:
                suffix = "_effects"
                base, ext = os.path.splitext(os.path.abspath(current_vocal))
                out_effects = base + suffix + ext
                out_effects_path = os.path.join(media_dir, os.path.basename(out_effects))
                try:
                    add_vocal_effects(
                        current_vocal,
                        out_effects_path,
                        reverb_room_size=vocal_reverb_room_size,
                        vocal_reverb_dryness=vocal_reverb_dryness,
                        reverb_damping=vocal_reverb_damping,
                        reverb_wet_level=vocal_reverb_wet_level,
                        delay_seconds=vocal_delay_seconds,
                        delay_mix=vocal_delay_mix,
                        compressor_threshold_db=vocal_compressor_threshold_db,
                        compressor_ratio=vocal_compressor_ratio,
                        compressor_attack_ms=vocal_compressor_attack_ms,
                        compressor_release_ms=vocal_compressor_release_ms,
                        gain_db=vocal_gain_db,
                        extreme_clean=True,
                    )
                    current_vocal = out_effects_path
                except Exception as e:
                    logger.warning(f"add_vocal_effects failed for {media_file}: {e}")
            outputs.append(current_vocal)

        # Background path handling
        if "background" in stem:
            background_audio, _ = process_uvr_task(
                orig_song_path=media_file,
                song_id=hash_audio + "voiceless",
                only_voiceless=True,
                remove_files_output_dir=False,
            )
            if background_audio:
                current_background = background_audio
                if background_effects:
                    suffix = "_effects"
                    base, ext = os.path.splitext(os.path.abspath(current_background))
                    out_effects = base + suffix + ext
                    out_effects_path = os.path.join(media_dir, os.path.basename(out_effects))
                    try:
                        add_instrumental_effects(
                            current_background,
                            out_effects_path,
                            highpass_freq=background_highpass_freq,
                            lowpass_freq=background_lowpass_freq,
                            reverb_room_size=background_reverb_room_size,
                            reverb_damping=background_reverb_damping,
                            reverb_wet_level=background_reverb_wet_level,
                            compressor_threshold_db=background_compressor_threshold_db,
                            compressor_ratio=background_compressor_ratio,
                            compressor_attack_ms=background_compressor_attack_ms,
                            compressor_release_ms=background_compressor_release_ms,
                            gain_db=background_gain_db,
                        )
                        current_background = out_effects_path
                    except Exception as e:
                        logger.warning(f"add_instrumental_effects failed for {media_file}: {e}")
                outputs.append(current_background)

        # Convert formats per-file if needed (preserve behavior)
        if target_format and target_format.lower() != "wav":
            try:
                converted = convert_format(outputs, media_dir, target_format)
                outputs = converted
            except Exception as e:
                logger.warning(f"convert_format failed for {media_file}: {e}")

    except Exception as e:
        logger.exception(f"Processing failed for {media_file}: {e}")

    return outputs


def sound_separate(
    media_files,  # now accepts list[str] or single str
    stem,
    main,
    dereverb,
    vocal_effects=True,
    background_effects=True,
    vocal_reverb_room_size=0.6,
    vocal_reverb_damping=0.6,
    vocal_reverb_dryness=0.8,
    vocal_reverb_wet_level=0.35,
    vocal_delay_seconds=0.4,
    vocal_delay_mix=0.25,
    vocal_compressor_threshold_db=-25,
    vocal_compressor_ratio=3.5,
    vocal_compressor_attack_ms=10,
    vocal_compressor_release_ms=60,
    vocal_gain_db=4,
    background_highpass_freq=120,
    background_lowpass_freq=11000,
    background_reverb_room_size=0.5,
    background_reverb_damping=0.5,
    background_reverb_wet_level=0.25,
    background_compressor_threshold_db=-20,
    background_compressor_ratio=2.5,
    background_compressor_attack_ms=15,
    background_compressor_release_ms=80,
    background_gain_db=3,
    target_format="WAV",
):
    """
    Accepts a single file path (str) or a list of file paths. Processes each independently
    and returns a flattened list of output file paths. Any single-file failure is logged,
    but processing continues for the rest.
    """

    # Normalize input to a list of paths
    if not media_files:
        raise ValueError("The audio path is missing.")

    if isinstance(media_files, (str, bytes)):
        files = [media_files]
    elif isinstance(media_files, (list, tuple)):
        # In Gradio sometimes a single file is passed as a list of dicts when type != filepath;
        # we expect list of file paths (strings). Coerce if necessary.
        coerced = []
        for item in media_files:
            if isinstance(item, dict) and "name" in item and "data" in item:
                # leave as-is; unlikely with type='filepath' but safe
                coerced.append(item["name"])
            else:
                coerced.append(item)
        files = coerced
    else:
        raise ValueError("Unsupported media_files type")

    all_outputs = []
    start_time = time.time()

    # Process files sequentially to avoid race conditions and excessive memory/GPU usage.
    # If you want parallel processing, we can add a thread pool, but for safety keep sequential here.
    for idx, fpath in enumerate(files):
        try:
            # Expand user and make absolute
            fpath = os.path.abspath(os.path.expanduser(fpath))
            if not os.path.exists(fpath):
                logger.warning(f"Skipping missing file: {fpath}")
                continue

            logger.info(f"Processing file {idx+1}/{len(files)}: {fpath}")
            outputs = _process_single_file(
                fpath,
                stem,
                main,
                dereverb,
                vocal_effects,
                background_effects,
                vocal_reverb_room_size,
                vocal_reverb_damping,
                vocal_reverb_dryness,
                vocal_reverb_wet_level,
                vocal_delay_seconds,
                vocal_delay_mix,
                vocal_compressor_threshold_db,
                vocal_compressor_ratio,
                vocal_compressor_attack_ms,
                vocal_compressor_release_ms,
                vocal_gain_db,
                background_highpass_freq,
                background_lowpass_freq,
                background_reverb_room_size,
                background_reverb_damping,
                background_reverb_wet_level,
                background_compressor_threshold_db,
                background_compressor_ratio,
                background_compressor_attack_ms,
                background_compressor_release_ms,
                background_gain_db,
                target_format,
            )

            # Append results
            for outp in outputs:
                if outp and os.path.exists(outp):
                    all_outputs.append(outp)
        except Exception as e:
            logger.exception(f"Unhandled error processing {fpath}: {e}")
            continue

    total_time = time.time() - start_time
    logger.info(f"Batch processing finished: {len(all_outputs)} outputs in {total_time:.2f}s")

    if not all_outputs:
        raise Exception("No outputs produced. Check logs for details.")

    # Gradio File output expects a list of file paths for multiple files
    return all_outputs


# In get_gui(), ensure the 'aud' component uses the updated audio_conf (file_count="multiple")
# and that button click passes that component directly to sound_separate.
# That binding was already shown in your get_gui; just ensure aud = audio_conf() and
# button_base.click uses aud as the first input (no change needed other than the audio_conf replacement).
