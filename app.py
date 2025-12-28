# Replace the existing final block (from "if __name__ == '__main__':" to the end)
# with this non-blocking launcher that starts model downloads in background and launches Gradio immediately.

if __name__ == "__main__":
    import threading
    import logging
    import time

    # ensure logs dir exists
    os.makedirs("logs", exist_ok=True)
    logging.basicConfig(
        filename=os.path.join("logs", "app_start.log"),
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    def _download_uvr_models():
        """
        Download UVR/MDX models in background. All output/errors go to logs.
        Runs in a daemon thread so it won't keep the process alive on exit.
        """
        try:
            logging.info("Background model downloader started.")
            MDX_DOWNLOAD_LINK = "https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/"
            UVR_MODELS_LOCAL = [
                "UVR-MDX-NET-Voc_FT.onnx",
                "UVR_MDXNET_KARA_2.onnx",
                "Reverb_HQ_By_FoxJoy.onnx",
                "UVR-MDX-NET-Inst_HQ_4.onnx",
            ]
            for id_model in UVR_MODELS_LOCAL:
                try:
                    dest_dir = mdxnet_models_dir
                    logging.info(f"Downloading {id_model} to {dest_dir}")
                    download_manager(os.path.join(MDX_DOWNLOAD_LINK, id_model), dest_dir)
                    logging.info(f"Finished download attempt for {id_model}")
                except Exception as e:
                    logging.exception(f"Failed downloading {id_model}: {e}")
            logging.info("Background UVR model downloads finished (or attempted).")
        except Exception:
            logging.exception("Unexpected error in _download_uvr_models")

    def _download_user_links():
        """
        If the user supplied model_links.txt, attempt to download them in background.
        Uses the existing download_models_from_list() if present; otherwise logs.
        """
        try:
            if os.path.exists("model_links.txt"):
                logging.info("Found model_links.txt — starting background downloads.")
                # call the function if defined in the file (from previous edits)
                try:
                    download_models_from_list()
                    logging.info("Background user model links download finished.")
                except NameError:
                    # Fallback: attempt to read and save file names only (no network)
                    logging.warning("download_models_from_list() not available; skipping automatic user downloads.")
            else:
                logging.info("No model_links.txt found — skipping user-model background downloads.")
        except Exception:
            logging.exception("Unexpected error in _download_user_links")

    # Start background download threads (daemon so they don't block shutdown)
    t1 = threading.Thread(target=_download_uvr_models, daemon=True, name="uvr-downloader")
    t1.start()
    t2 = threading.Thread(target=_download_user_links, daemon=True, name="userlinks-downloader")
    t2.start()

    # Small delay to let background threads initialize logs (not necessary)
    time.sleep(0.2)

    # Build and launch GUI immediately — downloads run in background now
    try:
        app = get_gui(theme)
        # keep a modest concurrency default — adjust if you need more throughput
        app.queue(default_concurrency_limit=8)
        app.launch(
            max_threads=40,
            share=IS_COLAB,     # shows public link in Colab if True
            show_error=True,
            quiet=False,        # show Gradio URLs and console output
            debug=IS_COLAB,
            ssr_mode=False,
        )
    except Exception:
        logging.exception("Failed launching Gradio GUI.")
        raise
