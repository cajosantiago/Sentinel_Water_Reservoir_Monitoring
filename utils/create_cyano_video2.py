import os
import glob
import cv2 as cv
import numpy as np
import subprocess
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

start_date = "2025-06-01"
end_date = "2026-06-15"
img_dir = "generated_data/quality/Maranhao_larsys"

INDEX_PRETTY = {
    0: "Clorofila-a [mg/m³] (NDCI, Mishra 2012)",
    1: "Clorofila-a [mg/m³] (Soria-Perpinyà 2021, altos)",
    2: "Clorofila-a [mg/m³] (Soria-Perpinyà 2021, baixos)",
    3: "Cianobactérias [células/mL] (Potes 2018)",
    4: "Cianobactérias [mg/m³] (Soria-Perpinyà 2021)",
    5: "Turbidez [NTU] (Zhan 2022)",
    6: "CDOM [µg/L] (Soria-Perpinyà 2021)",
    7: "TSS [mg/L] (Soria-Perpinyà 2021)",
}

VARIABLE_MAP = {
    "cyano": {"suffix": "cyano_cells_mL", "pretty": INDEX_PRETTY[3], "VMIN": 0, "VMAX": 1200},
    "CDOM": {"suffix": "CDOM_ug_L", "pretty": INDEX_PRETTY[6], "VMIN": 0, "VMAX": 10},
    "chla": {"suffix": "chla_ndci_mg_m3", "pretty": INDEX_PRETTY[0], "VMIN": 0, "VMAX": 100},
    "TSS": {"suffix": "TSS_mg_L", "pretty": INDEX_PRETTY[7], "VMIN": 0, "VMAX": 10},
    "turbidity": {"suffix": "turbidity_NTU", "pretty": INDEX_PRETTY[5], "VMIN": 0, "VMAX": 10}
}

def main():
    # Map quality variable choices to their corresponding filename suffix and pretty names

    for quality_variable in VARIABLE_MAP:
        
        if quality_variable not in VARIABLE_MAP:
            print(f"[ERROR] Invalid quality_variable '{quality_variable}'. "
                f"Must be one of {list(VARIABLE_MAP.keys())}")
            return

        var_info = VARIABLE_MAP[quality_variable]
        suffix = var_info["suffix"]
        pretty_name = var_info["pretty"]
        VMIN = var_info["VMIN"]
        VMAX = var_info["VMAX"]

        print("=" * 80)
        print(f"   {pretty_name} Over Time Video Generator ({start_date} - {end_date})")
        print("=" * 80)

        if not os.path.exists(img_dir):
            print(f"[ERROR] Directory '{img_dir}' does not exist! Run generate_mago_images.py first.")
            return

        # Find only the "_nb" images
        pattern = os.path.join(img_dir, f"*_{suffix}.npy")
        img_paths = sorted(glob.glob(pattern))
        print(f"Found {len(img_paths)} images!")

        # Filter img_paths to only include with start and end dates
        filtered_paths = []
        for path in img_paths:
            filename = os.path.basename(path)
            # Extract date from filename (e.g., 20260607T113024Z_cc0.6pct_chla_ndci_mg_m3.png -> 2026-06-07)
            date_str = f"{filename[0:4]}-{filename[4:6]}-{filename[6:8]}"
            if start_date <= date_str <= end_date:
                filtered_paths.append(path)
        img_paths = filtered_paths

        if not img_paths:
            print(f"[ERROR] No {pretty_name} map images found!")
            return

        total_images = len(img_paths)
        print(f"Found {total_images} images to compile into the timelapse video.")

        # Read the first image to establish the base resolution
        # first_img = cv.imread(img_paths[0])
        first_img = np.load(img_paths[0])
        if first_img is None:
            print(f"[ERROR] Could not read first image: {img_paths[0]}")
            return
            
        h, w = first_img.shape
        # Ensure dimensions are even (H.264 libx264 codec requirement for universal playback)
        w = w if w % 2 == 0 else w + 1
        h = h if h % 2 == 0 else h + 1

        print(f"Base video resolution: {w} x {h}")

        temp_avi = "temp_lossless.avi"
        # Use MJPG lossless-like format for fast intermediate writing
        fourcc = cv.VideoWriter_fourcc(*'MJPG')
        
        # 2.0 frames per second (0.5 seconds per observation, giving a great timelapse flow)
        fps = 2.0
        video = cv.VideoWriter(temp_avi, fourcc, fps, (w, h))

        if not video.isOpened():
            print("[ERROR] Could not open VideoWriter!")
            return

        mago_colors = ["blue", "cyan", "green", "yellow", "red"]
        mago_cmap = mcolors.LinearSegmentedColormap.from_list("mago", mago_colors)
        mago_cmap.set_bad(color="black", alpha=1.0)
        print("Stitching frames...")
        for idx, path in enumerate(img_paths):
            # Read image
            # img = cv.imread(path, cv.IMREAD_UNCHANGED)
            # if img is None:
            #     print(f"  [WARN] Skipping unreadable image: {os.path.basename(path)}")
            #     continue
            filename = os.path.basename(path)
            # Extract date from filename (e.g., 20260607T113024Z_cc0.6pct_chla_ndci_mg_m3.png -> 2026-06-07)
            date_str = f"{filename[0:4]}-{filename[4:6]}-{filename[6:8]}"

            img = np.load(path)
            mago_colors = ["blue", "cyan", "green", "yellow", "red"]
            mago_cmap   = mcolors.LinearSegmentedColormap.from_list("mago", mago_colors)
            mago_cmap.set_bad(color="black", alpha=1.0)

            # Convert to RGB for matplotlib
            if img.ndim == 3:
                img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

            fig, ax = plt.subplots(figsize=(8, 8))
            im = ax.imshow(
                img,
                cmap=mago_cmap,
                vmin=VMIN,
                vmax=VMAX
            )

            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label(pretty_name)   # e.g. "Turbidity (NTU)"
            ax.set_title(f"{date_str}\n{pretty_name}")
            ax.axis("off")
            fig.canvas.draw()

            frame = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
            frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (4,))
            frame = cv.cvtColor(frame, cv.COLOR_RGBA2BGR)

            frame = cv.resize(frame, (w, h), interpolation=cv.INTER_AREA)

            video.write(frame)

            plt.close(fig)
            
            # Display clean progress
            print(f"  [{idx+1}/{total_images}] Added: {os.path.basename(path)}")

        video.release()
        print("Intermediate lossless video successfully created.")

        # ── High-Compatibility H.264 MP4 Re-encoding ──
        print("Re-encoding video to high-compatibility MP4 using ffmpeg...")
        final_mp4 = os.path.join(img_dir, f"{quality_variable}_evolution_{start_date}_to_{end_date}.mp4")

        cmd = [
            "ffmpeg", "-y",
            "-i", temp_avi,
            "-c:v", "libx264",       # H.264 video codec
            "-pix_fmt", "yuv420p",   # YUV 4:2:0 chroma subsampling for browser compatibility
            "-r", str(fps),          # output framerate
            final_mp4
        ]

        try:
            subprocess.run(cmd, check=True)
            print("═" * 80)
            print(f"Timelapse video successfully generated: {final_mp4}")
            print("═" * 80)
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] ffmpeg re-encoding failed with exit code {e.returncode}: {e}")
        finally:
            if os.path.exists(temp_avi):
                os.remove(temp_avi)

if __name__ == "__main__":
    main()
