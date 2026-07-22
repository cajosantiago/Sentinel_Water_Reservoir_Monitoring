import os
import glob
import cv2 as cv
import numpy as np
import subprocess
import csv
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.dates as mdates

start_date = "2015-06-01"
end_date = "2026-06-30"
img_dir = "generated_data/quality/Maranhão"
output_dir = "generated_data/quality/Maranhão_snirh"

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
    # "cyano": {"suffix": "cyano_cells_mL", "pretty": INDEX_PRETTY[3], "VMIN": 0, "VMAX": 1200},
    # "CDOM": {"suffix": "CDOM_ug_L", "pretty": INDEX_PRETTY[6], "VMIN": 0, "VMAX": 10},
    # "chla": {"suffix": "chla_ndci_mg_m3", "pretty": INDEX_PRETTY[0], "VMIN": 0, "VMAX": 100},
    # "TSS": {"suffix": "TSS_mg_L", "pretty": INDEX_PRETTY[7], "VMIN": 0, "VMAX": 10},
    "turbidity": {"suffix": "turbidity_NTU", "pretty": INDEX_PRETTY[5], "VMIN": 0, "VMAX": 10}
}

def save_match_image(band_2d, snirh_value, label, pretty_label,
                     sat_date, snirh_date, delta_days, output_dir):
    mago_colors = ["blue", "cyan", "green", "yellow", "red"]
    mago_cmap   = mcolors.LinearSegmentedColormap.from_list("mago", mago_colors)

    valid = band_2d[np.isfinite(band_2d)]
    if len(valid) == 0:
        return None

    # Remover small areas (filtro do Guilherme)
    valid_mask  = np.isfinite(band_2d)
    binary      = (valid_mask.astype(np.uint8)) * 255
    n_labels, labels_cv, stats, _ = cv.connectedComponentsWithStats(binary, connectivity=8)
    clean_mask  = np.zeros(binary.shape, dtype=np.uint8)
    for i in range(1, n_labels):
        if stats[i, cv.CC_STAT_AREA] >= 2000:
            clean_mask[labels_cv == i] = 1
    band_clean  = band_2d.copy()
    band_clean[clean_mask == 0] = np.nan

    # Escala dinâmica da cena
    valid_clean = band_clean[np.isfinite(band_clean)]
    vmin = np.nanpercentile(valid_clean, 2)
    vmax = np.nanpercentile(valid_clean, 98)

    # Tolerância
    tol_rel  = PIXEL_TOLERANCE_REL * abs(snirh_value)
    tol_abs  = PIXEL_TOLERANCE_ABS.get(label, 0.0)
    tol      = max(tol_rel, tol_abs)
    mask_sim = (np.abs(band_clean - snirh_value) <= tol) & np.isfinite(band_clean)
    pct_sim  = 100 * mask_sim.sum() / max(np.isfinite(band_clean).sum(), 1)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.patch.set_facecolor("white")

    im = axes[0].imshow(band_clean, cmap=mago_cmap, vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(im, ax=axes[0], label=pretty_label, fraction=0.046, pad=0.04)

    snirh_norm = (snirh_value - vmin) / max(vmax - vmin, 1e-9)
    snirh_norm = np.clip(snirh_norm, 0, 1)
    cbar.ax.axhline(snirh_norm, color="white", linewidth=2.5, linestyle="--")
    cbar.ax.axhline(snirh_norm, color="black",  linewidth=1.0, linestyle="--")
    cbar.ax.text(1.05, snirh_norm, f"SNIRH\n{snirh_value:.2f}",
                 transform=cbar.ax.transAxes,
                 va="center", ha="left", fontsize=8, color="black")

    axes[0].set_title(f"Satélite: {sat_date}\n{pretty_label}\n"
                      f"Escala: [{vmin:.2f}, {vmax:.2f}]", fontsize=10)
    axes[0].axis("off")

    mask_display = np.full(band_clean.shape, np.nan)
    mask_display[np.isfinite(band_clean)] = 0.0
    mask_display[mask_sim]               = 0.5

    cmap_bwr = ListedColormap(["black", "red"])
    axes[1].set_facecolor("white")
    axes[1].imshow(mask_display, cmap=cmap_bwr, vmin=0, vmax=0.5)
    axes[1].set_title(
        f"Pixels ≈ valor SNIRH  (medição: {snirh_date}, Δ{delta_days}d)\n"
        f"SNIRH: {snirh_value:.2f}  |  Tolerância: ±{tol:.2f}  |  "
        f"Vermelho = similar  |  {pct_sim:.1f}% dos pixels de água",
        fontsize=10,
    )
    axes[1].axis("off")

    fig.suptitle(f"{label.upper().replace('_', ' ')} — {sat_date}", fontsize=13, y=1.01)
    plt.tight_layout()

    fname = os.path.join(output_dir, f"{sat_date}_{label}_snirh_match.png")
    plt.savefig(fname, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  → {os.path.basename(fname)}  ({pct_sim:.1f}% pixels similares)")

    return {
        "label":        label,
        "pct_similar":  round(pct_sim, 2),
        "n_similar":    int(mask_sim.sum()),
        "snirh_value":  snirh_value,
        "tol":          round(tol, 4),
        "scene_min":    round(float(np.nanmin(valid_clean)), 4),
        "scene_max":    round(float(np.nanmax(valid_clean)), 4),
        "scene_mean":   round(float(np.nanmean(valid_clean)), 4),
        "scene_median": round(float(np.nanmedian(valid_clean)), 4),
    }

PIXEL_TOLERANCE_REL = 0.20
PIXEL_TOLERANCE_ABS = {
    "cyano": 200.0,
    "chla": 10.0,
    "TSS": 5.0,
    "turbidity": 5.0
}

def main():
    # Ensure output_dir exists
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Load SNIRH CSV
    csv_path = "data/excel/SNIRH_Maranhão.csv"
    snirh_data = []
    if os.path.exists(csv_path):
        with open(csv_path, mode="r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f, delimiter=";")
            for row in reader:
                try:
                    d = datetime.strptime(row["date"].strip(), "%Y-%m-%d").date()
                    snirh_data.append((d, row))
                except Exception:
                    continue
        print(f"Loaded {len(snirh_data)} SNIRH ground truth records.")
    else:
        print(f"[WARN] SNIRH CSV not found at '{csv_path}'. Error timeseries videos won't be generated.")

    CSV_COL_MAP = {
        "cyano": "cyano_cells_ml",
        "chla": "chla_mg_m3",
        "TSS": "tss_mg_l",
        "turbidity": "turbidity_ntu",
    }

    for quality_variable in VARIABLE_MAP:
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

        pattern = os.path.join(img_dir, f"*_{suffix}.npy")
        img_paths = sorted(glob.glob(pattern))

        filtered_paths = []
        for path in img_paths:
            filename = os.path.basename(path)
            date_str = f"{filename[0:4]}-{filename[4:6]}-{filename[6:8]}"
            if start_date <= date_str <= end_date:
                filtered_paths.append(path)
        img_paths = filtered_paths

        if not img_paths:
            print(f"[ERROR] No {pretty_name} map images found!")
            continue

        total_images = len(img_paths)
        print(f"Found {total_images} images to compile into the timelapse video.")

        # --- First Pass: Pre-calculate matched errors for the error video ---
        csv_col = CSV_COL_MAP.get(quality_variable)
        matched_errors = {}
        max_error_val = 0.0

        if csv_col and snirh_data:
            print("Pre-calculating SNIRH matching errors...")
            for path in img_paths:
                filename = os.path.basename(path)
                date_str = f"{filename[0:4]}-{filename[4:6]}-{filename[6:8]}"
                d_sat = datetime.strptime(date_str, "%Y-%m-%d").date()

                # Find closest match within +-10 days
                matches = []
                for d_csv, row in snirh_data:
                    val_str = row.get(csv_col, "")
                    if val_str is None:
                        val_str = ""
                    val_str = val_str.strip()
                    if not val_str:
                        continue
                    try:
                        val = float(val_str)
                    except ValueError:
                        continue
                    diff = abs((d_sat - d_csv).days)
                    if diff <= 10:
                        matches.append((diff, val, d_csv))
                
                if matches:
                    matches.sort(key=lambda x: x[0])
                    best_diff, snirh_val, snirh_date = matches[0]

                    # Load and clean the image using existing filter
                    band_2d = np.load(path)
                    valid_mask = np.isfinite(band_2d)
                    binary = (valid_mask.astype(np.uint8)) * 255
                    n_labels, labels_cv, stats, _ = cv.connectedComponentsWithStats(binary, connectivity=8)
                    clean_mask = np.zeros(binary.shape, dtype=np.uint8)
                    for i in range(1, n_labels):
                        if stats[i, cv.CC_STAT_AREA] >= 2000:
                            clean_mask[labels_cv == i] = 1
                    band_clean = band_2d.copy()
                    band_clean[clean_mask == 0] = np.nan
                    
                    valid_clean = band_clean[np.isfinite(band_clean)]
                    if len(valid_clean) > 0:
                        smallest_err = float(np.nanmin(np.abs(valid_clean - snirh_val)))
                        average_err = float(np.nanmean(np.abs(valid_clean - snirh_val)))
                        days_diff = abs((d_sat - snirh_date).days)
                        
                        matched_errors[path] = {
                            "smallest_err": smallest_err,
                            "average_err": average_err,
                            "snirh_val": snirh_val,
                            "snirh_date": snirh_date,
                            "days_diff": days_diff
                        }
                        max_error_val = max(max_error_val, smallest_err, average_err)

        has_error_video = len(matched_errors) > 0
        if has_error_video:
            print(f"Found {len(matched_errors)} matches out of {total_images} images for {quality_variable}. Max error: {max_error_val:.4f}")
            # Ignore images that are not matched to a snirh gt data
            img_paths = [path for path in img_paths if path in matched_errors]
            total_images = len(img_paths)
            print(f"Filtered to {total_images} matched images for video generation.")
        else:
            print(f"No SNIRH ground truth matches for {quality_variable} (or no valid measurements). Skipping evolution and error timeseries videos.")
            continue

        # Read the first image to establish the base resolution
        first_img = np.load(img_paths[0])
        h, w = first_img.shape
        w = w if w % 2 == 0 else w + 1
        h = h if h % 2 == 0 else h + 1
        print(f"Base video resolution: {w} x {h}")

        # --- Generate Evolution Timelapse Video ---
        temp_avi = "temp_lossless.avi"
        fourcc = cv.VideoWriter_fourcc(*'MJPG')
        fps = 2.0
        video = cv.VideoWriter(temp_avi, fourcc, fps, (w, h))

        temp_error_avi = "temp_error_lossless.avi"
        video_error = cv.VideoWriter(temp_error_avi, fourcc, fps, (w, h))

        mago_colors = ["blue", "cyan", "green", "yellow", "red"]
        mago_cmap = mcolors.LinearSegmentedColormap.from_list("mago", mago_colors)
        mago_cmap.set_bad(color="black", alpha=1.0)
        
        plotted_dates = []
        plotted_smallest = []
        plotted_average = []
        plotted_days_diff = []

        for idx, path in enumerate(img_paths):
            filename = os.path.basename(path)
            date_str = f"{filename[0:4]}-{filename[4:6]}-{filename[6:8]}"
            d_sat = datetime.strptime(date_str, "%Y-%m-%d").date()
            img = np.load(path)

            fig, ax = plt.subplots(figsize=(8, 8))
            im = ax.imshow(img, cmap=mago_cmap, vmin=VMIN, vmax=VMAX)
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label(pretty_name)
            ax.set_title(f"{date_str}\n{pretty_name}")
            ax.axis("off")
            fig.canvas.draw()
            frame = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(fig.canvas.get_width_height()[::-1] + (4,))
            frame = cv.cvtColor(frame, cv.COLOR_RGBA2BGR)
            video.write(cv.resize(frame, (w, h), interpolation=cv.INTER_AREA))
            plt.close(fig)

            # Error frame
            if path in matched_errors:
                info = matched_errors[path]
                plotted_dates.append(d_sat)
                plotted_smallest.append(info["smallest_err"])
                plotted_average.append(info["average_err"])
                plotted_days_diff.append(info["days_diff"])

            fig_err, ax_err = plt.subplots(figsize=(8, 8))
            ax_err.set_facecolor("#fcfcfc")
            fig_err.patch.set_facecolor("white")
            ax_err.grid(True, linestyle="--", alpha=0.6, color="#cccccc")
            ax_err_twin = ax_err.twinx()

            if plotted_dates:
                ax_err.plot(plotted_dates, plotted_smallest, color="#1f77b4", marker="o", linewidth=2.5, label="Smallest Error")
                ax_err.plot(plotted_dates, plotted_average, color="#ff7f0e", marker="s", linewidth=2.5, label="Average Error")
                ax_err_twin.plot(plotted_dates, plotted_days_diff, color="#2ca02c", linestyle="--", marker="d", linewidth=1.5, label="Date Difference (Days)")
                ax_err.axvline(d_sat, color="red", linestyle=":", alpha=0.8)
            
            ax_err.set_xlim(datetime.strptime(start_date, "%Y-%m-%d").date(), datetime.strptime(end_date, "%Y-%m-%d").date())
            y_max_limit = max_error_val * 1.1 if max_error_val > 0 else 10.0
            ax_err.set_ylim(0, y_max_limit)
            ax_err_twin.set_ylim(0, 11)
            ax_err_twin.set_ylabel("Difference (Days)", color="#2ca02c", fontsize=11)
            ax_err_twin.tick_params(axis='y', labelcolor="#2ca02c")
            ax_err.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
            plt.xticks(rotation=45)
            ax_err.set_xlabel("Date", fontsize=11)
            ax_err.set_ylabel("Absolute Error", fontsize=11)
            ax_err.set_title(f"{pretty_name}\nError to SNIRH Ground Truth over Time", fontsize=12)
            if plotted_dates:
                lines1, labels1 = ax_err.get_legend_handles_labels()
                lines2, labels2 = ax_err_twin.get_legend_handles_labels()
                ax_err.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=10)
            plt.tight_layout()
            
            fig_err.canvas.draw()
            frame_err = np.frombuffer(fig_err.canvas.buffer_rgba(), dtype=np.uint8).reshape(fig_err.canvas.get_width_height()[::-1] + (4,))
            video_error.write(cv.resize(cv.cvtColor(frame_err, cv.COLOR_RGBA2BGR), (w, h), interpolation=cv.INTER_AREA))
            plt.close(fig_err)
            print(f"  [{idx+1}/{total_images}] Added: {os.path.basename(path)}")

        video.release()
        video_error.release()

        # --- Save static comparison image: error timeseries + days difference ---
        if plotted_dates:
            fig_cmp, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(18, 6))
            fig_cmp.patch.set_facecolor("white")

            # Left: Error timeseries
            ax_left.set_facecolor("#fcfcfc")
            ax_left.grid(True, linestyle="--", alpha=0.6, color="#cccccc")
            ax_left.plot(plotted_dates, plotted_smallest, color="#1f77b4", marker="o", linewidth=2.5, label="Smallest Error")
            ax_left.plot(plotted_dates, plotted_average, color="#ff7f0e", marker="s", linewidth=2.5, label="Average Error")
            ax_left.set_xlim(datetime.strptime(start_date, "%Y-%m-%d").date(), datetime.strptime(end_date, "%Y-%m-%d").date())
            y_max_limit = max_error_val * 1.1 if max_error_val > 0 else 10.0
            ax_left.set_ylim(0, y_max_limit)
            ax_left.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
            ax_left.tick_params(axis='x', rotation=45)
            ax_left.set_xlabel("Date", fontsize=11)
            ax_left.set_ylabel("Absolute Error", fontsize=11)
            ax_left.set_title(f"{pretty_name}\nError to SNIRH Ground Truth", fontsize=12)
            ax_left.legend(loc="upper right", fontsize=10)

            # Right: Days difference
            ax_right.set_facecolor("#fcfcfc")
            ax_right.grid(True, linestyle="--", alpha=0.6, color="#cccccc")
            ax_right.plot(plotted_dates, plotted_days_diff, color="#2ca02c", marker="d", linewidth=2.0, label="Days Difference")
            ax_right.set_xlim(datetime.strptime(start_date, "%Y-%m-%d").date(), datetime.strptime(end_date, "%Y-%m-%d").date())
            ax_right.set_ylim(0, 11)
            ax_right.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
            ax_right.tick_params(axis='x', rotation=45)
            ax_right.set_xlabel("Date", fontsize=11)
            ax_right.set_ylabel("Days Between GT and Prediction", fontsize=11)
            ax_right.set_title(f"{pretty_name}\nTemporal Distance to SNIRH Measurement", fontsize=12)
            ax_right.legend(loc="upper right", fontsize=10)

            plt.tight_layout()
            cmp_fname = os.path.join(output_dir, f"{quality_variable}_error_and_days_{start_date}_to_{end_date}.png")
            fig_cmp.savefig(cmp_fname, dpi=150, bbox_inches="tight", facecolor="white")
            plt.close(fig_cmp)
            print(f"  → Saved comparison image: {os.path.basename(cmp_fname)}")

            # print date of points with error higher than 150
            high_error_dates = np.array(plotted_dates)[np.array(plotted_average) > 150]
            print(f"Dates with error > 150: {high_error_dates}")

        # Re-encode
        for temp, final in [(temp_avi, f"{quality_variable}_evolution_{start_date}_to_{end_date}.mp4"),
                            (temp_error_avi, f"{quality_variable}_error_evolution_{start_date}_to_{end_date}.mp4")]:
            subprocess.run(["ffmpeg", "-y", "-i", temp, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", str(fps), os.path.join(output_dir, final)], check=True)
            os.remove(temp)
        

if __name__ == "__main__":
    main()
