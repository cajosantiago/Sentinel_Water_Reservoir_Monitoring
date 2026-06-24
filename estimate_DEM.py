import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import pandas as pd
from datetime import datetime
from pathlib import Path
from PIL import Image
from skimage.measure import find_contours
import unicodedata
import argparse

from estimate_elevation import segment_image, extract_border_pixels

import torch
import torch.optim as optim

def total_variation(Z):
    """Anisotropic total variation: mean of absolute differences between neighbouring pixels."""
    diff_y = torch.abs(Z[1:, :] - Z[:-1, :])   # vertical neighbours
    diff_x = torch.abs(Z[:, 1:] - Z[:, :-1])   # horizontal neighbours
    return torch.mean(diff_y) + torch.mean(diff_x)

def reconstruct_terrain(
    H, W,
    border_ys, border_xs, border_heights,
    accumulation,
    lambda_tv=5.0,
    n_iters=3000,
    lr=0.5,
    device=None,
    verbose=True,
    log_interval=500,
):
    """
    Reconstruct a dense Digital Elevation Model (DEM) from contour observations
    using PyTorch gradient descent, subject to hard boundary constraints.

    The terrain Z (shape H x W) is optimized to minimise:

        E(Z) = L_contour(Z) + lambda_tv * L_tv(Z)

    where:
        L_contour  = mean((Z[y_i, x_i] - h_i)^2)   contour consistency
        L_tv       = anisotropic total variation      smoothness regularizer

    Boundary Hard Constraints:
        - Z[accumulation == 0] = max(border_heights)
        - Z[accumulation == max_accumulation] = min(border_heights)
    These constraints are enforced exactly at each step, representing a
    projected gradient descent rather than a regularization term.

    Parameters
    ----------
    H, W : int
        Height and width of the terrain grid.
    border_ys : array-like, shape (N,)
        Row indices of all segmentation boundary pixels.
    border_xs : array-like, shape (N,)
        Column indices of all segmentation boundary pixels.
    border_heights : array-like, shape (N,)
        Known real-world elevation at each boundary pixel.
    accumulation : np.ndarray, shape (H, W)
        Water accumulation count map.
    lambda_tv : float
        Weight for the total variation regularization term.
    n_iters : int
        Number of gradient descent iterations.
    lr : float
        Learning rate for the Adam optimizer.
    device : str or torch.device, optional
        Compute device ('cpu' or 'cuda'). Auto-detected if None.
    verbose : bool
        If True, print loss every `log_interval` iterations.
    log_interval : int
        Logging frequency.

    Returns
    -------
    Z_np : np.ndarray, shape (H, W)
        Reconstructed elevation map in metres.
    loss_history : list of dict
        Per-logged-iteration dict.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    print(f"Device : {device}")
    print(f"Grid   : {H} x {W}  ({H*W:,} pixels)")
    print(f"Obs    : {len(border_ys):,} boundary pixels")
    print(f"lambda_tv={lambda_tv}, lr={lr}, n_iters={n_iters}")
    print("-" * 60)

    # ── Boundary levels from observation heights ──────────────────────
    min_elevation = float(np.min(border_heights))
    max_elevation = float(np.max(border_heights))

    # ── Set up hard constraints tensors ───────────────────────────────
    accum_tensor = torch.tensor(accumulation, dtype=torch.float32, device=device)
    zero_accum_mask = (accum_tensor == 0.0)
    max_accum_mask = (accum_tensor == accum_tensor.max())

    print(f"Hard constraints: zero accumulation = {max_elevation:.2f} m ({torch.sum(zero_accum_mask).item():,} px)")
    print(f"                 max accumulation  = {min_elevation:.2f} m ({torch.sum(max_accum_mask).item():,} px)")

    # ── Initialise learnable terrain ──────────────────────────────────
    # Start from the mean observed height so the optimiser has a warm start.
    h_mean = float(np.mean(border_heights))
    Z = torch.full((H, W), h_mean, dtype=torch.float32, device=device, requires_grad=True)

    # Initialize constrained pixels to their target values
    with torch.no_grad():
        Z[zero_accum_mask] = max_elevation
        Z[max_accum_mask] = min_elevation

    optimizer = optim.Adam([Z], lr=lr)

    # ── Observation tensors ───────────────────────────────────────────
    ys      = torch.tensor(border_ys,      dtype=torch.long,    device=device)
    xs      = torch.tensor(border_xs,      dtype=torch.long,    device=device)
    heights = torch.tensor(border_heights, dtype=torch.float32, device=device)

    loss_history = []

    # ── Optimisation loop ─────────────────────────────────────────────
    for it in range(n_iters):
        optimizer.zero_grad()

        # 1) Contour consistency: minimise squared error at boundary pixels
        z_at_contour = Z[ys, xs]
        L_contour = torch.mean((z_at_contour - heights) ** 2)

        # 2) Total variation: penalise large local elevation gradients
        L_tv = total_variation(Z)

        # 3) Total energy
        loss = L_contour + lambda_tv * L_tv

        loss.backward()
        optimizer.step()

        # Enforce hard constraints (Projected Gradient Descent step)
        with torch.no_grad():
            Z[zero_accum_mask] = max_elevation
            Z[max_accum_mask] = min_elevation

        if verbose and (it % log_interval == 0 or it == n_iters - 1):
            entry = {
                "iter":      it,
                "loss":      loss.item(),
                "L_contour": L_contour.item(),
                "L_tv":      L_tv.item(),
            }
            loss_history.append(entry)
            print(
                f"  iter {it:5d}/{n_iters-1}  "
                f"loss={entry['loss']:.4f}  "
                f"L_contour={entry['L_contour']:.4f}  "
                f"L_tv={entry['L_tv']:.4f}"
            )

    Z_np = Z.detach().cpu().numpy()
    return Z_np, loss_history

def to_dt(s):
    if isinstance(s, datetime):
        return s
    if hasattr(s, 'to_pydatetime'):
        return s.to_pydatetime()
    s = str(s).strip()
    return datetime.strptime(s[:10], "%Y-%m-%d")

# ── Main Script Entry ─────────────────────────────────────────────────────────
def sanitize_name(name):
    # Normalize unicode to remove accents (e.g. Maranhão -> Maranhao)
    return "".join(c for c in unicodedata.normalize('NFD', name) if unicodedata.category(c) != 'Mn')

def main():
    parser = argparse.ArgumentParser(
        description="Reconstruct dense Digital Elevation Model (DEM) from shoreline contours.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--albufeira",
        default="Maranhão",
        help="Name of the albufeira (reservoir)"
    )
    parser.add_argument(
        "--mask-dir",
        default=None,
        help="Input directory containing segmentation mask PNG files (defaults to generated_data/segmentation_masks/{albufeira}/ndwi)"
    )
    parser.add_argument(
        "--excel-file",
        default=None,
        help="Path to the CSV file containing historical water level data (defaults to data/excel/cota_{albufeira}.csv)"
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory to save the reconstructed DEM, SAM and plots (defaults to generated_data/DEM/{albufeira})"
    )
    parser.add_argument(
        "--lambda-tv",
        type=float,
        default=5.0,
        help="Lambda coefficient for Total Variation regularization"
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=3000,
        help="Number of optimization iterations"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.5,
        help="Learning rate for Adam optimizer"
    )
    args = parser.parse_args()

    # Set up default paths if not provided
    albufeira = args.albufeira
    sanitized_albufeira = sanitize_name(albufeira)
    
    mask_dir = args.mask_dir
    if mask_dir is None:
        mask_dir = f"generated_data/segmentation_masks/{albufeira}/ndwi"
        
    excel_file = args.excel_file
    if excel_file is None:
        # excel_file = f"data/excel/Albufeiras{sanitized_albufeira}_18-07-2025.xlsx"
        excel_file = f"data/excel/cota_{albufeira}.csv"
        
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = f"generated_data/DEM/{albufeira}"

    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "="*80)
    print("  DEM RECONSTRUCTION PIPELINE")
    print("="*80)
    print(f"  Albufeira       : {albufeira}")
    print(f"  Masks Directory : {mask_dir}")
    print(f"  Excel File      : {excel_file}")
    print(f"  Output Directory: {output_dir}")
    print("="*80 + "\n")

    # get masks filenames from mask_dir
    if not os.path.exists(mask_dir):
        print(f"[ERROR] Masks directory does not exist: {mask_dir}")
        return

    mask_paths = sorted(
        str(mask_file)
        for mask_file in Path(mask_dir).glob("*.png")
    )
    print(f"Found {len(mask_paths)} mask files in {mask_dir}.")

    if not os.path.exists(excel_file):
        print(f"[ERROR] Excel file does not exist: {excel_file}")
        return

    # Load file (Excel or CSV)
    if str(excel_file).lower().endswith(('.xlsx', '.xls')):
        df_excel = pd.read_excel(excel_file)
    else:
        try:
            df_excel = pd.read_csv(excel_file)
            if len(df_excel.columns) == 1 and ';' in str(df_excel.columns[0]):
                df_excel = pd.read_csv(excel_file, sep=';')
        except Exception as e:
            print(f"[ERROR] Failed to read CSV file {excel_file}: {e}")
            return
    print(f"Loaded data file with {len(df_excel)} rows.")

    # Parse metadata from mask file names
    results = []
    for img_path in mask_paths:
        filename = os.path.basename(img_path)
        # Discard segmentations from 2025 and 2026
        year = filename[0:4]
        if year in ["2025", "2026"]:
            continue
        # Extract date from prefix (YYYYMMDD)
        date_str = f"{year}-{filename[4:6]}-{filename[6:8]}"
        # Extract cloud cover percentage
        try:
            cloud_per = float(filename.split('_cc')[1].split('pct')[0])
        except Exception:
            cloud_per = 0.0
        results.append((img_path, date_str, cloud_per))

    df_images = pd.DataFrame(results, columns=["path", "date", "cloud"])
    print(f"Using {len(df_images)} mask files after discarding years 2025 and 2026.")

    if len(df_images) == 0:
        print("[ERROR] No valid mask files left after filter. Exiting.")
        return

    path_img = df_images["path"]
    date_img = df_images["date"]
    cloud_values = df_images["cloud"]

    # Find the date column
    date_col = None
    for col in df_excel.columns:
        col_lower = str(col).lower()
        if col_lower in ['data', 'date']:
            date_col = col
            break
            
    # Find the cota/height column
    cota_col = None
    for col in df_excel.columns:
        col_lower = str(col).lower()
        if 'cota' in col_lower or 'height' in col_lower or 'elevation' in col_lower:
            cota_col = col
            break
            
    if date_col is None or cota_col is None:
        print(f"[ERROR] Could not find date or cota columns in {excel_file}.")
        print(f"Available columns: {list(df_excel.columns)}")
        return

    # Clean the dataframe (drop NaNs in the relevant columns)
    df_excel_clean = df_excel.dropna(subset=[date_col, cota_col])
    
    date_excel = df_excel_clean[date_col].tolist()
    height_excel = df_excel_clean[cota_col].tolist()

    # relate the quota with the date
    result_list = []
    for i in range(len(date_img)):
        if cloud_values[i] < 10.0:
            img_dt = to_dt(str(date_img[i]))
            best_j = -1
            min_diff = float('inf') 
            
            for j in range(len(date_excel)):
                excel_dt = to_dt(str(date_excel[j]))
                diff = abs((img_dt - excel_dt).total_seconds())
                if diff < min_diff:
                    min_diff = diff
                    best_j = j

            if best_j != -1 and min_diff <= (3 * 86400): 
                if date_img[i] not in ['2018-08-28', '2024-10-30', '2018-08-03', '2022-04-29']: # outliers
                    result_list.append([path_img[i], height_excel[best_j], cloud_values[i]])

    result_array = np.array(result_list, dtype=object)
    print(f"Successfully matched {len(result_array)} masks with historical water levels.")

    if len(result_array) == 0:
        print("[ERROR] No matched masks found. Exiting.")
        return

    # compute accumulation map and extract border-pixel observations
    all_border_pixels = []
    accumulation = 0
    
    for path, quota, cloud in result_array:
        # load ndwi segmentation mask as binary image with ImageMagick
        ndwi_mask = np.array(Image.open(path))
        ndwi_mask = ndwi_mask > 0
        accumulation += ndwi_mask
        
        # extract border pixels
        border_xs_list, border_ys_list = extract_border_pixels(ndwi_mask)

        if len(border_xs_list) == 0:
            continue

        border_xs = np.concatenate(border_xs_list)
        border_ys = np.concatenate(border_ys_list)

        # remove non integer coordinates
        y = border_ys.astype(int)
        x = border_xs.astype(int)

        # Build final array directly
        pixels_with_quota = np.empty((len(border_xs), 4), dtype=np.float64)
        pixels_with_quota[:, 0] = y
        pixels_with_quota[:, 1] = x
        pixels_with_quota[:, 2] = quota
        pixels_with_quota[:, 3] = accumulation[y, x]

        all_border_pixels.append(pixels_with_quota)

    all_border_pixels = np.concatenate(all_border_pixels, axis=0)

    # Extract observations from the accumulated border-pixel table
    # all_border_pixels columns: [y, x, quota (m), accumulation_count]
    border_ys      = all_border_pixels[:, 0].astype(int)
    border_xs      = all_border_pixels[:, 1].astype(int)
    border_heights = all_border_pixels[:, 2].astype(np.float32)

    # Grid dimensions from accumulation map
    H, W = accumulation.shape

    print(f"Grid size            : {H} x {W}")
    print(f"Total observations   : {len(border_ys):,}")
    print(f"Elevation range      : [{border_heights.min():.2f}, {border_heights.max():.2f}] m")

    # ── Run terrain reconstruction ────────────────────────────────────────
    DEM, loss_history = reconstruct_terrain(
        H=H, W=W,
        border_ys=border_ys,
        border_xs=border_xs,
        border_heights=border_heights,
        accumulation=accumulation,
        lambda_tv=args.lambda_tv,
        n_iters=args.iters,
        lr=args.lr,
        verbose=True,
        log_interval=500,
    )

    # remove outside pixels
    DEM[accumulation == 0] = DEM.max() + 1

    dem_path = os.path.join(output_dir, "DEM.npy")
    sam_path = os.path.join(output_dir, "SAM.npy")
    np.save(dem_path, DEM)
    print(f"Reconstructed DEM saved to {dem_path}.")
    np.save(sam_path, accumulation)
    print(f"Accumulation map saved to {sam_path}.")

    # ── Visualisation and Plot Saving ─────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Reconstructed DEM
    im0 = axes[0].imshow(DEM, cmap="terrain")
    axes[0].set_title("Reconstructed DEM (m)", fontsize=13)
    axes[0].axis("off")
    plt.colorbar(im0, ax=axes[0], label="Elevation (m)")

    # Contour observations coloured by height
    sc = axes[1].scatter(
        border_xs, border_ys,
        c=border_heights, cmap="plasma",
        s=0.3, alpha=0.5,
    )
    axes[1].set_xlim(0, W)
    axes[1].set_ylim(H, 0)
    axes[1].set_title("Contour Observations\n(coloured by height)", fontsize=13)
    axes[1].set_aspect("equal")
    plt.colorbar(sc, ax=axes[1], label="Elevation (m)")

    # Water accumulation map
    im2 = axes[2].imshow(accumulation, cmap="hot")
    axes[2].set_title("Water Accumulation Map", fontsize=13)
    axes[2].axis("off")
    plt.colorbar(im2, ax=axes[2], label="Count")

    plt.tight_layout()
    results_plot_path = os.path.join(output_dir, "reconstruction_results.png")
    plt.savefig(results_plot_path, dpi=150)
    print(f"Reconstruction visualisations saved to {results_plot_path}.")
    plt.show()

    # ── Convergence curve ─────────────────────────────────────────────────
    iters      = [e["iter"]      for e in loss_history]
    total_loss = [e["loss"]      for e in loss_history]
    lc_vals    = [e["L_contour"] for e in loss_history]
    ltv_vals   = [e["L_tv"]      for e in loss_history]

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(iters, total_loss, marker="o", label="Total loss")
    ax.plot(iters, lc_vals,    marker="s", label="L_contour")
    ax.plot(iters, ltv_vals,   marker="^", label="L_tv")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.set_title("Optimisation Convergence")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    curve_plot_path = os.path.join(output_dir, "convergence_curve.png")
    plt.savefig(curve_plot_path, dpi=150)
    print(f"Convergence curve saved to {curve_plot_path}.")
    plt.show()

if __name__ == "__main__":
    main()