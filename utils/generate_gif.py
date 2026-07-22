import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from datetime import datetime
import cv2
import imageio.v2 as imageio
from tqdm import tqdm
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from estimate_elevation import compute_NDWI

def main():
    start_date = "2024-01-01"
    end_date = "2026-06-30"

    # Paths
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    bands_dir = os.path.join(project_root, "data/sentinelhub/Bandas/Maranhão")
    seg_dir = os.path.join(project_root, "generated_data/segmentation_masks/Maranhão/ndwi+dem2/")
    pred_csv_path = os.path.join(project_root, "data/excel/Maranhão/predicted_elevation.csv")
    gt_csv_path = os.path.join(project_root, "data/excel/cota_Maranhão.csv")
    output_gif_path = os.path.join(project_root, "generated_data/segmentation_masks/maranhao_timeseries.gif")
    output_err_gif_path = os.path.join(project_root, "generated_data/segmentation_masks/maranhao_error_timeseries.gif")

    # Read CSVs
    df_pred = pd.read_csv(pred_csv_path)
    df_gt = pd.read_csv(gt_csv_path)
    
    df_pred['date'] = pd.to_datetime(df_pred['date'])
    df_gt['date'] = pd.to_datetime(df_gt['date'])
    
    # Filter valid ground truth points (ignoring 0.0 which are placeholders)
    df_gt_valid = df_gt[df_gt['cota'] > 0].copy()
    
    # Find all predictions within start_date and end_date
    ndwi_files = glob.glob(os.path.join(bands_dir, "*.tif"))
    pred_date_range = df_pred[(df_pred['date'] >= start_date) & (df_pred['date'] <= end_date)]
    
    # List all dates with predictions
    dates_with_pred = pred_date_range['date'].dt.date.unique()
    dates_with_pred.sort()
    
    # Add dates with predictions to selected_files
    selected_files = []
    for date_with_pred in dates_with_pred:
        date_obj = datetime.combine(date_with_pred, datetime.min.time())
        # Find corresponding tiff in ndwi_files
        search_path = os.path.join(bands_dir, str(date_with_pred.strftime("%Y%m%d")) + "*.tif")
        found_files = glob.glob(search_path)
        if found_files:
            selected_files.append((date_obj, found_files[0], os.path.basename(found_files[0])))
            
    selected_files.sort(key=lambda x: x[0])
    
    if not selected_files:
        print("No images found for 2024 and 2025.")
        return
    else:
        print("Found {} images for 2024 and 2025.".format(len(selected_files)))

    # Custom colormap
    cmap = mcolors.LinearSegmentedColormap.from_list("ndwi", ["green", "white", "blue"])
    
    # Overall plot range limits to keep the axes fixed
    min_date = datetime.strptime(start_date, "%Y-%m-%d")
    max_date = datetime.strptime(end_date, "%Y-%m-%d")
    
    y_min = 120
    y_max = 135
                   
    temp_dir = "/tmp/maranhao_gif_frames"
    temp_err_dir = "/tmp/maranhao_err_gif_frames"
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(temp_err_dir, exist_ok=True)

    # Apply a 3-point rolling median filter to remove transient outliers (e.g. single overpass segmentation errors)
    df_pred['elevation'] = df_pred['elevation'].rolling(window=3, center=True, min_periods=1).median()
    
    # Limit elevation to 130
    df_pred['elevation'] = df_pred['elevation'].clip(upper=130)

    # Find the closest GT for each date of predicted, allowing matches up to 3 days apart
    df_pred_sorted = df_pred.sort_values('date')
    df_gt_sorted = df_gt_valid.sort_values('date')
    df_merged = pd.merge_asof(
        df_pred_sorted,
        df_gt_sorted,
        on='date',
        direction='nearest',
        tolerance=pd.Timedelta(days=3)
    )
    df_merged = df_merged.dropna(subset=['cota']).copy()

    # Compute errors
    mse = ((df_merged['elevation'] - df_merged['cota']) ** 2).mean()
    mae = (df_merged['elevation'] - df_merged['cota']).abs().mean()
    print(f"\nOverall Average Error (MSE): {mse:.4f}")
    print(f"Overall Mean Absolute Error (MAE): {mae:.4f}\n")
    df_merged['error'] = (df_merged['elevation'] - df_merged['cota']).abs()

    frame_paths = []
    frame_err_paths = []
    for i, (date_obj, f, basename) in enumerate(tqdm(selected_files)):
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig_err, axes_err = plt.subplots(1, 2, figsize=(14, 6))
        
        # LEFT: NDWI image
        ax_img = axes[0]
        ax_img_err = axes_err[0]
        ndwi_img, cloud_pct = compute_NDWI(f)
        
        im = ax_img.imshow(ndwi_img, cmap=cmap, vmin=-1, vmax=1)
        plt.colorbar(im, ax=ax_img, fraction=0.046, pad=0.04)
        ax_img.set_title(f"NDWI: {date_obj.strftime('%Y-%m-%d')}")
        ax_img.axis("off")
        
        im_err = ax_img_err.imshow(ndwi_img, cmap=cmap, vmin=-1, vmax=1)
        plt.colorbar(im_err, ax=ax_img_err, fraction=0.046, pad=0.04)
        ax_img_err.set_title(f"NDWI: {date_obj.strftime('%Y-%m-%d')}")
        ax_img_err.axis("off")
        
        # Find segmentation mask
        date_str_exact = basename.split('_')[0]
        mask_search = glob.glob(os.path.join(seg_dir, f"*{date_str_exact}*.png"))
        if mask_search:
            mask_path = mask_search[0]
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                # Find contours
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in contours:
                    ax_img.plot(cnt[:, 0, 0], cnt[:, 0, 1], color='red', linewidth=1.5)
                    ax_img_err.plot(cnt[:, 0, 0], cnt[:, 0, 1], color='red', linewidth=1.5)
        
        # RIGHT: Timeseries (Elevation)
        ax_ts = axes[1]
        
        current_pd_date = pd.to_datetime(date_obj)
        pred_curr = df_pred[df_pred['date'] <= current_pd_date]
        gt_curr = df_gt_valid[df_gt_valid['date'] <= current_pd_date]
        
        ax_ts.scatter(pred_curr['date'], pred_curr['elevation'], label="Predicted Elevation", color="orange", s=15, zorder=5)
        ax_ts.plot(gt_curr['date'], gt_curr['cota'], label="Ground Truth", color="green", linewidth=2)
        
        ax_ts.set_xlim(min_date, max_date)
        ax_ts.set_ylim(y_min, y_max)
        
        # Vertical line for current date
        ax_ts.axvline(x=current_pd_date, color='red', linestyle='--', alpha=0.5, label='Current Date')
        
        ax_ts.set_title("Elevation Timeseries")
        ax_ts.set_xlabel("Date")
        ax_ts.set_ylabel("Elevation (m)")
        ax_ts.legend()
        plt.setp(ax_ts.xaxis.get_majorticklabels(), rotation=45)
        
        fig.tight_layout()
        
        frame_path = os.path.join(temp_dir, f"frame_{i:04d}.png")
        fig.savefig(frame_path)
        plt.close(fig)
        frame_paths.append(frame_path)
        
        # RIGHT: Timeseries (Error)
        ax_err = axes_err[1]
        error_curr = df_merged[df_merged['date'] <= current_pd_date]
        
        ax_err.scatter(error_curr['date'], error_curr['error'], label="Error (Pred - GT)", color="red", s=15, zorder=5)
        ax_err.plot(error_curr['date'], error_curr['error'], color="red", linewidth=1.5, alpha=0.7)
        ax_err.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        ax_err.set_xlim(min_date, max_date)
        ax_err.set_ylim(-2, 18)
        
        # Vertical line for current date
        ax_err.axvline(x=current_pd_date, color='red', linestyle='--', alpha=0.5, label='Current Date')
        
        ax_err.set_title("Elevation Error (Predicted - Ground Truth)")
        ax_err.set_xlabel("Date")
        ax_err.set_ylabel("Error (m)")
        ax_err.legend()
        plt.setp(ax_err.xaxis.get_majorticklabels(), rotation=45)
        
        fig_err.tight_layout()
        
        frame_err_path = os.path.join(temp_err_dir, f"frame_err_{i:04d}.png")
        fig_err.savefig(frame_err_path)
        plt.close(fig_err)
        frame_err_paths.append(frame_err_path)
        
    print("Saving GIFs...")
    with imageio.get_writer(output_gif_path, mode='I', duration=1) as writer:
        for fp in frame_paths:
            image = imageio.imread(fp)
            writer.append_data(image)
            
    with imageio.get_writer(output_err_gif_path, mode='I', duration=1) as writer:
        for fp in frame_err_paths:
            image_err = imageio.imread(fp)
            writer.append_data(image_err)
        
    # Save as mp4
    output_mp4_path = output_gif_path.replace('.gif', '.mp4')
    output_err_mp4_path = output_err_gif_path.replace('.gif', '.mp4')
    
    if frame_paths:
        first_frame = cv2.imread(frame_paths[0])
        height, width, _ = first_frame.shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        out = cv2.VideoWriter(output_mp4_path, fourcc, 1.0, (width, height))
        for fp in frame_paths:
            frame = cv2.imread(fp)
            out.write(frame)
        out.release()
        
        out_err = cv2.VideoWriter(output_err_mp4_path, fourcc, 1.0, (width, height))
        for fp in frame_err_paths:
            frame = cv2.imread(fp)
            out_err.write(frame)
        out_err.release()
    
    # Create a single plot with both timeseries within start_date and end_date
    fig_both, axes_both = plt.subplots(1, 2, figsize=(20, 8))
    fig_both.patch.set_facecolor("white")
    
    # Filter merged data within start_date and end_date
    df_both = df_merged[(df_merged['date'] >= start_date) & (df_merged['date'] <= end_date)]

    # Plot Elevation
    ax_elev = axes_both[0]
    ax_elev.plot(df_both['date'], df_both['elevation'], label="Predicted Elevation", color="orange", linewidth=2)
    ax_elev.plot(df_both['date'], df_both['cota'], label="Ground Truth", color="green", linewidth=2)
    ax_elev.set_title("Elevation Timeseries")
    ax_elev.set_xlabel("Date")
    ax_elev.set_ylabel("Elevation (m)")
    ax_elev.legend()
    plt.setp(ax_elev.xaxis.get_majorticklabels(), rotation=45)
    
    # Plot Error
    ax_err_plot = axes_both[1]
    ax_err_plot.scatter(df_both['date'], df_both['error'], label="Absolute Error", color="red", s=20)
    ax_err_plot.axhline(y=mae, color='black', linestyle='--', alpha=0.5)
    ax_err_plot.set_title("Elevation Error |Predicted - Ground Truth|")
    ax_err_plot.set_xlabel("Date")
    ax_err_plot.set_ylabel("Error (m)")
    ax_err_plot.legend()
    ax_err_plot.set_ylim(0, 1.5)
    plt.setp(ax_err_plot.xaxis.get_majorticklabels(), rotation=45)
    
    fig_both.tight_layout()
    output_err_path = output_gif_path.replace('.gif', '_error.png')
    fig_both.savefig(output_err_path)
    plt.close(fig_both)

    print(f"Elevation GIF saved to {output_gif_path}")
    print(f"Elevation MP4 saved to {output_mp4_path}")
    print(f"Error GIF saved to {output_err_gif_path}")
    print(f"Error MP4 saved to {output_err_mp4_path}")
    print(f"Elevation and Error plot saved to {output_err_path}")

if __name__ == "__main__":
    main()
