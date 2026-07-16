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
    # Paths
    bands_dir = "/home/csantiago/data/sentinelhub/Bandas/Maranhão"
    seg_dir = "/home/csantiago/generated_data/segmentation_masks/Maranhão/ndwi+dem2/"
    pred_csv_path = "/home/csantiago/data/excel/Maranhão/predicted_elevation.csv"
    gt_csv_path = "/home/csantiago/data/excel/cota_Maranhão.csv"
    output_gif_path = "/home/csantiago/generated_data/maranhao_timeseries.gif"

    # Read CSVs
    df_pred = pd.read_csv(pred_csv_path)
    df_gt = pd.read_csv(gt_csv_path)
    
    df_pred['date'] = pd.to_datetime(df_pred['date'])
    df_gt['date'] = pd.to_datetime(df_gt['date'])
    
    # Filter valid ground truth points (ignoring 0.0 which are placeholders)
    df_gt_valid = df_gt[df_gt['cota'] > 0].copy()
    
    # We want dates in 2024 and 2025
    ndwi_files = glob.glob(os.path.join(bands_dir, "*.tif"))
    
    # selected_files = []
    # for f in ndwi_files:
    #     basename = os.path.basename(f)
    #     date_str = basename.split('T')[0]
    #     if date_str.startswith("2025") or date_str.startswith("2024"):
    #         date_obj = datetime.strptime(date_str, "%Y%m%d")
    #         selected_files.append((date_obj, f, basename))

    # Find all predictions in 2024 and 2025
    pred_2024_2025 = df_pred[(df_pred['date'].dt.year == 2024) | (df_pred['date'].dt.year == 2025)]
    
    # List all dates with predictions
    dates_with_pred = pred_2024_2025['date'].dt.date.unique()
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
    min_date = datetime.strptime("20240101", "%Y%m%d") #min(df_pred['date'].min(), df_gt_valid['date'].min())
    max_date = datetime.strptime("20251231", "%Y%m%d") #max(df_pred['date'].max(), df_gt_valid['date'].max())
    
    #y_min = min(df_pred['elevation'].min(), df_gt_valid['cota'].min()) - 2
    #y_max = max(df_pred['elevation'].max(), df_gt_valid['cota'].max()) + 2
    y_min = 120
    y_max = 135
                   
    temp_dir = "/tmp/maranhao_gif_frames"
    os.makedirs(temp_dir, exist_ok=True)

    # Apply a 3-point rolling median filter to remove transient outliers (e.g. single overpass segmentation errors)
    df_pred['elevation'] = df_pred['elevation'].rolling(window=3, center=True, min_periods=1).median()
    
    # Limit elevation to 130
    df_pred['elevation'] = df_pred['elevation'].clip(upper=130)

    frame_paths = []
    for i, (date_obj, f, basename) in enumerate(tqdm(selected_files)):
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # LEFT: NDWI image
        ax_img = axes[0]
        ndwi_img, cloud_pct = compute_NDWI(f)
        
        im = ax_img.imshow(ndwi_img, cmap=cmap, vmin=-1, vmax=1)
        plt.colorbar(im, ax=ax_img, fraction=0.046, pad=0.04)
        ax_img.set_title(f"NDWI: {date_obj.strftime('%Y-%m-%d')}")
        ax_img.axis("off")
        
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
        
        # RIGHT: Timeseries
        ax_ts = axes[1]
        
        current_pd_date = pd.to_datetime(date_obj)
        pred_curr = df_pred[df_pred['date'] <= current_pd_date]
        gt_curr = df_gt_valid[df_gt_valid['date'] <= current_pd_date]
        
        #ax_ts.plot(pred_curr['date'], pred_curr['elevation'], label="Predicted Elevation", color="orange", linewidth=2, linestyle='dotted')
        ax_ts.scatter(pred_curr['date'], pred_curr['elevation'], label="Predicted Elevation", color="orange", s=15, zorder=5)
        #ax_ts.scatter(gt_curr['date'], gt_curr['cota'], label="Ground Truth", color="blue", s=15, zorder=5)
        ax_ts.plot(gt_curr['date'], gt_curr['cota'], label="Ground Truth", color="green", linewidth=2)#, linestyle='dashed')

            
        ax_ts.set_xlim(min_date, max_date)
        ax_ts.set_ylim(y_min, y_max)
        
        # Vertical line for current date
        ax_ts.axvline(x=current_pd_date, color='red', linestyle='--', alpha=0.5, label='Current Date')
        
        ax_ts.set_title("Elevation Timeseries")
        ax_ts.set_xlabel("Date")
        ax_ts.set_ylabel("Elevation (m)")
        ax_ts.legend()
        plt.setp(ax_ts.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        
        frame_path = os.path.join(temp_dir, f"frame_{i:04d}.png")
        plt.savefig(frame_path)
        plt.close(fig)
        frame_paths.append(frame_path)
        
    print("Saving GIF...")
    with imageio.get_writer(output_gif_path, mode='I', duration=1) as writer:
        for fp in frame_paths:
            image = imageio.imread(fp)
            writer.append_data(image)
        
    # Save as mp4
    output_mp4_path = output_gif_path.replace('.gif', '.mp4')
    if frame_paths:
        first_frame = cv2.imread(frame_paths[0])
        height, width, _ = first_frame.shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_mp4_path, fourcc, 1.0, (width, height))
        for fp in frame_paths:
            frame = cv2.imread(fp)
            out.write(frame)
        out.release()
            
    print(f"GIF saved to {output_gif_path}")
    print(f"MP4 saved to {output_mp4_path}")

if __name__ == "__main__":
    main()
