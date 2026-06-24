import os
import sys
import argparse
import unicodedata
import glob
import pandas as pd

def normalize_string(s):
    if not isinstance(s, str):
        return ""
    # Normalize unicode characters to decompose them, then keep only non-diacritic characters
    nfkd_form = unicodedata.normalize('NFKD', s)
    return "".join([c for c in nfkd_form if not unicodedata.combining(c)]).lower().strip()

def main():
    parser = argparse.ArgumentParser(description="Extract Cota (elevation) CSV data for a generic Albufeira.")
    parser.add_argument(
        "-a", "--albufeira",
        required=True,
        help="Name of the albufeira (reservoir/dam) to filter for (e.g., 'Maranhão', 'Montargil')."
    )
    parser.add_argument(
        "-o", "--output",
        help="Path to the output CSV file. Defaults to 'cota_<normalized_name>.csv' in current directory."
    )
    parser.add_argument(
        "--excel-dir",
        default="data/excel",
        help="Directory where excel files are stored (default: 'data/excel')."
    )
    
    args = parser.parse_args()
    
    # 1. Load Historico Excel
    historico_path = os.path.join(args.excel_dir, "Historico_2005_2025_V15NOV2025.xlsx")
    if not os.path.exists(historico_path):
        print(f"Error: Historical Excel file not found at: {historico_path}", file=sys.stderr)
        sys.exit(1)
        
    print(f"Loading historical data from {historico_path}...")
    df_hist = pd.read_excel(historico_path, sheet_name=0)
    
    # Get all unique barragem names
    unique_names = df_hist.iloc[:, 0].dropna().unique()
    
    # Find matching name
    query_norm = normalize_string(args.albufeira)
    if not query_norm:
        print("Error: Albufeira name query is empty.", file=sys.stderr)
        sys.exit(1)
        
    exact_matches = []
    substring_matches = []
    
    for name in unique_names:
        name_norm = normalize_string(name)
        if name_norm == query_norm:
            exact_matches.append(name)
        elif query_norm in name_norm:
            substring_matches.append(name)
            
    # Determine the target albufeira name
    target_name = None
    if exact_matches:
        # If there's an exact match, use it
        target_name = exact_matches[0]
    elif len(substring_matches) == 1:
        target_name = substring_matches[0]
    elif len(substring_matches) > 1:
        print(f"Error: Multiple matches found for '{args.albufeira}':", file=sys.stderr)
        for m in substring_matches:
            print(f"  - {m}", file=sys.stderr)
        print("Please be more specific.", file=sys.stderr)
        sys.exit(1)
    else:
        print(f"Error: No matching albufeira found for '{args.albufeira}'.", file=sys.stderr)
        print("Available choices in database:", file=sys.stderr)
        # Print choices sorted alphabetically, grouped 4 per line for readability
        sorted_choices = sorted(unique_names)
        for i in range(0, len(sorted_choices), 4):
            print("  " + ", ".join(sorted_choices[i:i+4]), file=sys.stderr)
        sys.exit(1)
        
    print(f"Selected Barragem: '{target_name}'")
    
    # 2. Filter historical data
    # Filter where column 0 matches target_name
    df_hist_filtered = df_hist[df_hist.iloc[:, 0] == target_name]
    
    # Extract Data (col 1) and Cota (col 2)
    df_hist_extracted = pd.DataFrame({
        'date': pd.to_datetime(df_hist_filtered.iloc[:, 1], errors='coerce'),
        'cota': pd.to_numeric(df_hist_filtered.iloc[:, 2], errors='coerce')
    })
    
    # 3. Look for a specific Albufeira file (e.g. AlbufeirasMaranhao_18-07-2025.xlsx)
    # Match pattern: data/excel/*Albufeiras*{target_name_normalized}*.xlsx
    # target_name_norm = normalize_string(target_name)
    target_name_norm = target_name
    
    # We will search the excel directory for any xlsx files
    specific_file = None
    xlsx_files = glob.glob(os.path.join(args.excel_dir, "*.xlsx"))
    for filepath in xlsx_files:
        filename = os.path.basename(filepath)
        # Skip historical and linked data files
        if "Historico_" in filename or "linked_data" in filename:
            continue
        # Check if the filename contains "Albufeiras" and target_name_norm (normalized comparison)
        filename_norm = normalize_string(filename)
        if "albufeira" in filename_norm and target_name_norm in filename_norm:
            specific_file = filepath
            break
            
    df_spec_extracted = pd.DataFrame(columns=['date', 'cota'])
    if specific_file:
        print(f"Found specific albufeira Excel file: {specific_file}")
        try:
            df_spec = pd.read_excel(specific_file, sheet_name=0)
            # Extract Data (col 0) and Cota (col 5)
            df_spec_extracted = pd.DataFrame({
                'date': pd.to_datetime(df_spec.iloc[:, 0], errors='coerce'),
                'cota': pd.to_numeric(df_spec.iloc[:, 5], errors='coerce')
            })
            print(f"Extracted {len(df_spec_extracted)} entries from specific file.")
        except Exception as e:
            print(f"Warning: Could not read specific file {specific_file}: {e}", file=sys.stderr)
            
    # 4. Concatenate and clean data
    combined = pd.concat([df_spec_extracted, df_hist_extracted], ignore_index=True)
    
    # Ensure date column is datetimelike
    combined['date'] = pd.to_datetime(combined['date'], errors='coerce')
    
    # Drop rows where date or cota is null/invalid
    combined = combined.dropna(subset=['date', 'cota'])
    
    # Format dates as YYYY-MM-DD
    combined['date'] = combined['date'].dt.strftime('%Y-%m-%d')
    
    # Drop duplicate dates
    combined = combined.drop_duplicates(subset=['date'])
    
    # Sort chronologically by date
    combined = combined.sort_values(by='date').reset_index(drop=True)
    
    # Save to CSV
    norm_name_for_file = target_name_norm.replace(" ", "_")
    output_path = args.output if args.output else f"cota_{norm_name_for_file}.csv"
    
    # Create output directory if it doesn't exist
    out_dir = os.path.dirname(output_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
        
    combined.to_csv(output_path, index=False)
    
    print(f"\nSuccessfully generated CSV at: {os.path.abspath(output_path)}")
    print(f"Total entries: {len(combined)}")
    if not combined.empty:
        print(f"Date range: {combined['date'].min()} to {combined['date'].max()}")
        print("\nFirst 5 rows:")
        print(combined.head(5))
        print("\nLast 5 rows:")
        print(combined.tail(5))
    else:
        print("Warning: The resulting dataset is empty.")

if __name__ == "__main__":
    main()
