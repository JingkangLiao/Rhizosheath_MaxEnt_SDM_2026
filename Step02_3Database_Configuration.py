import rasterio
import glob
import os

# --- Configuration ---
# 请将这些路径修改为你的实际文件夹路径
ENV_FOLDERS = [
    r"D:\DOCTOR\Work\根鞘综述\Figures\Figure_Location&MaxEnt\c.LocationMaxEntFigure\BioclimaticVariables\WorldClim\2000-2021\version02\Bioclim_1km_avg",
    r"D:\DOCTOR\Work\根鞘综述\Figures\Figure_Location&MaxEnt\c.LocationMaxEntFigure\BioclimaticVariables\Veg_GLC_FCS",
    r"D:\DOCTOR\Work\根鞘综述\Figures\Figure_Location&MaxEnt\c.LocationMaxEntFigure\BioclimaticVariables\Soil_HWSD2"
]

def verify_raster_alignment(folders_to_check):
    """
    Verifies that all .tif files in a list of folders are perfectly aligned, including nodata values.

    Args:
        folders_to_check (list): A list of directory paths containing raster files.
    """
    all_files = []
    for folder in folders_to_check:
        files_in_folder = glob.glob(os.path.join(folder, "*.tif"))
        if not files_in_folder:
            print(f"Warning: No .tif files found in folder: {folder}")
        all_files.extend(files_in_folder)

    if not all_files:
        print("Error: No .tif files found in any of the specified directories.")
        return False, 0

    print(f"Found a total of {len(all_files)} raster files to check.\n")

    # --- Use the first file as the reference standard ---
    reference_file = all_files[0]
    try:
        with rasterio.open(reference_file) as src:
            ref_crs = src.crs
            ref_transform = src.transform
            ref_width = src.width
            ref_height = src.height
            ref_nodata = src.nodata
    except Exception as e:
        print(f"Error reading reference file {reference_file}: {e}")
        return False, len(all_files)

    print("--- Reference Standard ---")
    print(f"File: {os.path.basename(reference_file)}")
    print(f"CRS: {ref_crs}")
    print(f"Dimensions: {ref_width} x {ref_height}")
    print(f"Transform (Origin & Resolution):\n{ref_transform}")
    print(f"Nodata: {ref_nodata}")
    print("--------------------------\n")

    mismatch_found = False

    # --- Compare all other files against the reference ---
    for file_path in all_files[1:]:
        try:
            with rasterio.open(file_path) as src:
                current_crs = src.crs
                current_transform = src.transform
                current_width = src.width
                current_height = src.height
                current_nodata = src.nodata

                # Check for mismatches
                if current_crs != ref_crs:
                    print(f"MISMATCH FOUND in file: {os.path.basename(file_path)}")
                    print(f"  > Expected CRS: {ref_crs}")
                    print(f"  > Found CRS:    {current_crs}\n")
                    mismatch_found = True

                if current_transform != ref_transform:
                    print(f"MISMATCH FOUND in file: {os.path.basename(file_path)}")
                    print(f"  > Expected Transform:\n{ref_transform}")
                    print(f"  > Found Transform:   \n{current_transform}\n")
                    mismatch_found = True

                if current_width != ref_width or current_height != ref_height:
                    print(f"MISMATCH FOUND in file: {os.path.basename(file_path)}")
                    print(f"  > Expected Dimensions: {ref_width} x {ref_height}")
                    print(f"  > Found Dimensions:    {current_width} x {current_height}\n")
                    mismatch_found = True

                if current_nodata != ref_nodata:
                    print(f"MISMATCH FOUND in file: {os.path.basename(file_path)}")
                    print(f"  > Expected Nodata: {ref_nodata}")
                    print(f"  > Found Nodata:    {current_nodata}\n")
                    mismatch_found = True

        except Exception as e:
            print(f"Error reading file {os.path.basename(file_path)}: {e}")
            mismatch_found = True

    if not mismatch_found:
        print("SUCCESS! All raster files are perfectly aligned, including nodata values.")
        return True, len(all_files)
    else:
        print("FAILURE! Mismatches were found. Please re-project/resample the files listed above or fix nodata values.")
        return False, len(all_files)

if __name__ == "__main__":
    print("Starting raster alignment verification...")
    is_aligned, num_files = verify_raster_alignment(ENV_FOLDERS)
    print("\nVerification process complete.")
    if is_aligned:
        print(f"All {num_files} environmental layers are ready for analysis.")
    else:
        print("Please correct the alignment issues or nodata mismatches before proceeding to Checklist 3.")