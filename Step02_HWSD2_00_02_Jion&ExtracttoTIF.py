"失败了"

import os
import numpy as np
import pandas as pd
import pyodbc
from osgeo import gdal

# ================= 用户可修改区域 =================

MDB_PATH = r"G:\World_SoilDatabase\HWSD2.mdb"
RASTER_PATH = r"G:\World_SoilDatabase\HWSD2_RASTER\HWSD2.bil"
OUT_BASE_DIR = r"G:\World_SoilDatabase\HWSD2_RASTER\LAYER_Property_tif"

LAYERS_TO_EXPORT = ["D1"]

PROPERTIES = [
    "ALUM_SAT",
    "CEC_CLAY",
    "ECEC",
    "BSAT",
    "PH_H2O",
    "OC",
    "CLAY",
    "SAND",
    "SILT"
]

NODATA_VALUE = -9999.0

# ================================================


def load_layer_table(layer_code):
    """读取 HWSD2_LAYERS 中指定 layer 的属性表"""
    conn = pyodbc.connect(
        r"DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};"
        f"DBQ={MDB_PATH};"
    )

    sql = f"""
        SELECT HWSD2_SMU_ID, {",".join(PROPERTIES)}
        FROM HWSD2_LAYERS
        WHERE LAYER = ?
    """

    df = pd.read_sql(sql, conn, params=[layer_code])
    conn.close()

    df.set_index("HWSD2_SMU_ID", inplace=True)
    return df


def write_property_tif(base_ds, smu_array, lookup_df, field, out_tif):
    """按 SMU_ID → 属性值 逐像元查表并输出 GeoTIFF"""

    rows, cols = smu_array.shape
    out_arr = np.full((rows, cols), NODATA_VALUE, dtype=np.float32)

    smu_ids = np.unique(smu_array)
    smu_ids = smu_ids[smu_ids > 0]

    for smu_id in smu_ids:
        if smu_id not in lookup_df.index:
            continue

        val = lookup_df.at[smu_id, field]

        if pd.isna(val):
            continue

        out_arr[smu_array == smu_id] = float(val)

    driver = gdal.GetDriverByName("GTiff")
    ds_out = driver.Create(
        out_tif,
        cols,
        rows,
        1,
        gdal.GDT_Float32,
        options=["COMPRESS=LZW"]
    )

    ds_out.SetGeoTransform(base_ds.GetGeoTransform())
    ds_out.SetProjection(base_ds.GetProjection())

    band = ds_out.GetRasterBand(1)
    band.WriteArray(out_arr)
    band.SetNoDataValue(NODATA_VALUE)
    band.FlushCache()

    ds_out = None


def main():
    os.makedirs(OUT_BASE_DIR, exist_ok=True)

    base_ds = gdal.Open(RASTER_PATH)
    if base_ds is None:
        raise RuntimeError("无法打开 HWSD2.bil")

    smu_array = base_ds.GetRasterBand(1).ReadAsArray()

    for layer_code in LAYERS_TO_EXPORT:
        print(f"\n[Layer] {layer_code}")

        out_dir = os.path.join(
            OUT_BASE_DIR,
            f"LAYER_{layer_code}_Property_tif"
        )
        os.makedirs(out_dir, exist_ok=True)

        lookup_df = load_layer_table(layer_code)

        for field in PROPERTIES:
            out_tif = os.path.join(
                out_dir,
                f"{field}_{layer_code}.tif"
            )

            print(f"  -> {field}")
            write_property_tif(
                base_ds,
                smu_array,
                lookup_df,
                field,
                out_tif
            )

    base_ds = None
    print("\n完成：所有属性已成功输出为 GeoTIFF")


if __name__ == "__main__":
    main()
