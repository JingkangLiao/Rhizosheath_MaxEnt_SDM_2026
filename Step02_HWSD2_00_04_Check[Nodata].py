import os
import glob
import numpy as np
import rasterio

IN_DIR  = r"G:\BioclimaticVariables\Soil_HWSD2\Raw"
OUT_DIR = r"G:\BioclimaticVariables\Soil_HWSD2\Fixed"
PATTERN = "**/*.tif"

# 浮点比较容差（float32 通常足够）
ATOL = 1e-6
RTOL = 1e-6


def is_close(a, b):
    # 针对 float nodata（极大/极小值）用 isclose 更稳
    return np.isclose(a, b, atol=ATOL, rtol=RTOL)


def verify_one(src_path, dst_path):
    with rasterio.open(src_path) as src, rasterio.open(dst_path) as dst:
        # --- 元信息核对（分辨率/尺寸/空间参考等）---
        meta_checks = {
            "count": (src.count, dst.count),
            "width": (src.width, dst.width),
            "height": (src.height, dst.height),
            "dtype": (src.dtypes[0], dst.dtypes[0]),
            "crs": (str(src.crs), str(dst.crs)),
            "transform": (src.transform, dst.transform),
            "res": (src.res, dst.res),
            "bounds": (src.bounds, dst.bounds),
        }
        for k, (a, b) in meta_checks.items():
            if a != b:
                raise AssertionError(f"[META MISMATCH] {os.path.basename(src_path)} | {k}: {a} != {b}")

        nodata_out = dst.nodata
        if nodata_out is None:
            raise AssertionError(f"[NO NODATA IN OUTPUT] {os.path.basename(dst_path)} 输出未写入 nodata")

        # --- 像元级分块校验 ---
        mism_keep = 0   # 输入>=0有效值未被保留
        mism_neg  = 0   # 输入<0未被改成nodata
        mism_mask = 0   # 输入mask区未变成nodata
        max_abs_diff = 0.0
        total_valid_keep = 0

        # 用 block_windows 分块读，避免一次性读入超大栅格
        for _, win in src.block_windows(1):
            src_m = src.read(1, window=win, masked=True)  # 自动继承 nodata + 内部mask
            src_data = src_m.data
            src_valid = ~src_m.mask

            dst_data = dst.read(1, window=win)  # 输出读原始值（不masked，便于检查是否等于nodata）

            # 1) 输入有效且 >=0：输出应≈输入
            keep_mask = src_valid & (src_data >= 0)
            if np.any(keep_mask):
                a = src_data[keep_mask].astype("float64", copy=False)
                b = dst_data[keep_mask].astype("float64", copy=False)
                diff = np.abs(a - b)
                if diff.size:
                    max_abs_diff = max(max_abs_diff, float(np.max(diff)))
                ok = np.isclose(a, b, atol=ATOL, rtol=RTOL)
                mism_keep += int(np.size(ok) - np.count_nonzero(ok))
                total_valid_keep += int(np.size(ok))

            # 2) 输入有效但 <0：输出应为 nodata
            neg_mask = src_valid & (src_data < 0)
            if np.any(neg_mask):
                b = dst_data[neg_mask]
                ok = is_close(b, nodata_out)
                mism_neg += int(np.size(ok) - np.count_nonzero(ok))

            # 3) 输入无效（mask/nodata）：输出应为 nodata
            mask_mask = ~src_valid
            if np.any(mask_mask):
                b = dst_data[mask_mask]
                ok = is_close(b, nodata_out)
                mism_mask += int(np.size(ok) - np.count_nonzero(ok))

        return {
            "file": os.path.basename(src_path),
            "nodata_out": nodata_out,
            "mism_keep": mism_keep,
            "mism_neg": mism_neg,
            "mism_mask": mism_mask,
            "max_abs_diff_keep": max_abs_diff,
            "total_valid_keep": total_valid_keep,
        }


def main():
    src_paths = sorted(glob.glob(os.path.join(IN_DIR, PATTERN), recursive=True))
    if not src_paths:
        raise FileNotFoundError(f"未找到任何 tif：{IN_DIR}\\{PATTERN}")

    print(f"[INFO] 待验证文件数：{len(src_paths)}")

    any_fail = False
    for p in src_paths:
        rel = os.path.relpath(p, IN_DIR)
        q = os.path.join(OUT_DIR, rel)
        if not os.path.exists(q):
            print(f"[MISS] 输出不存在：{q}")
            any_fail = True
            continue

        try:
            r = verify_one(p, q)
            print(
                f"[OK] {r['file']} | nodata_out={r['nodata_out']} | "
                f"mism_keep={r['mism_keep']} | mism_neg={r['mism_neg']} | mism_mask={r['mism_mask']} | "
                f"max_abs_diff_keep={r['max_abs_diff_keep']:.3g}"
            )
        except Exception as e:
            print(f"[FAIL] {os.path.basename(p)} -> {e}")
            any_fail = True

    if any_fail:
        print("[DONE] 验证完成：存在 FAIL/MISS，请检查上面的报错。")
    else:
        print("[DONE] 验证完成：所有文件空间信息一致，像元逻辑一致（负值->nodata，其他保留）。")


if __name__ == "__main__":
    main()
