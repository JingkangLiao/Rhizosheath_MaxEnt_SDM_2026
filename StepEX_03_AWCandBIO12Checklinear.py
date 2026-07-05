import os
import math
import csv
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.vrt import WarpedVRT
from rasterio.warp import Resampling
from tqdm import tqdm

# ======= 输入路径（按需改）=======
BIO12_PATH = r"D:\DOCTOR\Work\根鞘综述\Figures\Figure_Location&MaxEnt\c.LocationMaxEntFigure\VariablesIn\version04_1021\BIO12.tif"
AWC_PATH   = r"D:\DOCTOR\Work\根鞘综述\Figures\Figure_Location&MaxEnt\c.LocationMaxEntFigure\VariablesIn\version04_1021\LAYERS_AWC.tif"

# ======= 输出目录（固定到你给的文件夹）=======
OUT_DIR    = r"D:\DOCTOR\Work\根鞘综述\Figures\Figure_Location&MaxEnt\c.LocationMaxEntFigure\VariablesIn\version04_1021"
os.makedirs(OUT_DIR, exist_ok=True)
OUTPUT_CSV  = os.path.join(OUT_DIR, "BIO12_AWC_samples.csv")
SCATTER_PNG = os.path.join(OUT_DIR, "BIO12_AWC_scatter.png")
STATS_TXT   = os.path.join(OUT_DIR, "BIO12_AWC_stats.txt")

# ======= 采样/数值控制 =======
SAMPLE_SIZE = 10000
BLOCK_SIZE  = 1024
EXTREME_ABS = 1e20   # |v| >= 1e20 视为无效（拦截 -3e38、3e38 等哨兵）
SEED        = 20251021  # 可复现实验

rng = np.random.default_rng(SEED)

def _valid_mask(a, b):
    am = (~np.isnan(a)) & np.isfinite(a) & (np.abs(a) < EXTREME_ABS)
    bm = (~np.isnan(b)) & np.isfinite(b) & (np.abs(b) < EXTREME_ABS)
    return am & bm

def _pearson_r(x, y):
    x = x.astype(np.float64); y = y.astype(np.float64)
    vx = x - x.mean(); vy = y - y.mean()
    denom = np.sqrt((vx*vx).sum()) * np.sqrt((vy*vy).sum())
    return np.nan if denom == 0 else float((vx*vy).sum() / denom)

def _spearman_rho(x, y):
    def rankdata(v):
        order = np.argsort(v, kind="mergesort")
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(len(v), dtype=np.float64)
        sorted_v = v[order]
        i, n = 0, len(v)
        while i < n:
            j = i + 1
            while j < n and sorted_v[j] == sorted_v[i]:
                j += 1
            avg = 0.5 * (i + j - 1)
            ranks[order[i:j]] = avg
            i = j
        return ranks
    rx = rankdata(x); ry = rankdata(y)
    return _pearson_r(rx, ry)

def _try_plot(x, y, path):
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6,6))
        plt.scatter(x, y, s=3, alpha=0.5)
        # 简单线性拟合
        m = np.polyfit(x, y, 1)
        xx = np.linspace(np.nanmin(x), np.nanmax(x), 200)
        yy = m[0]*xx + m[1]
        plt.plot(xx, yy)
        plt.xlabel("BIO12 (mm)")
        plt.ylabel("AWC (units)")
        plt.title("BIO12 vs AWC (random 10k samples)")
        plt.tight_layout()
        plt.savefig(path, dpi=180)
        plt.close()
        return True
    except Exception as e:
        print(f"（提示）无法绘图：{e}")
        return False

def main():
    with rasterio.open(BIO12_PATH) as bio:
        height, width = bio.height, bio.width
        transform, crs = bio.transform, bio.crs

        with rasterio.open(AWC_PATH) as awc_src, WarpedVRT(
            awc_src, crs=crs, transform=transform, width=width, height=height,
            resampling=Resampling.bilinear
        ) as vrt:

            # 第1遍：统计重叠有效像元总数
            ncols = (width  + BLOCK_SIZE - 1) // BLOCK_SIZE
            nrows = (height + BLOCK_SIZE - 1) // BLOCK_SIZE
            total_valid = 0
            print("扫描有效像元计数（第1遍）...")
            for r in tqdm(range(nrows)):
                for c in range(ncols):
                    xoff, yoff = c*BLOCK_SIZE, r*BLOCK_SIZE
                    w = min(BLOCK_SIZE, width  - xoff)
                    h = min(BLOCK_SIZE, height - yoff)
                    win = Window(xoff, yoff, w, h)
                    a = bio.read(1, window=win, masked=True).astype(np.float32)
                    b = vrt.read(1, window=win, masked=True).astype(np.float32)
                    if hasattr(a, "mask"): a = a.filled(np.nan)
                    if hasattr(b, "mask"): b = b.filled(np.nan)
                    total_valid += int(_valid_mask(a, b).sum())

            if total_valid == 0:
                print("没有重叠有效像元。")
                return

            K = min(SAMPLE_SIZE, total_valid)
            print(f"有效像元总数: {total_valid}，将随机抽样: {K}")
            targets = np.sort(rng.choice(total_valid, size=K, replace=False))

            # 第2遍：按 targets 均匀抽样
            samples_x = np.empty(K, dtype=np.float32)  # BIO12
            samples_y = np.empty(K, dtype=np.float32)  # AWC
            seen = 0
            t_idx = 0

            print("执行均匀随机采样（第2遍）...")
            for r in tqdm(range(nrows)):
                for c in range(ncols):
                    if t_idx >= K: break
                    xoff, yoff = c*BLOCK_SIZE, r*BLOCK_SIZE
                    w = min(BLOCK_SIZE, width  - xoff)
                    h = min(BLOCK_SIZE, height - yoff)
                    win = Window(xoff, yoff, w, h)
                    a = bio.read(1, window=win, masked=True).astype(np.float32)
                    b = vrt.read(1, window=win, masked=True).astype(np.float32)
                    if hasattr(a, "mask"): a = a.filled(np.nan)
                    if hasattr(b, "mask"): b = b.filled(np.nan)
                    mask = _valid_mask(a, b)
                    valid_idx = np.flatnonzero(mask.ravel())
                    n_valid_block = valid_idx.size
                    if n_valid_block == 0:
                        continue
                    # 命中本块的目标
                    while t_idx < K and targets[t_idx] < seen + n_valid_block:
                        local_rank = targets[t_idx] - seen
                        flat_pos = valid_idx[local_rank]
                        rr, cc = divmod(flat_pos, w)
                        samples_x[t_idx] = a[rr, cc]
                        samples_y[t_idx] = b[rr, cc]
                        t_idx += 1
                    seen += n_valid_block
                if t_idx >= K: break

            # 计算相关
            m = np.isfinite(samples_x) & np.isfinite(samples_y)
            x, y = samples_x[m], samples_y[m]
            n = x.size
            print(f"最终有效样本数: {n}")
            r_pearson = _pearson_r(x, y)
            rho_spear = _spearman_rho(x, y)

            print("\n==== 相关性结果 ====")
            print(f"Pearson r  : {r_pearson:.4f}")
            print(f"Spearman ρ : {rho_spear:.4f}")

            # 保存 CSV
            with open(OUTPUT_CSV, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["BIO12", "AWC"])
                for i in range(n):
                    w.writerow([float(x[i]), float(y[i])])
            print(f"采样点已保存：{OUTPUT_CSV}")

            # 保存简单统计
            with open(STATS_TXT, "w", encoding="utf-8") as f:
                f.write("BIO12 vs AWC correlation (random 10k samples)\n")
                f.write(f"Pearson r  : {r_pearson:.6f}\n")
                f.write(f"Spearman ρ : {rho_spear:.6f}\n")
                f.write(f"n          : {n}\n")
            print(f"统计摘要已保存：{STATS_TXT}")

            # 可选画图
            if _try_plot(x, y, SCATTER_PNG):
                print(f"散点图已保存：{SCATTER_PNG}")

if __name__ == "__main__":
    main()
