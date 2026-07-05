# -*- coding: utf-8 -*-
"""
Export HWSD2_LAYERS (Access .mdb) into 7 DBF tables for D1-D7.

Input:
  G:\World_SoilDatabase\HWSD2.mdb
Table:
  HWSD2_LAYERS

Output:
  G:\World_SoilDatabase\LAYER_dbf\HWSD2_LAYERS_D1.dbf
  ...
  G:\World_SoilDatabase\LAYER_dbf\HWSD2_LAYERS_D7.dbf
Plus:
  G:\World_SoilDatabase\LAYER_dbf\field_mapping.csv
"""

import os
import re
import csv
import zlib
import datetime as dt
from decimal import Decimal

import pyodbc
import dbf

# =========================
# 配置区（按需修改）
# =========================
MDB_PATH = r"G:\World_SoilDatabase\HWSD2.mdb"
OUT_DIR  = r"G:\World_SoilDatabase\LAYER_dbf"

TABLE       = "HWSD2_LAYERS"
LAYER_FIELD = "LAYER"           # 深度层字段名
KEY_FIELD   = "HWSD2_SMU_ID"    # 你已确认可用于 join 的主键字段

# 如果 DBF 太大/字段太多导致失败，强烈建议只导出必要字段：
# KEEP_COLS = [KEY_FIELD, LAYER_FIELD, "ALUM_SAT", ...]
KEEP_COLS = None  # 默认导出（几乎）全部字段；如需限制请改成列表

CHUNK_SIZE = 200_000
DBF_CODEPAGE = "cp936"  # 中文系统建议 cp936

# =========================
# ODBC type codes (common)
# =========================
ODBC_BOOL = {-7}                 # SQL_BIT
ODBC_TINYINT = {-6}              # SQL_TINYINT
ODBC_SMALLINT = {5}              # SQL_SMALLINT
ODBC_INTEGER = {4}               # SQL_INTEGER
ODBC_BIGINT = {-5}               # SQL_BIGINT

ODBC_NUMERIC = {2, 3}            # SQL_NUMERIC, SQL_DECIMAL
ODBC_FLOATS = {6, 7, 8}          # SQL_FLOAT, SQL_REAL, SQL_DOUBLE

ODBC_CHAR = {1, 12, -1, -8, -9, -10}  # SQL_CHAR, SQL_VARCHAR, SQL_LONGVARCHAR, SQL_WCHAR, SQL_WVARCHAR, SQL_WLONGVARCHAR

ODBC_DATE = {9, 91}              # (some drivers use 9), ODBC standard SQL_DATE=91
ODBC_TIME = {10, 92}             # SQL_TIME=92
ODBC_TIMESTAMP = {11, 93}        # SQL_TIMESTAMP=93

ODBC_BINARY = {-2, -3, -4}       # SQL_BINARY, SQL_VARBINARY, SQL_LONGVARBINARY


# =========================
# 工具函数
# =========================
def pick_access_driver() -> str:
    drivers = list(pyodbc.drivers())
    candidates = [
        "Microsoft Access Driver (*.mdb, *.accdb)",
        "Microsoft Access Driver (*.mdb)",
    ]
    for c in candidates:
        if c in drivers:
            return c
    raise RuntimeError(
        "未找到可用的 Access ODBC 驱动。\n"
        "当前 pyodbc.drivers() 返回：\n- " + "\n- ".join(drivers) +
        "\n\n请安装 Microsoft Access Database Engine（ACE），并确保 Python 与驱动位数一致（通常 64-bit）。"
    )

def connect_mdb(mdb_path: str) -> pyodbc.Connection:
    driver = pick_access_driver()
    conn_str = rf"DRIVER={{{driver}}};DBQ={mdb_path};"
    return pyodbc.connect(conn_str, autocommit=True)

def safe_remove_existing_dbf(path_dbf: str):
    base = os.path.splitext(path_dbf)[0]
    for ext in [".dbf", ".dbt", ".fpt", ".cpg"]:
        p = base + ext
        if os.path.exists(p):
            os.remove(p)

def sanitize_dbf_name(name: str, used: set) -> str:
    """DBF 字段名 <=10、唯一、字母开头。"""
    raw = re.sub(r"[^A-Za-z0-9_]", "_", str(name).upper()).strip("_")
    if not raw:
        raw = "F"
    if not raw[0].isalpha():
        raw = "F" + raw

    base = raw[:10]
    if base not in used:
        used.add(base)
        return base

    # 冲突：8位前缀 + 2位hash
    h = zlib.crc32(raw.encode("utf-8")) % 100
    base = (raw[:8] + f"{h:02d}")[:10]
    if base not in used:
        used.add(base)
        return base

    # 计数兜底
    for i in range(1000):
        base = (raw[:7] + f"{i:03d}")[:10]
        if base not in used:
            used.add(base)
            return base

    raise RuntimeError(f"字段名冲突无法解决：{name}")

def get_columns_metadata(conn: pyodbc.Connection, table: str):
    """通过 ODBC schema 获取字段元数据：name, data_type, type_name, column_size, decimal_digits。"""
    cur = conn.cursor()
    cols = []
    for row in cur.columns(table=table):
        cols.append({
            "name": row.column_name,
            "data_type": row.data_type,          # ODBC type code
            "type_name": (row.type_name or ""),  # driver-specific type name
            "column_size": row.column_size,
            "decimal_digits": row.decimal_digits
        })
    if not cols:
        raise RuntimeError(f"无法读取表结构：{table}。请检查表名是否正确。")
    return cols

def access_col_to_dbf_spec(src_name: str, data_type: int, type_name: str, size, dec) -> str:
    """
    将 Access/ODBC 字段映射为 DBF 字段定义。
    规则：
      - KEY_FIELD 强制 N(18,0) 便于和栅格 Value 做整数 join
      - Binary 类字段跳过（返回空字符串，外层会剔除）
      - 日期/时间：DBF 只有 D（日期），timestamp/time 会转为 C(19) 或 D（取日期）
    """
    if src_name == KEY_FIELD:
        return f"{src_name} N(18,0)"

    tname = (type_name or "").upper()
    size = int(size) if size not in (None, "") else 0
    dec  = int(dec) if dec not in (None, "") else 0

    # Binary / OLE / Attachment 之类，DBF 不适合存，直接跳过
    if data_type in ODBC_BINARY or any(k in tname for k in ["BINARY", "VARBINARY", "LONGBINARY", "OLE", "ATTACH", "IMAGE"]):
        return ""

    # Boolean
    if data_type in ODBC_BOOL or "YESNO" in tname or "BOOLEAN" in tname:
        return f"{src_name} L"

    # Date
    if data_type in ODBC_DATE:
        return f"{src_name} D"

    # Time/Timestamp：DBF 没有时间类型，最稳妥存为字符串
    if data_type in ODBC_TIME or data_type in ODBC_TIMESTAMP or "TIMESTAMP" in tname or "DATETIME" in tname:
        # 例如 2020-01-01 12:34:56，长度 19
        return f"{src_name} C(19)"

    # Integers
    if data_type in (ODBC_TINYINT | ODBC_SMALLINT | ODBC_INTEGER | ODBC_BIGINT) or any(k in tname for k in ["BYTE", "COUNTER", "AUTOINCREMENT"]):
        return f"{src_name} N(18,0)"

    # Numeric/Decimal/Float
    if data_type in (ODBC_NUMERIC | ODBC_FLOATS) or any(k in tname for k in ["DECIMAL", "NUMERIC", "DOUBLE", "FLOAT", "REAL", "SINGLE", "CURRENCY"]):
        # 小数位最多 6
        dec2 = min(max(dec, 0), 6)
        # 若没给 decimal_digits，但类型像浮点，给 6
        if dec2 == 0 and any(k in tname for k in ["DOUBLE", "FLOAT", "REAL", "SINGLE", "CURRENCY"]):
            dec2 = 6
        return f"{src_name} N(18,{dec2})"

    # Text
    if data_type in ODBC_CHAR or any(k in tname for k in ["CHAR", "TEXT", "VARCHAR", "MEMO", "LONGTEXT"]):
        width = 50 if size <= 0 else size
        width = min(max(width, 1), 254)  # DBF C 最大 254
        return f"{src_name} C({width})"

    # fallback：按字符存
    return f"{src_name} C(50)"

def parse_dbf_field_spec(spec: str):
    """
    spec 示例：
      NAME C(50)
      VAR  N(18,6)
      DATE D
      FLAG L
    返回 (field_name, field_type, width, dec)
    """
    parts = spec.strip().split()
    fname = parts[0].upper()
    fdef = parts[1].upper()
    ftype = fdef[0]  # C/N/D/L
    width = None
    dec = None
    if ftype in ("C", "N") and "(" in fdef and ")" in fdef:
        inner = fdef[fdef.find("(")+1:fdef.find(")")]
        if "," in inner:
            w, d = inner.split(",", 1)
            width, dec = int(w), int(d)
        else:
            width, dec = int(inner), 0
    return fname, ftype, width, dec

def coerce_for_dbf(value, ftype, width=None, dec=None):
    """
    按 DBF 字段类型强制转换，避免 int->string 这类报错。
    """
    if value is None:
        return "" if ftype == "C" else None

    # bytes -> str
    if isinstance(value, (bytes, bytearray)):
        try:
            value = value.decode("utf-8", errors="ignore")
        except Exception:
            value = str(value)

    if ftype == "C":
        s = str(value)
        if width is not None and len(s) > width:
            s = s[:width]
        return s

    if ftype == "L":
        # Access 可能返回 0/1 或 True/False
        if isinstance(value, str):
            return value.strip().lower() in ("1", "t", "true", "y", "yes")
        return bool(value)

    if ftype == "D":
        # date/datetime/字符串
        if isinstance(value, dt.datetime):
            return value.date()
        if isinstance(value, dt.date):
            return value
        if isinstance(value, str):
            # 尝试解析 YYYY-MM-DD
            try:
                return dt.datetime.strptime(value[:10], "%Y-%m-%d").date()
            except Exception:
                return None
        return None

    if ftype == "N":
        # Decimal/float/int
        try:
            if isinstance(value, Decimal):
                value = float(value)
            if dec is None or dec == 0:
                # 整数
                return int(round(float(value)))
            else:
                return float(value)
        except Exception:
            return None

    # fallback
    return str(value)

def export_one_layer(conn, cols_meta, layer_code: str, out_dbf: str, keep_cols=None):
    # 选择列
    all_cols = [c["name"] for c in cols_meta]
    if keep_cols is None:
        select_cols = all_cols
    else:
        missing = [c for c in keep_cols if c not in all_cols]
        if missing:
            raise RuntimeError(f"这些列在表 {TABLE} 中不存在：{missing}")
        select_cols = keep_cols

    # 构建 DBF spec（并剔除不支持字段）
    meta_map = {c["name"]: c for c in cols_meta}
    raw_specs = []
    dropped = []
    for src in select_cols:
        m = meta_map[src]
        spec = access_col_to_dbf_spec(src, m["data_type"], m["type_name"], m["column_size"], m["decimal_digits"])
        if not spec:
            dropped.append(src)
            continue
        raw_specs.append(spec)

    if dropped:
        print(f"  [Skip] {layer_code} dropped unsupported/binary fields: {len(dropped)}")

    # DBF 字段名映射（<=10）
    used = set()
    name_map = {}
    final_specs = []
    field_meta = {}  # dbf_field_name -> (ftype,width,dec)

    for spec in raw_specs:
        src_field = spec.split()[0]
        dbf_field = sanitize_dbf_name(src_field, used)
        name_map[src_field] = dbf_field

        # 把 spec 里的字段名替换成 dbf_field
        spec2 = spec.replace(src_field, dbf_field, 1)
        final_specs.append(spec2)

        fname, ftype, width, dec = parse_dbf_field_spec(spec2)
        field_meta[fname] = (ftype, width, dec)

    os.makedirs(os.path.dirname(out_dbf), exist_ok=True)
    safe_remove_existing_dbf(out_dbf)

    t = dbf.Table(out_dbf, final_specs, codepage=DBF_CODEPAGE)
    t.open(mode=dbf.READ_WRITE)

    # SQL 查询（字段名用 [] 包裹）
    src_fields_order = list(name_map.keys())
    sel = ", ".join([f"[{c}]" for c in src_fields_order])
    sql = f"SELECT {sel} FROM {TABLE} WHERE [{LAYER_FIELD}] = ?"

    cur = conn.cursor()
    cur.execute(sql, (layer_code,))

    # cursor.description 顺序与 select 一致
    src_cols = [d[0] for d in cur.description]
    dbf_cols = [name_map[c] for c in src_cols]

    total = 0
    while True:
        rows = cur.fetchmany(CHUNK_SIZE)
        if not rows:
            break

        for r in rows:
            rec = {}
            for col_name, val in zip(dbf_cols, r):
                ftype, width, dec = field_meta[col_name.upper()]
                rec[col_name] = coerce_for_dbf(val, ftype, width, dec)
            t.append(rec)
            total += 1

        if total % (CHUNK_SIZE * 2) == 0:
            print(f"  ...{layer_code} rows written: {total}")

    t.close()
    return name_map, total


def main():
    if not os.path.exists(MDB_PATH):
        raise FileNotFoundError(f"找不到 mdb：{MDB_PATH}")

    os.makedirs(OUT_DIR, exist_ok=True)

    print("Connecting:", MDB_PATH)
    conn = connect_mdb(MDB_PATH)

    cols_meta = get_columns_metadata(conn, TABLE)

    mapping_out = os.path.join(OUT_DIR, "field_mapping.csv")
    mapping_rows = []

    for i in range(1, 8):
        layer_code = f"D{i}"
        out_dbf = os.path.join(OUT_DIR, f"{TABLE}_{layer_code}.dbf")
        print(f"[Export] {layer_code} -> {out_dbf}")

        name_map, n = export_one_layer(
            conn=conn,
            cols_meta=cols_meta,
            layer_code=layer_code,
            out_dbf=out_dbf,
            keep_cols=KEEP_COLS
        )
        print(f"  rows written: {n}")

        for src, dst in name_map.items():
            mapping_rows.append((layer_code, src, dst))

    with open(mapping_out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["layer", "src_field", "dbf_field"])
        seen = set()
        for row in mapping_rows:
            if row in seen:
                continue
            seen.add(row)
            w.writerow(list(row))

    print("[OK] field mapping saved:", mapping_out)
    conn.close()
    print("[DONE]")


if __name__ == "__main__":
    main()
