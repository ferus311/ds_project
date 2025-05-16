import os
from datetime import datetime
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

LOCAL_DIR = "./data_output"


def ensure_local_dir():
    os.makedirs(LOCAL_DIR, exist_ok=True)


def generate_filename(prefix="data", file_format="parquet"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(LOCAL_DIR, f"{prefix}_{timestamp}.{file_format}")


def write_to_local(data: list, prefix="data", file_format="parquet"):
    """
    Ghi dữ liệu list[dict] ra file Parquet hoặc CSV trong thư mục local.
    Args:
        data: List[dict] hoặc List[object.__dict__]
        prefix: Tiền tố tên file
        file_format: "parquet" hoặc "csv"
    """
    if not data:
        print("⚠️ No data to save.")
        return

    ensure_local_dir()
    df = pd.DataFrame(data)
    file_path = generate_filename(prefix, file_format)

    if file_format == "csv":
        df.to_csv(file_path, index=False, encoding="utf-8-sig")
        print(f"✅ Saved {len(data)} records to local CSV: {file_path}")
    elif file_format == "parquet":
        table = pa.Table.from_pandas(df)
        pq.write_table(table, file_path)
        print(f"✅ Saved {len(data)} records to local Parquet: {file_path}")
    else:
        raise ValueError(f"Unsupported file format: {file_format}")


def load_latest_from_local(prefix="data", file_format="parquet"):
    """
    Tự động tìm file mới nhất theo prefix và file format trong thư mục LOCAL_DIR, rồi load.
    Returns:
        pandas.DataFrame
    """
    ensure_local_dir()
    files = [
        f
        for f in os.listdir(LOCAL_DIR)
        if f.startswith(prefix) and f.endswith(f".{file_format}")
    ]
    if not files:
        raise FileNotFoundError(
            f"No {file_format} files found with prefix '{prefix}' in {LOCAL_DIR}"
        )

    # Sắp xếp theo timestamp trong tên file
    files.sort(reverse=True)  # latest first
    latest_file = os.path.join(LOCAL_DIR, files[0])

    if file_format == "csv":
        return pd.read_csv(latest_file)
    elif file_format == "parquet":
        return pd.read_parquet(latest_file)
    else:
        raise ValueError(f"Unsupported file format: {file_format}")


def list_all_house_list_files(prefix="data", file_format="parquet"):
    os.makedirs(LOCAL_DIR, exist_ok=True)
    files = [
        f
        for f in os.listdir(LOCAL_DIR)
        if f.startswith(prefix) and f.endswith(f".{file_format}")
    ]
    files.sort()  # Tăng dần theo thời gian (nếu tên có timestamp)
    return [os.path.join(LOCAL_DIR, f) for f in files]


def load_links_from_file(path):
    if path.endswith(".csv"):
        df = pd.read_csv(path)
    elif path.endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported file format: {path}")

    if "link" not in df.columns:
        raise ValueError(f"❌ File {path} thiếu cột 'link'")

    return df["link"].dropna().unique().tolist()
