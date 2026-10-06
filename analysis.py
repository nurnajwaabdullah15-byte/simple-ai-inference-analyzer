from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REQUIRED_COLUMNS = [
    "image_id",
    "object_class",
    "confidence",
    "inference_time_ms"
]

def load_data(file_path):
    """
    Membaca data inferens daripada fail CSV.
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Fail tidak ditemui: {file_path}"
        )

    data = pd.read_csv(file_path)

    if data.empty:
        raise ValueError("Fail CSV tidak mengandungi sebarang data.")

    missing_columns = [
        column for column in REQUIRED_COLUMNS
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Lajur berikut tidak ditemui: {missing_columns}"
        )

    data["image_id"] = pd.to_numeric(
        data["image_id"], errors="raise"
    )
    data["confidence"] = pd.to_numeric(
        data["confidence"], errors="raise"
    )
    data["inference_time_ms"] = pd.to_numeric(
        data["inference_time_ms"], errors="raise"
    )

    return data

    def filter_data(data, threshold):
    """
    Menapis rekod berdasarkan nilai confidence.
    """
    if threshold < 0 or threshold > 1:
        raise ValueError(
            "Nilai confidence mesti antara 0 hingga 1."
        )

    filtered_data = data[
        data["confidence"] >= threshold
    ].copy()

    return filtered_data

    def calculate_statistics(data):
    """
    Mengira statistik asas bagi data inferens.
    """
    if data.empty:
        raise ValueError("Tiada data untuk dianalisis.")

    statistics = {
        "total_detections": len(data),
        "average_confidence": data["confidence"].mean(),
        "average_inference_time_ms": data["inference_time_ms"].mean(),
        "max_inference_time_ms": data["inference_time_ms"].max(),
        "min_inference_time_ms": data["inference_time_ms"].min()
    }

    return statistics

    def save_summary(statistics, output_path):
    """
    Menyimpan ringkasan statistik ke dalam fail CSV.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True, exist_ok=True
    )

    summary_df = pd.DataFrame(
        [statistics]
    )

    summary_df.to_csv(
        output_path, index=False
    )

    return output_path