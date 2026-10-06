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
        raise ValueError(
            "Fail CSV tidak mengandungi sebarang data."
        )

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Lajur berikut tidak ditemui: {missing_columns}"
        )

    data["image_id"] = pd.to_numeric(
        data["image_id"],
        errors="raise"
    )

    data["confidence"] = pd.to_numeric(
        data["confidence"],
        errors="raise"
    )

    data["inference_time_ms"] = pd.to_numeric(
        data["inference_time_ms"],
        errors="raise"
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


def calculate_statistics(data, total_records):
    """
    Mengira statistik data inferens yang telah ditapis.
    """

    if data.empty:
        return {
            "total_records": total_records,
            "accepted_records": 0,
            "average_confidence": 0,
            "highest_confidence": 0,
            "lowest_confidence": 0,
            "average_inference_time": 0,
            "object_counts": {}
        }

    confidence_values = data["confidence"].to_numpy()

    inference_time_values = data[
        "inference_time_ms"
    ].to_numpy()

    statistics = {
        "total_records": total_records,
        "accepted_records": len(data),
        "average_confidence": np.mean(
            confidence_values
        ),
        "highest_confidence": np.max(
            confidence_values
        ),
        "lowest_confidence": np.min(
            confidence_values
        ),
        "average_inference_time": np.mean(
            inference_time_values
        ),
        "object_counts": (
            data["object_class"]
            .value_counts()
            .to_dict()
        )
    }

    return statistics


def create_bar_chart(data, output_path):
    """
    Menghasilkan carta bar bilangan objek.
    """

    if data.empty:
        print(
            "Carta tidak dihasilkan kerana tiada data."
        )
        return False

    object_counts = data[
        "object_class"
    ].value_counts()

    plt.figure(figsize=(8, 5))

    bars = plt.bar(
        object_counts.index,
        object_counts.values,
        color=[
            "#1565C0",
            "#2E7D32",
            "#F57C00",
            "#7B1FA2"
        ]
    )

    plt.title(
        "Jumlah Pengesanan Mengikut Kelas Objek"
    )

    plt.xlabel("Kelas Objek")
    plt.ylabel("Jumlah Pengesanan")

    plt.bar_label(bars)

    plt.grid(
        axis="y",
        linestyle="--",
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300
    )

    plt.close()

    return True