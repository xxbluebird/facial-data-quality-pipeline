from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

QC_PATH = (
    PROJECT_ROOT
    / "reports"
    / "qc_face_analysis.csv"
)

EVALUATION_PATH = (
    PROJECT_ROOT
    / "reports"
    / "qc_evaluation.csv"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
)

STATUS_CHART = (
    REPORT_DIR
    / "qc_status_distribution.png"
)

REJECTION_CHART = (
    REPORT_DIR
    / "rejection_reasons.png"
)


# ============================================================
# QC STATUS DISTRIBUTION
# ============================================================

def plot_status_distribution(df):

    counts = (
        df["qc_status"]
        .value_counts()
        .reindex(
            ["ACCEPT", "REJECT"],
            fill_value=0
        )
    )

    plt.figure(
        figsize=(6, 4)
    )

    counts.plot(
        kind="bar"
    )

    plt.title(
        "Automated Facial Image QC Results"
    )

    plt.xlabel(
        "QC Status"
    )

    plt.ylabel(
        "Number of Images"
    )

    plt.xticks(
        rotation=0
    )

    plt.tight_layout()

    plt.savefig(
        STATUS_CHART,
        dpi=200
    )

    plt.close()


# ============================================================
# REJECTION REASONS
# ============================================================

def plot_rejection_reasons(df):

    rejected = df[
        df["qc_status"] == "REJECT"
    ]

    reasons = []

    for value in rejected["rejection_reason"].dropna():

        individual_reasons = [
            reason.strip()
            for reason in str(value).split(";")
            if reason.strip()
        ]

        reasons.extend(
            individual_reasons
        )

    if not reasons:

        print(
            "No rejection reasons found."
        )

        return

    reason_counts = (
        pd.Series(reasons)
        .value_counts()
    )

    plt.figure(
        figsize=(7, 4)
    )

    reason_counts.plot(
        kind="bar"
    )

    plt.title(
        "Image Rejection Reasons"
    )

    plt.xlabel(
        "Rejection Reason"
    )

    plt.ylabel(
        "Number of Images"
    )

    plt.xticks(
        rotation=30,
        ha="right"
    )

    plt.tight_layout()

    plt.savefig(
        REJECTION_CHART,
        dpi=200
    )

    plt.close()


# ============================================================
# PRINT SUMMARY
# ============================================================

def print_summary(df):

    total = len(df)

    accepted = (
        df["qc_status"] == "ACCEPT"
    ).sum()

    rejected = (
        df["qc_status"] == "REJECT"
    ).sum()

    print()
    print("=" * 60)
    print("PROJECT QC SUMMARY")
    print("=" * 60)

    print(
        f"Total images     : {total}"
    )

    print(
        f"Accepted         : {accepted}"
    )

    print(
        f"Rejected         : {rejected}"
    )

    if total > 0:

        print(
            f"Acceptance rate  : "
            f"{accepted / total * 100:.2f}%"
        )

    print()


# ============================================================
# MANUAL AGREEMENT
# ============================================================

def print_evaluation():

    if not EVALUATION_PATH.exists():

        return

    df = pd.read_csv(
        EVALUATION_PATH
    )

    correct = (
        df["evaluation"].isin(
            [
                "TRUE_ACCEPT",
                "TRUE_REJECT"
            ]
        )
    ).sum()

    total = len(df)

    agreement = (
        correct / total * 100
        if total > 0
        else 0
    )

    print(
        "Manual vs Automated QC"
    )

    print(
        f"Agreement         : "
        f"{correct}/{total} "
        f"({agreement:.2f}%)"
    )

    print()


# ============================================================
# MAIN
# ============================================================

def main():

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    if not QC_PATH.exists():

        print(
            "QC report not found:"
        )

        print(
            QC_PATH
        )

        return

    df = pd.read_csv(
        QC_PATH
    )

    print_summary(
        df
    )

    print_evaluation()

    plot_status_distribution(
        df
    )

    plot_rejection_reasons(
        df
    )

    print(
        "Charts generated:"
    )

    print(
        STATUS_CHART
    )

    print(
        REJECTION_CHART
    )


if __name__ == "__main__":
    main()
    