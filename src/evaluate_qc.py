from pathlib import Path

import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

AUTO_QC_PATH = (
    PROJECT_ROOT
    / "reports"
    / "qc_face_analysis.csv"
)

MANUAL_LABEL_PATH = (
    PROJECT_ROOT
    / "metadata"
    / "manual_labels.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "qc_evaluation.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    automated = pd.read_csv(
        AUTO_QC_PATH
    )

    manual = pd.read_csv(
        MANUAL_LABEL_PATH
    )

    return automated, manual


# ============================================================
# MERGE AUTOMATED + MANUAL LABELS
# ============================================================

def merge_labels(automated, manual):

    merged = automated.merge(
        manual,
        on="filename",
        how="inner"
    )

    return merged


# ============================================================
# CLASSIFY RESULT
# ============================================================

def classify_result(row):

    manual = row["manual_status"]
    automated = row["qc_status"]

    if (
        manual == "ACCEPT"
        and automated == "ACCEPT"
    ):
        return "TRUE_ACCEPT"

    if (
        manual == "REJECT"
        and automated == "REJECT"
    ):
        return "TRUE_REJECT"

    if (
        manual == "REJECT"
        and automated == "ACCEPT"
    ):
        return "FALSE_ACCEPT"

    if (
        manual == "ACCEPT"
        and automated == "REJECT"
    ):
        return "FALSE_REJECT"

    return "UNKNOWN"


# ============================================================
# MAIN
# ============================================================

def main():

    automated, manual = load_data()

    merged = merge_labels(
        automated,
        manual
    )

    merged["evaluation"] = merged.apply(
        classify_result,
        axis=1
    )

    # --------------------------------------------------------
    # COUNTS
    # --------------------------------------------------------

    true_accept = (
        merged["evaluation"]
        == "TRUE_ACCEPT"
    ).sum()

    true_reject = (
        merged["evaluation"]
        == "TRUE_REJECT"
    ).sum()

    false_accept = (
        merged["evaluation"]
        == "FALSE_ACCEPT"
    ).sum()

    false_reject = (
        merged["evaluation"]
        == "FALSE_REJECT"
    ).sum()

    total = len(merged)

    correct = (
        true_accept
        + true_reject
    )

    accuracy = (
        correct / total
        if total > 0
        else 0
    )

    # --------------------------------------------------------
    # REJECT DETECTION METRICS
    #
    # Here:
    # REJECT = positive class
    # ACCEPT = negative class
    # --------------------------------------------------------

    precision_reject = (
        true_reject
        / (true_reject + false_reject)
        if (true_reject + false_reject) > 0
        else 0
    )

    recall_reject = (
        true_reject
        / (true_reject + false_accept)
        if (true_reject + false_accept) > 0
        else 0
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    merged.to_csv(
        OUTPUT_PATH,
        index=False
    )

    # --------------------------------------------------------
    # PRINT SUMMARY
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("AUTOMATED QC EVALUATION")
    print("=" * 60)

    print(
        f"Total evaluated : {total}"
    )

    print()

    print(
        f"True Accept     : {true_accept}"
    )

    print(
        f"True Reject     : {true_reject}"
    )

    print(
        f"False Accept    : {false_accept}"
    )

    print(
        f"False Reject    : {false_reject}"
    )

    print()

    print(
        f"Accuracy        : "
        f"{accuracy * 100:.2f}%"
    )

    print(
        f"Reject Precision: "
        f"{precision_reject * 100:.2f}%"
    )

    print(
        f"Reject Recall   : "
        f"{recall_reject * 100:.2f}%"
    )

    print()

    # --------------------------------------------------------
    # SHOW DISAGREEMENTS
    # --------------------------------------------------------

    disagreements = merged[
        merged["evaluation"].isin(
            [
                "FALSE_ACCEPT",
                "FALSE_REJECT"
            ]
        )
    ]

    if len(disagreements) > 0:

        print("DISAGREEMENTS")
        print("-" * 60)

        for _, row in disagreements.iterrows():

            print(
                f"File              : "
                f"{row['filename']}"
            )

            print(
                f"Manual status     : "
                f"{row['manual_status']}"
            )

            print(
                f"Automated status  : "
                f"{row['qc_status']}"
            )

            print(
                f"Manual reason     : "
                f"{row['manual_reason']}"
            )

            print(
                f"Automated reason  : "
                f"{row['rejection_reason']}"
            )

            print()

    else:

        print(
            "No disagreements found."
        )

    print("=" * 60)

    print(
        "Detailed evaluation saved to:"
    )

    print(
        OUTPUT_PATH
    )


if __name__ == "__main__":
    main()