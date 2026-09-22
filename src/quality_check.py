from pathlib import Path

import cv2
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DIR = PROJECT_ROOT / "data" / "raw"
REPORT_DIR = PROJECT_ROOT / "reports"

OUTPUT_CSV = REPORT_DIR / "qc_face_analysis.csv"


# ============================================================
# PILOT QC THRESHOLDS
# ============================================================
# Thresholds are based on the current pilot dataset.
# They are NOT universal facial-image quality standards.

MIN_FACE_AREA_RATIO = 0.05
MIN_FACE_BRIGHTNESS = 70
MIN_FACE_BLUR_SCORE = 8


# ============================================================
# FACE DETECTOR
# ============================================================

FACE_CASCADE_PATH = (
    cv2.data.haarcascades
    + "haarcascade_frontalface_default.xml"
)

FACE_CASCADE = cv2.CascadeClassifier(
    FACE_CASCADE_PATH
)

if FACE_CASCADE.empty():
    raise RuntimeError(
        f"Failed to load Haar Cascade from:\n{FACE_CASCADE_PATH}"
    )


# ============================================================
# IMAGE QUALITY FUNCTIONS
# ============================================================

def calculate_brightness(gray_image):
    """
    Calculate mean grayscale intensity.

    Approximate range:
    0   = completely black
    255 = completely white
    """
    return float(gray_image.mean())


def calculate_blur_score(gray_image):
    """
    Estimate image sharpness using Variance of Laplacian.

    Lower score  = blurrier
    Higher score = sharper

    Threshold is dataset/camera dependent.
    """
    laplacian = cv2.Laplacian(
        gray_image,
        cv2.CV_64F
    )

    return float(laplacian.var())


# ============================================================
# FACE DETECTION
# ============================================================

def detect_faces(gray_image):
    """
    Detect faces using OpenCV Haar Cascade.
    """

    faces = FACE_CASCADE.detectMultiScale(
        gray_image,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(80, 80)
    )

    return faces


def get_largest_face(faces):
    """
    Return the largest detected face.

    If no face is detected, return None.
    """

    if len(faces) == 0:
        return None

    return max(
        faces,
        key=lambda face: face[2] * face[3]
    )


# ============================================================
# PROCESS ONE IMAGE
# ============================================================

def analyze_image(image_path):

    image = cv2.imread(
        str(image_path)
    )

    # --------------------------------------------------------
    # FAILED IMAGE READ
    # --------------------------------------------------------

    if image is None:

        return {
            "filename": image_path.name,
            "read_status": "ERROR",
            "width": None,
            "height": None,
            "global_brightness": None,
            "global_blur_score": None,
            "face_count": None,
            "face_detected": False,
            "face_width": None,
            "face_height": None,
            "face_area_ratio": None,
            "face_brightness": None,
            "face_blur_score": None,
        }

    # --------------------------------------------------------
    # IMAGE DIMENSIONS
    # --------------------------------------------------------

    height, width = image.shape[:2]

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # --------------------------------------------------------
    # GLOBAL IMAGE QUALITY
    # --------------------------------------------------------

    global_brightness = calculate_brightness(
        gray
    )

    global_blur_score = calculate_blur_score(
        gray
    )

    # --------------------------------------------------------
    # FACE DETECTION
    # --------------------------------------------------------

    faces = detect_faces(
        gray
    )

    face_count = len(faces)

    largest_face = get_largest_face(
        faces
    )

    # Default values

    face_detected = False

    face_width = None
    face_height = None

    face_area_ratio = None

    face_brightness = None
    face_blur_score = None

    # --------------------------------------------------------
    # FACE-SPECIFIC QUALITY
    # --------------------------------------------------------

    if largest_face is not None:

        face_detected = True

        x, y, w, h = largest_face

        face_width = int(w)
        face_height = int(h)

        # How much of the total image is occupied by the face
        face_area = w * h
        image_area = width * height

        face_area_ratio = (
            face_area / image_area
        )

        # Crop only detected face
        face_gray = gray[
            y:y + h,
            x:x + w
        ]

        face_brightness = calculate_brightness(
            face_gray
        )

        face_blur_score = calculate_blur_score(
            face_gray
        )

    # --------------------------------------------------------
    # RETURN RAW METRICS
    # --------------------------------------------------------

    return {
        "filename":
            image_path.name,

        "read_status":
            "OK",

        "width":
            width,

        "height":
            height,

        "global_brightness":
            round(
                global_brightness,
                2
            ),

        "global_blur_score":
            round(
                global_blur_score,
                2
            ),

        "face_count":
            face_count,

        "face_detected":
            face_detected,

        "face_width":
            face_width,

        "face_height":
            face_height,

        "face_area_ratio":
            (
                round(
                    face_area_ratio,
                    4
                )
                if face_area_ratio is not None
                else None
            ),

        "face_brightness":
            (
                round(
                    face_brightness,
                    2
                )
                if face_brightness is not None
                else None
            ),

        "face_blur_score":
            (
                round(
                    face_blur_score,
                    2
                )
                if face_blur_score is not None
                else None
            ),
    }


# ============================================================
# AUTOMATED QC DECISION
# ============================================================

def evaluate_quality(result):
    """
    Produce automated ACCEPT / REJECT decision.

    Current thresholds are pilot thresholds derived from
    the current controlled test images.
    """

    rejection_reasons = []

    # --------------------------------------------------------
    # IMAGE READ
    # --------------------------------------------------------

    if result["read_status"] != "OK":

        return (
            "REJECT",
            "image_read_error"
        )

    # --------------------------------------------------------
    # FACE MUST BE DETECTED
    # --------------------------------------------------------

    if not result["face_detected"]:

        return (
            "REJECT",
            "no_face_detected"
        )

    # --------------------------------------------------------
    # EXPECT EXACTLY ONE FACE
    # --------------------------------------------------------

    if result["face_count"] != 1:

        rejection_reasons.append(
            "invalid_face_count"
        )

    # --------------------------------------------------------
    # FACE SIZE
    # --------------------------------------------------------

    if (
        result["face_area_ratio"] is not None
        and
        result["face_area_ratio"] < MIN_FACE_AREA_RATIO
    ):

        rejection_reasons.append(
            "face_too_small"
        )

    # --------------------------------------------------------
    # UNDEREXPOSURE
    # --------------------------------------------------------

    if (
        result["face_brightness"] is not None
        and
        result["face_brightness"] < MIN_FACE_BRIGHTNESS
    ):

        rejection_reasons.append(
            "underexposed"
        )

    # --------------------------------------------------------
    # BLUR
    # --------------------------------------------------------

    if (
        result["face_blur_score"] is not None
        and
        result["face_blur_score"] < MIN_FACE_BLUR_SCORE
    ):

        rejection_reasons.append(
            "blurry"
        )

    # --------------------------------------------------------
    # FINAL DECISION
    # --------------------------------------------------------

    if rejection_reasons:

        return (
            "REJECT",
            "; ".join(rejection_reasons)
        )

    return (
        "ACCEPT",
        ""
    )


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Supported image extensions

    image_extensions = {
        ".jpg",
        ".jpeg",
        ".png"
    }

    # Search all images in raw directory

    image_files = sorted(
        [
            path
            for path in RAW_DIR.iterdir()
            if path.suffix.lower()
            in image_extensions
        ]
    )

    # --------------------------------------------------------
    # NO IMAGES FOUND
    # --------------------------------------------------------

    if not image_files:

        print()
        print("No images found in:")
        print(RAW_DIR)
        print()

        return

    print()
    print(
        f"Found {len(image_files)} images."
    )
    print()

    results = []

    # --------------------------------------------------------
    # PROCESS EACH IMAGE
    # --------------------------------------------------------

    for image_path in image_files:

        result = analyze_image(
            image_path
        )

        qc_status, rejection_reason = evaluate_quality(
            result
        )

        result["qc_status"] = qc_status

        result["rejection_reason"] = (
            rejection_reason
        )

        results.append(
            result
        )

        # ----------------------------------------------------
        # TERMINAL OUTPUT
        # ----------------------------------------------------

        print(
            result["filename"]
        )

        print(
            f"  Resolution        : "
            f"{result['width']} x "
            f"{result['height']}"
        )

        print(
            f"  Global brightness : "
            f"{result['global_brightness']}"
        )

        print(
            f"  Global blur       : "
            f"{result['global_blur_score']}"
        )

        print(
            f"  Faces detected    : "
            f"{result['face_count']}"
        )

        print(
            f"  Face detected     : "
            f"{result['face_detected']}"
        )

        print(
            f"  Face area ratio   : "
            f"{result['face_area_ratio']}"
        )

        print(
            f"  Face brightness   : "
            f"{result['face_brightness']}"
        )

        print(
            f"  Face blur         : "
            f"{result['face_blur_score']}"
        )

        print(
            f"  QC Status         : "
            f"{result['qc_status']}"
        )

        print(
            f"  Rejection reason  : "
            f"{result['rejection_reason']}"
        )

        print()

    # --------------------------------------------------------
    # SAVE REPORT
    # --------------------------------------------------------

    df = pd.DataFrame(
        results
    )

    df.to_csv(
        OUTPUT_CSV,
        index=False
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    total_images = len(df)

    accepted = (
        df["qc_status"] == "ACCEPT"
    ).sum()

    rejected = (
        df["qc_status"] == "REJECT"
    ).sum()

    acceptance_rate = (
        accepted / total_images * 100
        if total_images > 0
        else 0
    )

    print("=" * 60)

    print(
        "FACIAL IMAGE QUALITY CONTROL SUMMARY"
    )

    print("=" * 60)

    print(
        f"Total images       : {total_images}"
    )

    print(
        f"Accepted           : {accepted}"
    )

    print(
        f"Rejected           : {rejected}"
    )

    print(
        f"Acceptance rate    : "
        f"{acceptance_rate:.2f}%"
    )

    print()

    print(
        "Pilot thresholds:"
    )

    print(
        f"  Minimum face area ratio : "
        f"{MIN_FACE_AREA_RATIO}"
    )

    print(
        f"  Minimum face brightness : "
        f"{MIN_FACE_BRIGHTNESS}"
    )

    print(
        f"  Minimum face blur score : "
        f"{MIN_FACE_BLUR_SCORE}"
    )

    print()

    print(
        f"Report saved to:\n"
        f"{OUTPUT_CSV}"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()