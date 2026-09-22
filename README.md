# Facial Image Data Collection & Quality Validation Pipeline

A privacy-aware facial image acquisition and automated quality-control
pipeline designed as a proof of concept for AI training dataset preparation.

## Overview

High-quality AI datasets require more than simply collecting images.
Image quality, acquisition consistency, metadata integrity, and responsible
handling of identifiable data are important parts of the dataset preparation
process.

This project demonstrates a small-scale workflow for controlled facial-image
collection and automated quality assessment.

The pipeline evaluates facial images based on:

- Face presence
- Number of detected faces
- Face size within the image
- Facial-region brightness
- Facial-region sharpness
- Image resolution
- Quality-control acceptance or rejection

The project also incorporates informed-consent procedures, pseudonymous
subject identifiers, and privacy-aware handling of facial image data.

---

## Project Objectives

The project was developed to demonstrate:

1. Standardized facial-image acquisition under different capture conditions.
2. Automated facial-image quality assessment.
3. Structured QC reporting for AI dataset preparation.
4. Comparison between automated QC decisions and manual quality labels.
5. Privacy-aware handling of identifiable facial data.

---

## Data Collection

The pilot dataset contains facial images collected under controlled variations
in viewing angle, distance, lighting, framing, and image sharpness.

Examples of evaluated conditions include:

- Frontal view
- Left-side view
- Upward and downward head orientation
- Standard capture distance
- Increased capture distance
- Normal lighting
- Side lighting
- Low-light conditions
- Progressive image blur
- Invalid/cropped framing

A pseudonymous subject identifier is used instead of the participant's name.

Example:

S001_S01_frontal_normal_001.jpeg

---

## Quality-Control Pipeline

The processing workflow is:

Facial Image Acquisition  
↓  
Image Loading & Resolution Check  
↓  
Face Detection  
↓  
Facial Region Extraction  
↓  
Brightness Assessment  
↓  
Sharpness Assessment  
↓  
Face Size Assessment  
↓  
Automated ACCEPT / REJECT Decision  
↓  
QC Report Generation

OpenCV Haar Cascade is used for initial face detection.

Image sharpness is estimated using the Variance of Laplacian.

---

## Pilot QC Criteria

The current proof-of-concept uses empirically selected thresholds derived
from controlled pilot images.

| Metric | Pilot Criterion |
|---|---|
| Face presence | Required |
| Face count | 1 |
| Minimum face-area ratio | 0.05 |
| Minimum facial brightness | 70 |
| Minimum facial sharpness score | 8 |

These values are pilot thresholds and are not intended to represent universal
facial-image quality standards.

The current implementation evaluates underexposure but does not apply an
upper-brightness rejection threshold.

---

## Pilot Results

A total of 14 controlled test images were evaluated.

| Result | Images |
|---|---:|
| Accepted | 7 |
| Rejected | 7 |
| Total | 14 |

Automated QC decisions showed **14/14 agreement with manually assigned
quality labels on the pilot dataset**.

This result represents agreement on the small calibration dataset and should
not be interpreted as general validation accuracy.

---

## Evaluated Failure Conditions

The pilot dataset includes intentionally introduced quality issues such as:

- Severe underexposure
- Slight, moderate, and severe blur
- Excessive subject distance
- Invalid facial framing
- Unsupported or difficult head orientation

Rejected images are accompanied by a documented reason such as:

- `underexposed`
- `blurry`
- `face_too_small`
- `no_face_detected`
- `invalid_face_count`

---

## Privacy and Responsible Data Handling

Facial images are identifiable biometric-related data and are therefore not
included in the public repository.

The project applies privacy-aware data handling principles including:

- Informed consent
- Purpose limitation
- Data minimization
- Pseudonymous subject IDs
- Restricted access to identifiable images
- Defined data-retention considerations
- Data withdrawal and deletion procedures

Raw facial images remain stored locally and are excluded from GitHub.

The public repository contains only source code, documentation, non-identifying
metadata examples, QC results, and aggregate visualizations.

This proof-of-concept does not claim formal GDPR or CCPA compliance.

---

## Project Structure

```text
facial-data-quality-pipeline/
│
├── data/
│   ├── raw/                 # Private - excluded from GitHub
│   ├── accepted/            # Private - excluded from GitHub
│   └── rejected/            # Private - excluded from GitHub
│
├── metadata/
│   └── manual_labels.csv
│
├── reports/
│   ├── qc_face_analysis.csv
│   ├── qc_evaluation.csv
│   ├── qc_status_distribution.png
│   └── rejection_reasons.png
│
├── src/
│   ├── quality_check.py
│   ├── evaluate_qc.py
│   └── generate_report.py
│
├── capture_protocol.md
├── consent_template.md
├── data_privacy.md
├── requirements.txt
├── .gitignore
└── README.md