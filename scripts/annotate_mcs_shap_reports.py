from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]

REPORT_DIRECTORIES = [
    PROJECT_ROOT
    / "experiments"
    / "reports"
    / "shap"
    / "balanced_bagging_standard_ml"
    / "gradient_boosting",
    PROJECT_ROOT
    / "experiments"
    / "reports"
    / "shap"
    / "balanced_bagging_longitudinal_ml"
    / "merwav_time_plus__lexico_random_forest",
]

FEATURE_COLUMN_FILES = [
    "multiclass_mean_abs_shap_pivot.csv",
    "multiclass_mean_signed_shap_pivot.csv",
    "multiclass_mean_shap_long.csv",
]

FEATURE_RE = re.compile(
    r"^(?:(?P<non_long>non_long)__w(?P<non_long_wave>\d+)__)?"
    r"(?P<root>[A-Z0-9]+)"
    r"(?:__v(?P<category>\d+))?"
    r"(?:_w(?P<wave>\d+))?$"
)


VARIABLE_LABELS = {
    "ACACCA00": "Accidents or injuries count",
    "ACADMA00": "Hospital admissions count",
    "ACADMO00": "Health problems count",
    "ADACAQ00": "Highest academic qualification (NVQ equivalent)",
    "ADC06E00": "Cohort member ethnic group (6-category)",
    "ADD05C00": "NS-SEC 5-class (last known job)",
    "ADOEDE00": "OECD equivalised household income",
    "ADROOW00": "Housing tenure",
    "AHCSEX00": "Cohort member sex",
    "APALDR00": "Alcohol consumption frequency",
    "APDEAN00": "Ever diagnosed with depression or serious anxiety",
    "APDIST9": "Psychological distress composite",
    "APDROF00": "Alcohol consumption during pregnancy",
    "APFCIN00": "Current legal marital status",
    "APGEHE00": "General health",
    "APILPR00": "Illness/problems during pregnancy",
    "APLOSA00": "Felt low or sad since child was born",
    "APWALI00": "Life satisfaction",
}

FEATURE_TYPES = {
    "ACACCA00": "ordinal_numeric",
    "ACADMA00": "ordinal_numeric",
    "ACADMO00": "ordinal_numeric",
    "ADACAQ00": "ordinal_ranked",
    "ADC06E00": "one_hot",
    "ADD05C00": "one_hot",
    "ADOEDE00": "continuous_numeric",
    "ADROOW00": "one_hot",
    "AHCSEX00": "binary_indicator",
    "APALDR00": "ordinal_numeric",
    "APDEAN00": "binary_indicator",
    "APDIST9": "continuous_numeric",
    "APDROF00": "ordinal_numeric",
    "APFCIN00": "one_hot",
    "APGEHE00": "ordinal_numeric",
    "APILPR00": "binary_indicator",
    "APLOSA00": "binary_indicator",
    "APWALI00": "ordinal_numeric",
}

CATEGORY_LABELS = {
    "ADC06E00": {
        "1": "White",
        "2": "Mixed",
        "3": "Indian",
        "4": "Pakistani and Bangladeshi",
        "5": "Black or Black British",
        "6": "Other ethnic group (including Chinese/Other)",
    },
    "ADD05C00": {
        "1": "Managerial and professional",
        "2": "Intermediate",
        "3": "Small employers and self-employed",
        "4": "Lower supervisory and technical",
        "5": "Semi-routine and routine",
    },
    "ADROOW00": {
        "1": "Own outright",
        "2": "Own with mortgage/loan",
        "3": "Part rent/part mortgage (shared equity)",
        "4": "Rent from local authority",
        "5": "Rent from Housing Association",
        "6": "Rent privately",
        "7": "Living with parents",
        "8": "Live rent free",
        "9": "Squatting",
        "95": "Other",
    },
    "APFCIN00": {
        "1": "Legally separated",
        "2": "Married, first and only marriage",
        "3": "Remarried, second or later marriage",
        "4": "Single, never married",
        "5": "Divorced",
        "6": "Widowed",
        "7": "Civil partner",
        "8": "Former civil partner",
        "9": "Surviving civil partner",
    },
}


def build_source_feature_name(parsed: dict[str, str]) -> str:
    if parsed["is_non_longitudinal"] == "yes":
        return f"non_long__w{parsed['wave']}__{parsed['root_feature']}"
    return f"{parsed['root_feature']}_w{parsed['wave']}"


def build_naming_template(parsed: dict[str, str]) -> str:
    if parsed["feature_type"] != "one_hot":
        return ""
    if parsed["is_non_longitudinal"] == "yes":
        return "non_long__w{wave}__{root}__v{code}"
    return "{root}__v{code}_w{wave}"


def parse_feature(feature: str) -> dict[str, str]:
    match = FEATURE_RE.match(feature)
    if not match:
        raise ValueError(f"Unrecognised feature name: {feature}")

    wave = match.group("wave") or match.group("non_long_wave")
    parsed = {
        "feature": feature,
        "root_feature": match.group("root"),
        "wave": wave or "",
        "wave_label": f"W{wave}" if wave else "",
        "category_code": match.group("category") or "",
        "is_non_longitudinal": "yes" if match.group("non_long") else "no",
    }
    parsed["feature_type"] = FEATURE_TYPES[parsed["root_feature"]]
    parsed["source_feature"] = build_source_feature_name(parsed)
    parsed["naming_template"] = build_naming_template(parsed)
    return parsed


def annotate_feature(feature: str) -> dict[str, str]:
    parsed = parse_feature(feature)
    root = parsed["root_feature"]
    wave = parsed["wave"]
    category_code = parsed["category_code"]
    wave_label = parsed["wave_label"]
    source_feature = parsed["source_feature"]

    friendly_name: str
    meaning: str
    coding_notes: str
    encoded_source_code = ""
    encoded_source_label = ""

    if root == "ACACCA00":
        friendly_name = f"{VARIABLE_LABELS[root]} ({wave_label})"
        meaning = (
            f"Numeric count copied from `{source_feature}` for {wave_label}."
        )
        coding_notes = (
            "Kept as a numeric count after sentinel codes were converted to missing. "
            "Higher values mean more reported accidents or injuries. W7 comes from "
            "the cohort member interview rather than the parent interview."
        )
    elif root == "ACADMA00":
        friendly_name = f"{VARIABLE_LABELS[root]} ({wave_label})"
        meaning = (
            f"Numeric count copied from `{source_feature}` for hospital admissions "
            f"because of accident, illness, or health problems at {wave_label}."
        )
        coding_notes = (
            "Kept as a numeric count after sentinel codes were converted to missing. "
            "Higher values mean more hospital admissions. W7 comes from the cohort "
            "member interview rather than the parent interview."
        )
    elif root == "ACADMO00":
        friendly_name = f"{VARIABLE_LABELS[root]} ({wave_label})"
        meaning = f"Numeric count copied from `{source_feature}` for {wave_label}."
        coding_notes = (
            "Kept as a numeric count after preprocessing. In this harmonisation the "
            "feature only appears for early waves."
        )
    elif root == "ADACAQ00":
        friendly_name = f"{VARIABLE_LABELS[root]} ({wave_label})"
        meaning = (
            f"Ordinal value copied from `{source_feature}` for the NVQ-equivalent "
            f"qualification level at {wave_label}."
        )
        coding_notes = (
            "Encoded as an ordinal rank with ordered values 1, 2, 3, 4, 5, 95. "
            "Higher values indicate higher qualification level, with 95 reserved "
            "for 'overseas qualification only' and kept as an ordered category."
        )
    elif root == "ADC06E00":
        category_label = CATEGORY_LABELS[root][category_code]
        encoded_source_code = category_code
        encoded_source_label = category_label
        friendly_name = (
            f"{VARIABLE_LABELS[root]}: code {category_code} = {category_label} "
            f"({wave_label})"
        )
        meaning = (
            f"Dummy column derived from `{source_feature}`. Here `v{category_code}` "
            f"means the original harmonised category code was `{category_code}` "
            f"(`{category_label}`). A value of 1 means `{source_feature} == "
            f"{category_code}`; 0 means another retained ethnic-group category."
        )
        coding_notes = (
            "Created by one-hot encoding the harmonised 6-category census-style "
            "ethnic-group variable using the non-longitudinal template "
            "`non_long__w{wave}__{root}__v{code}` from the stage 5 encoding policy."
        )
    elif root == "ADD05C00":
        category_label = CATEGORY_LABELS[root][category_code]
        encoded_source_code = category_code
        encoded_source_label = category_label
        friendly_name = (
            f"{VARIABLE_LABELS[root]}: code {category_code} = {category_label} "
            f"({wave_label})"
        )
        meaning = (
            f"Dummy column derived from `{source_feature}`. Here `v{category_code}` "
            f"means the original harmonised category code was `{category_code}` "
            f"(`{category_label}`). A value of 1 means `{source_feature} == "
            f"{category_code}`; 0 means another retained NS-SEC class."
        )
        coding_notes = (
            "Created by one-hot encoding the harmonised 5-class NS-SEC variable "
            "using the longitudinal template `{root}__v{code}_w{wave}`."
        )
    elif root == "ADOEDE00":
        friendly_name = f"{VARIABLE_LABELS[root]} ({wave_label})"
        meaning = (
            f"Continuous value copied from `{source_feature}` for OECD-equivalised "
            f"household income at {wave_label}."
        )
        coding_notes = (
            "Kept as a continuous numeric variable after sentinel codes were "
            "converted to missing."
        )
    elif root == "ADROOW00":
        category_label = CATEGORY_LABELS[root].get(
            category_code, f"Category code {category_code}"
        )
        encoded_source_code = category_code
        encoded_source_label = category_label
        friendly_name = (
            f"{VARIABLE_LABELS[root]}: code {category_code} = {category_label} "
            f"({wave_label})"
        )
        meaning = (
            f"Dummy column derived from `{source_feature}`. Here `v{category_code}` "
            f"means the original harmonised category code was `{category_code}` "
            f"(`{category_label}`). A value of 1 means `{source_feature} == "
            f"{category_code}`; 0 means another retained tenure category."
        )
        coding_notes = (
            "Created by one-hot encoding the tenure-of-current-home variable using "
            "the longitudinal template `{root}__v{code}_w{wave}`."
        )
    elif root == "AHCSEX00":
        friendly_name = f"{VARIABLE_LABELS[root]} (female indicator, {wave_label})"
        meaning = (
            f"Binary feature derived from `{source_feature}`. A value of 1 means "
            "female and 0 means male."
        )
        coding_notes = "Mapped from the W1 cohort member sex variable using 1->0 and 2->1."
    elif root == "APALDR00":
        friendly_name = f"{VARIABLE_LABELS[root]} ({wave_label})"
        meaning = (
            f"Ordinal code copied from `{source_feature}` for how often the "
            f"respondent drinks alcohol at {wave_label}."
        )
        if wave in {"1", "2", "3", "4"}:
            coding_notes = (
                "Lower values mean more frequent drinking. Early waves use the "
                "more detailed frequency scale that runs from 'every day' to "
                "'never'; non-response codes were converted to missing."
            )
        else:
            coding_notes = (
                "Lower values mean more frequent drinking. Later waves use the "
                "revised scale from '4 or more times a week' to 'never'; "
                "non-response codes were converted to missing."
            )
    elif root == "APDEAN00":
        friendly_name = f"{VARIABLE_LABELS[root]} ({wave_label})"
        meaning = (
            f"Binary feature derived from `{source_feature}`. A value of 1 means "
            "the respondent reports ever being diagnosed with depression or "
            "serious anxiety; 0 means no."
        )
        coding_notes = "Mapped from Yes/No to 1/0 before modelling."
    elif root == "APDIST9":
        friendly_name = f"{VARIABLE_LABELS[root]} (0-9, {wave_label})"
        meaning = (
            "W1 parent psychological distress composite created during preprocessing "
            "by summing nine yes/no distress items."
        )
        coding_notes = (
            "Higher values indicate more distress. This is a preprocessing "
            "composite, not a single raw questionnaire field."
        )
    elif root == "APDROF00":
        friendly_name = f"{VARIABLE_LABELS[root]} ({wave_label})"
        meaning = (
            f"Ordinal code copied from `{source_feature}` for how often the "
            "respondent drank alcohol during pregnancy."
        )
        coding_notes = (
            "Lower values mean more frequent drinking during pregnancy; the raw "
            "response scale runs from 'every day' to 'never'."
        )
    elif root == "APFCIN00":
        category_label = CATEGORY_LABELS[root][category_code]
        encoded_source_code = category_code
        encoded_source_label = category_label
        friendly_name = (
            f"{VARIABLE_LABELS[root]}: code {category_code} = {category_label} "
            f"({wave_label})"
        )
        meaning = (
            f"Dummy column derived from `{source_feature}`. Here `v{category_code}` "
            f"means the original harmonised category code was `{category_code}` "
            f"(`{category_label}`). A value of 1 means `{source_feature} == "
            f"{category_code}`; 0 means another retained legal-status category."
        )
        coding_notes = (
            "Created by one-hot encoding the harmonised marital-status variable. "
            "The naming template is `{root}__v{code}_w{wave}`, so in "
            f"`{feature}` the `v{category_code}` part is the retained source code. "
            "Civil-partnership categories appear only in later waves."
        )
    elif root == "APGEHE00":
        friendly_name = f"{VARIABLE_LABELS[root]} ({wave_label})"
        meaning = (
            f"Ordinal self-rated general-health code copied from `{source_feature}` "
            f"at {wave_label}."
        )
        if wave in {"1", "2"}:
            coding_notes = (
                "Higher values mean worse health. W1-W2 use a 4-point scale from "
                "'excellent' to 'poor'."
            )
        else:
            coding_notes = (
                "Higher values mean worse health. W3+ use a 5-point scale from "
                "'excellent' to 'poor'."
            )
    elif root == "APILPR00":
        friendly_name = f"{VARIABLE_LABELS[root]} ({wave_label})"
        meaning = (
            f"Binary feature derived from `{source_feature}`. A value of 1 means "
            "the respondent reported illnesses or problems during pregnancy; "
            "0 means no."
        )
        coding_notes = "Mapped from Yes/No to 1/0 before modelling."
    elif root == "APLOSA00":
        friendly_name = f"{VARIABLE_LABELS[root]} ({wave_label})"
        meaning = (
            f"Binary feature derived from `{source_feature}`. A value of 1 means "
            "the respondent felt low or sad since the most recent child was born; "
            "0 means no."
        )
        coding_notes = "Mapped from Yes/No to 1/0 before modelling."
    elif root == "APWALI00":
        friendly_name = f"{VARIABLE_LABELS[root]} ({wave_label})"
        meaning = (
            f"Ordinal life-satisfaction score copied from `{source_feature}` at "
            f"{wave_label}."
        )
        coding_notes = (
            "Higher valid values mean more satisfaction. The exact numeric response "
            "scale differs by wave, and non-response codes were converted to missing "
            "during preprocessing."
        )
    else:
        raise KeyError(f"No annotation metadata configured for root {root}")

    return {
        **parsed,
        "encoded_source_code": encoded_source_code,
        "encoded_source_label": encoded_source_label,
        "friendly_name": friendly_name,
        "meaning": meaning,
        "coding_notes": coding_notes,
    }


def natural_sort_key(value: str) -> list[Any]:
    parts = re.split(r"(\d+)", value)
    return [int(part) if part.isdigit() else part for part in parts]


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv_rows(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def annotate_report(report_dir: Path) -> None:
    feature_files = [
        report_dir / file_name
        for file_name in FEATURE_COLUMN_FILES
        if (report_dir / file_name).exists()
    ]
    if not feature_files:
        raise FileNotFoundError(f"No feature CSV files found in {report_dir}")

    all_features: set[str] = set()
    abs_rows = read_csv_rows(report_dir / "multiclass_mean_abs_shap_pivot.csv")
    top20_rank = {
        row["feature"]: str(index)
        for index, row in enumerate(abs_rows, start=1)
    }

    for feature_file in feature_files:
        for row in read_csv_rows(feature_file):
            if "feature" in row:
                all_features.add(row["feature"])

    annotations = {
        feature: annotate_feature(feature)
        for feature in sorted(all_features, key=natural_sort_key)
    }

    dictionary_rows = []
    for feature in sorted(all_features, key=natural_sort_key):
        annotation = dict(annotations[feature])
        annotation["top20_abs_rank"] = top20_rank.get(feature, "")
        dictionary_rows.append(annotation)

    dictionary_fieldnames = [
        "feature",
        "top20_abs_rank",
        "wave",
        "wave_label",
        "root_feature",
        "source_feature",
        "category_code",
        "encoded_source_code",
        "encoded_source_label",
        "feature_type",
        "naming_template",
        "is_non_longitudinal",
        "friendly_name",
        "meaning",
        "coding_notes",
    ]
    write_csv_rows(
        report_dir / "mcs_shap_feature_dictionary.csv",
        dictionary_rows,
        dictionary_fieldnames,
    )

    for feature_file in feature_files:
        rows = read_csv_rows(feature_file)
        if not rows or "feature" not in rows[0]:
            continue

        annotated_rows = []
        is_pivot = feature_file.name.endswith("_pivot.csv")

        for index, row in enumerate(rows, start=1):
            feature = row["feature"]
            annotation = annotations[feature]
            annotated_row = {}
            if is_pivot:
                annotated_row["rank"] = top20_rank.get(feature, str(index))
            annotated_row.update(
                {
                    "feature": feature,
                    "wave": annotation["wave"],
                    "wave_label": annotation["wave_label"],
                    "root_feature": annotation["root_feature"],
                    "source_feature": annotation["source_feature"],
                    "category_code": annotation["category_code"],
                    "encoded_source_code": annotation["encoded_source_code"],
                    "encoded_source_label": annotation["encoded_source_label"],
                    "feature_type": annotation["feature_type"],
                    "naming_template": annotation["naming_template"],
                    "is_non_longitudinal": annotation["is_non_longitudinal"],
                    "friendly_name": annotation["friendly_name"],
                    "meaning": annotation["meaning"],
                    "coding_notes": annotation["coding_notes"],
                }
            )
            for key, value in row.items():
                if key != "feature":
                    annotated_row[key] = value
            annotated_rows.append(annotated_row)

        annotated_fieldnames = list(annotated_rows[0].keys())
        write_csv_rows(
            report_dir / f"{feature_file.stem}_annotated.csv",
            annotated_rows,
            annotated_fieldnames,
        )

    glossary_lines = [
        "# MCS SHAP feature glossary",
        "",
        "This glossary annotates the top-20 features in "
        "`multiclass_mean_abs_shap_pivot.csv` using the per-wave MCS UKDA data "
        "dictionaries under `data/MCS` and the preprocessing logs under "
        "`data/processed/MCS/preprocess_logs`.",
        "",
        "Useful conventions:",
        "",
        "- `__vN` means the retained one-hot source category code `N`.",
        "- Example: `APFCIN00__v5_w3` means the source feature was `APFCIN00_w3`, "
        "and the dummy is 1 exactly when the original harmonised value was code "
        "`5` (`Divorced`).",
        "- `non_long__w1__...` marks a W1-only non-longitudinal predictor.",
        "- Binary features were remapped to 0/1 before modelling.",
        "- The full union of annotated features for this report is available in "
        "`mcs_shap_feature_dictionary.csv`.",
        "",
        "| Rank | Feature | Friendly name | Meaning | Coding notes |",
        "| --- | --- | --- | --- | --- |",
    ]

    for index, row in enumerate(abs_rows, start=1):
        annotation = annotations[row["feature"]]
        table_cells = [
            str(index),
            row["feature"],
            annotation["friendly_name"],
            annotation["meaning"],
            annotation["coding_notes"],
        ]
        escaped_cells = [cell.replace("|", "\\|") for cell in table_cells]
        glossary_lines.append("| " + " | ".join(escaped_cells) + " |")

    glossary_path = report_dir / "mcs_shap_feature_glossary.md"
    glossary_path.write_text("\n".join(glossary_lines) + "\n", encoding="utf-8")


def main() -> None:
    for report_dir in REPORT_DIRECTORIES:
        annotate_report(report_dir)
        print(f"Annotated SHAP report: {report_dir}")


if __name__ == "__main__":
    main()
