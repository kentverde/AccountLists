# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Account segmentation report generator for the "Set the Floor" initiative. Processes account data to apply territory alignment and floor removal rules, then generates summary and detail reports.

## Commands

```bash
# Run the report generator
python generate_reports.py

# Run with custom input/output
python generate_reports.py Account_Segmentation_V3_3_Final.csv ./reports
```

## Architecture

### Main Script: `generate_reports.py`

Single-file Python script with modular functions:

1. **Data Loading** (`load_and_preprocess_data`): Loads CSV, adds `Final_Rep_Name` column if missing, handles blank rep names → "House"

2. **Territory Alignment** (`apply_territory_alignment`): Inside Sales accounts move to `Re-Assigned_Rep_Name`, others stay with `Assigned_Rep_Name`

3. **Floor Logic** (`apply_floor_logic`): Marks accounts for removal if: `LOW_REVENUE` + `DESIGN` segment + not new in 2025

4. **Summary Report** (`generate_summary_report`): One row per rep with before/after metrics

5. **Rep Detail Reports** (`generate_all_rep_detail_reports`): Individual CSV per rep with account listings by group

6. **Validation** (`validate_totals`): Ensures total revenue matches across input, summary, and detail reports

### Output Structure

```
reports/
├── Rep_Level_Impacts_2025_Summary.csv
└── rep_details/
    ├── Aaron_Kirsch_detail.csv
    ├── Adam_Paurowski_detail.csv
    └── ... (one per rep)
```

## Key Business Rules

- **Revenue Groups**: HIGH_REVENUE=Group 1, MID_REVENUE=Group 2, LOW_REVENUE=Group 3
- **Floor Removal**: Group 3 + DESIGN + non-2025 customer = removed from rep
- **Protected**: Non-DESIGN accounts and new 2025 customers are never removed
- **Alignment**: Only `Rep_Type = "Inside Sales"` accounts are reassigned

## Data File

**Account_Segmentation_V3_3_Final.csv** (~17,000 records)

Key columns:
- `Assigned_Rep_Name` / `Re-Assigned_Rep_Name`: Current and realigned rep
- `Rep_Type`: Inside Sales, Account Managers, Independent rep, Others, UNKNOWN
- `Revenue_Tier`: HIGH_REVENUE, MID_REVENUE, LOW_REVENUE
- `Segment`: DESIGN, RETAIL, E-TAILER, INTERNAL, MASS
- `Customer_Since_Date`: M/D/YYYY format
- `Total_Rev_2024` / `Total_Rev_2025`: Revenue figures
- `Final_Rep_Name`: Reserved for future use (currently blank)

## Validation

Total Rev 2025 must equal $84,760,387.57 across:
- Input file sum
- Summary report sum
- Sum of all rep detail reports
