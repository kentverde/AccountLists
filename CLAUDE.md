# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Status

**COMPLETED** - Core functionality working. Script generates validated reports.

Last session completed:
- Updated rep detail report format for spreadsheet-friendly output
- Summary Metrics as horizontal table (labels row, values row) for easy calculations
- Breakdown by Segment as matrix format (row labels in col A, column headers in row 1)
- Revenue fields rounded to 2 decimal places
- Order counts displayed as whole integers
- Booleans as uppercase TRUE/FALSE
- Removed Growth_Classification from detail output

Previous sessions:
- Built `generate_reports.py` with territory alignment and floor removal logic
- Summary report matching Power BI format (63 reps)
- Individual rep detail reports with account listings
- Data type validation for numeric columns
- Fixed CSV formatting (removed comma thousand-separators to avoid column splitting)

Open items for future sessions:
- `Final_Rep_Name` column is blank (reserved for future use)

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

4. **Data Type Validation** (`validate_and_enforce_dtypes`): Ensures numeric columns are proper float64/int64 before CSV export

5. **Summary Report** (`generate_summary_report`): One row per rep with before/after metrics

6. **Rep Detail Reports** (`generate_all_rep_detail_reports`): Individual CSV per rep with account listings by group

7. **Validation** (`validate_totals`): Ensures total revenue matches across input, summary, and detail reports

### Output Structure

```
reports/
├── Rep_Level_Impacts_2025_Summary.csv
└── rep_details/
    ├── Aaron_Kirsch_detail.csv
    ├── Adam_Paurowski_detail.csv
    └── ... (63 rep files total)
```

### Rep Detail Report Format

Each rep detail CSV is formatted for easy spreadsheet use with text/numbers separated:

1. **Header**: Rep name and generated timestamp

2. **Summary Metrics** (horizontal table):
   - Row 1: Labels (Total Accounts, Total 2024 Revenue, Total 2025 Revenue, etc.)
   - Row 2: Values directly below each label
   - Allows selecting value row for calculations without text interference

3. **Breakdown by Segment** (matrix):
   - Column A: Segment labels (Group 1 High Potential, etc.)
   - Row 1: Column headers (Accounts, Revenue, Removed)
   - Data values at intersections
   - Totals row at bottom

4. **Account Sections** (grouped by revenue tier):
   - GROUP 1 (HIGH REVENUE) - KEEPING
   - GROUP 2 (MID REVENUE) - KEEPING
   - GROUP 3 (LOW REVENUE) - KEEPING (protected accounts)
   - GROUP 3 (LOW REVENUE) - LOSING (floor removed)
   - ZERO REVENUE ACCOUNTS

5. **Detail Columns**: Account_ID, Account_Name, Segment, Sub_Segment, Total_Rev_2024, Total_Rev_2025, Orders_2024, Orders_2025, Revenue_Tier, Potential_Tier, Segment_Label, Customer_Since_Date, Is_New_2025, Composite_Score, Floor_Removed

Reference template: `Rep_detailFormatTemplate.csv`

## Key Business Rules

- **Revenue Groups**: HIGH_REVENUE=Group 1, MID_REVENUE=Group 2, LOW_REVENUE=Group 3
- **Potential Tiers**: HIGH_POTENTIAL, MID_POTENTIAL, LOW_POTENTIAL (9 total segments)
- **Floor Removal**: Group 3 + DESIGN + non-2025 customer = removed from rep
- **Protected**: Non-DESIGN accounts and new 2025 customers are never removed
- **Alignment**: Only `Rep_Type = "Inside Sales"` accounts are reassigned
- **Account Counts**: Include ALL accounts (differs from Power BI which excluded $0 revenue)

## Data File

**Account_Segmentation_V3_3_Final.csv** (~17,332 records)

Key columns:
- `Assigned_Rep_Name` / `Re-Assigned_Rep_Name`: Current and realigned rep
- `Rep_Type`: Inside Sales, Account Managers, Independent rep, Others, UNKNOWN
- `Revenue_Tier`: HIGH_REVENUE, MID_REVENUE, LOW_REVENUE
- `Potential_Tier`: HIGH_POTENTIAL, MID_POTENTIAL, LOW_POTENTIAL
- `Segment`: DESIGN, RETAIL, E-TAILER, INTERNAL, MASS
- `Customer_Since_Date`: M/D/YYYY format (year 2025 = new customer, protected)
- `Total_Rev_2024` / `Total_Rev_2025`: Revenue figures
- `Final_Rep_Name`: Reserved for future use (currently blank)

## Validation

Total Rev 2025 must equal $84,760,387.57 across:
- Input file sum
- Summary report sum
- Sum of all rep detail reports

## Reference Files

- `Rep Level Impacts - 2025.csv`: Power BI export used as reference for summary report format
- `Rep_detailFormatTemplate.csv`: Template showing desired rep detail report format (based on Angie Robison)
