# Account Segmentation Report Generator

Generates summary and rep detail reports for the "Set the Floor" initiative.

## Overview

This project processes account segmentation data to:
1. Apply territory alignment (Inside Sales rep reassignments)
2. Apply floor removal (Group 3 DESIGN accounts for non-2025 customers)
3. Generate a summary report by rep
4. Generate individual rep detail reports
5. Validate totals match input data

## Usage

```bash
python generate_reports.py [input_file] [output_directory]
```

**Defaults:**
- `input_file`: `Account_Segmentation_V3_3_Final.csv`
- `output_directory`: `./reports`

**Example:**
```bash
python generate_reports.py Account_Segmentation_V3_3_Final.csv ./reports
```

## Output Files

### Summary Report
`reports/Rep_Level_Impacts_2025_Summary.csv`

One row per rep showing:
- Starting revenue/accounts (2024, 2025)
- Account changes from territory alignment
- Post-alignment revenue/accounts
- Floor impact (Design accounts removed)
- Final revenue/accounts after floor removal
- Zero revenue new 2025 account counts

### Rep Detail Reports
`reports/rep_details/{Rep_Name}_detail.csv`

One file per rep containing:
- Summary metrics at top
- Breakdown by segment (Group + Potential)
- Group 1 (HIGH_REVENUE) accounts - keeping
- Group 2 (MID_REVENUE) accounts - keeping
- Group 3 (LOW_REVENUE) accounts - keeping (protected)
- Group 3 (LOW_REVENUE) accounts - losing (floor removed)
- Zero revenue accounts (separate visibility)

## Business Logic

### Territory Alignment
- **Inside Sales reps**: Accounts move from `Assigned_Rep_Name` to `Re-Assigned_Rep_Name`
- **All other rep types**: Accounts stay with `Assigned_Rep_Name`

### "Set the Floor" Initiative
Accounts are removed from reps if ALL conditions are true:
- `Revenue_Tier = LOW_REVENUE` (Group 3)
- `Segment = DESIGN`
- `Customer_Since_Date` year ≠ 2025 (not a new customer)

**Protected accounts** (not removed):
- Non-DESIGN segment accounts
- New 2025 customers (regardless of segment)

### Segment Naming Convention
| Revenue Tier | Group Name |
|--------------|------------|
| HIGH_REVENUE | Group 1 |
| MID_REVENUE | Group 2 |
| LOW_REVENUE | Group 3 |

Combined with Potential Tier for full segment label:
- "Group 1 High Potential", "Group 1 Medium Potential", "Group 1 Low Potential"
- "Group 2 High Potential", etc.

## Data Notes

### Account Counting Methodology
This script counts **all accounts** including those with $0 revenue in 2025. This differs from the Power BI reference report which excluded $0 revenue accounts from counts.

| Metric | This Script | Power BI Report |
|--------|-------------|-----------------|
| Account counts | All accounts | Only accounts with Rev 2025 ≠ 0 |
| Revenue totals | All accounts | All accounts |
| Floor removal counts | All accounts | Only accounts with Rev 2025 ≠ 0 |

**Revenue totals match exactly** between both approaches.

### Input File Requirements
- CSV format with headers
- Required columns:
  - `Account_ID`, `Account_Name`
  - `Segment`, `Sub_Segment`
  - `Assigned_Rep_Name`, `Re-Assigned_Rep_Name`, `Rep_Type`
  - `Customer_Since_Date` (M/D/YYYY format)
  - `Revenue_Tier`, `Potential_Tier`
  - `Total_Rev_2024`, `Total_Rev_2025`
  - `Orders_2024`, `Orders_2025`

### Special Handling
- Blank `Assigned_Rep_Name` or `Re-Assigned_Rep_Name` → replaced with "House"
- `Final_Rep_Name` column added if not present (for future use)

## Validation

The script validates that:
- Input total Rev 2025 = Summary report total
- Input total Rev 2025 = Sum of all rep detail reports

Validation errors will cause the script to exit with code 1.

## Requirements

- Python 3.x
- pandas
- numpy
