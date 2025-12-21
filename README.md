# Account Segmentation Report Generator

Generates summary and rep detail reports for the "Set the Floor" initiative.

## Overview

This project processes account segmentation data to:
1. Compare before/after account assignments (territory alignment)
2. Identify account movements and alignment impacts
3. Track floor exceptions and dormant accounts
4. Generate a summary report by rep
5. Generate individual rep detail reports
6. Validate totals match input data

## Versions

- **V2** (Current): Uses `Final_Rep_Name` as authoritative source; includes floor exceptions and SF ex BI tracking
- **V1** (Legacy): Uses Inside Sales logic with `Re-Assigned_Rep_Name`

## Usage

```bash
python generate_reports_v2.py [input_file] [output_directory]
```

**Defaults:**
- `input_file`: `Account_Segmentation_V3_3_Final_SOT_12-21-25.csv`
- `output_directory`: `./reports`

**Example:**
```bash
python generate_reports_v2.py Account_Segmentation_V3_3_Final_SOT_12-21-25.csv ./reports
```

## Notes

- Encoding: All output CSV files are written with UTF-8 encoding including a
  Byte Order Mark (BOM) using `utf-8-sig`. This preserves Unicode characters
  and helps Excel on Windows detect the file as UTF-8 when opening.
- Excel: If you open the CSV in Microsoft Excel and characters look garbled,
  re-open the file using Excel's "From Text/CSV" import option and set the
  file encoding to UTF-8.
- Output location: The summary report is written to `reports/Rep_Level_Impacts_2025_Summary_V2.csv`
  and individual rep detail files are in `reports/rep_details/` (with `_v2` suffix).

## Output Files

### Summary Report
`reports/Rep_Level_Impacts_2025_Summary_V2.csv`

One row per rep showing:
- **Before State**: Starting accounts/revenue (using `Assigned_Rep_Name`)
- **Account Movement**: How many accounts moved in/out, revenue impact, and % change metrics
- **After State**: Final accounts/revenue (using `Final_Rep_Name`)
- **Change Percentages**: % change relative to starting accounts/revenue
- **Floor Exceptions**: BTF='Y' accounts kept as exceptions (not removed)
- **SF ex BI Accounts**: Dormant accounts in Salesforce but not in BI
- **Zero Revenue**: New 2025 customers with $0 revenue

### Rep Detail Reports
`reports/rep_details/{Rep_Name}_detail_v2.csv`

One file per rep containing:
- **Summary Metrics**: Complete before/after comparison with the same columns as the summary report (for easy reference)
  - Before state, movements (in/out), change percentages, and after state
  - Floor exceptions, SF ex BI dormant accounts, and zero-revenue accounts
- **Segment Breakdown**: Accounts and revenue by segment
- **Group 1 (HIGH_REVENUE)**: All HIGH_REVENUE accounts (always kept)
- **Group 2 (MID_REVENUE)**: All MID_REVENUE accounts (always kept)
- **Group 3 (LOW_REVENUE)**: Protected LOW_REVENUE accounts (not flagged for removal)
- **Floor Exceptions**: BTF='Y' but account was kept with original rep
- **SF ex BI (Dormant)**: Accounts in Salesforce but with no transaction data
- **Zero Revenue**: Separate section highlighting $0 revenue accounts

## Business Logic

### Before vs After State

**BEFORE (Assigned_Rep_Name):**
- Where accounts started (original assignment)

**AFTER (Final_Rep_Name):**
- Where accounts ended (authoritative final assignment)
- Includes all moves, exceptions, and assignments

### Account Movement Categories

1. **Moved Accounts**: `Assigned_Rep_Name` ≠ `Final_Rep_Name`
   - Accounts transferred from one rep to another

2. **Floor Removed (BTF='Y' AND Moved)**: 
   - Flagged for removal AND moved away from original rep
   - Accounts removed from below-the-floor reps

3. **Floor Exceptions (BTF='Y' BUT NOT Moved)**:
   - Flagged for removal BUT kept with original rep
   - Exceptions made to the floor policy

4. **SF ex BI (In SF ex BI='Y')**:
   - In Salesforce but not in BI (no transaction data)
   - Dormant/inactive accounts being tracked separately

5. **Zero Revenue (Total_Rev_2025 = $0)**:
   - No revenue in 2025 (may be new customers or inactive)
   - Flagged for special attention

### Change Percentage Calculations

Two new columns provide context for account and revenue movements:

**Account_Change_Pct:**
- Formula: `(Net_Account_Change / Before_Accounts) × 100`
- Shows what % of starting accounts the net change represents
- Example: Lisa Lowder lost 159 accounts from 364 starting accounts = **-43.68%**
- Interpretation: Negative % = accounts lost, Positive % = accounts gained

**Revenue_Change_Pct:**
- Formula: `(Net_Rev_Change / Before_Rev_2025) × 100`
- Shows what % of starting 2025 revenue the net change represents
- Example: Lisa Lowder's -$47,985.22 change from $3,089,300.76 revenue = **-1.55%**
- Interpretation: Negative % = revenue lost, Positive % = revenue gained
- Note: Can be positive even if accounts decreased (better accounts kept, lower-value ones moved out)

### Input Column Requirements

**New V2 Columns:**
- `Final_Rep_Name`: The authoritative final rep after all assignments
- `BTF`: Y/N flag indicating "Below The Floor" account
- `In SF ex BI`: Y/N flag for Salesforce-only accounts (no BI transaction data)

**Existing Columns (still required):**
- `Customer No.`, `Account_Name`
- `Segment`, `Sub_Segment`
- `Assigned_Rep_Name`, `Rep_Type`
- `Customer_Since_Date` (M/D/YYYY format)
- `Revenue_Tier`, `Potential_Tier`
- `Total_Rev_2024`, `Total_Rev_2025`
- `Orders_2024`, `Orders_2025`

### Special Handling
- Blank `Assigned_Rep_Name` or `Final_Rep_Name` → replaced with "House"
- BTF and SF ex BI flags → standardized to uppercase Y/N
- Missing revenue/order values → filled with 0
- New 2025 customers (`Customer_Since_Date` year = 2025) → tracked separately

### Segment Naming Convention
| Revenue Tier | Group Name |
|--------------|------------|
| HIGH_REVENUE | Group 1 |
| MID_REVENUE | Group 2 |
| LOW_REVENUE | Group 3 |

Combined with Potential Tier for full segment label:
- "Group 1 High Potential", "Group 1 Medium Potential", "Group 1 Low Potential"
- "Group 2 High Potential", etc.

## Data Processing Steps

1. **Load & Preprocess**: Read CSV, clean rep names, parse dates, add helper columns
2. **Identify Movements**: Compare Assigned_Rep vs Final_Rep to detect changes
3. **Categorize Accounts**: Classify by floor status, exceptions, and SF ex BI
4. **Generate Summary**: Create rep-level before/after comparison
5. **Generate Details**: Create individual rep reports with account breakdowns
6. **Validate**: Confirm total revenue matches input data

## Data Notes

### Account Counting Methodology
This script counts **all accounts** including those with $0 revenue in 2025.

| Category | Count Method |
|----------|--------------|
| All accounts | Includes $0 revenue |
| Moved accounts | Based on Assigned vs Final rep |
| Floor exceptions | BTF='Y' AND Assigned_Rep = Final_Rep |
| Floor removed | BTF='Y' AND Assigned_Rep ≠ Final_Rep |
| SF ex BI | SF ex BI flag = 'Y' |

## Validation

The script validates that:
- Input total Rev 2025 = Summary report total (all reps combined)
- Totals are consistent across all processing steps

Validation errors will cause the script to exit with code 1 and display error details.

## Requirements

- Python 3.x
- pandas
- numpy
