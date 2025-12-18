"""
Account Segmentation Report Generator
=====================================
Generates summary and rep detail reports for the "Set the Floor" initiative.

This script processes account segmentation data to:
1. Apply territory alignment (Inside Sales rep reassignments)
2. Apply floor removal (Group 3 DESIGN accounts, non-2025 customers)
3. Generate a summary report by rep
4. Generate individual rep detail reports
5. Validate totals match input data

Usage:
        python generate_reports.py [input_file] [output_directory]

        Defaults:
                input_file: Account_Segmentation_V3_3_Final.csv
                output_directory: ./reports

Notes about encoding and Excel:
- All generated CSV files are written using UTF-8 with a BOM (`utf-8-sig`).
    This encoding preserves Unicode characters and helps Microsoft Excel
    recognize UTF-8 files correctly when opening on Windows.
- If you open the CSVs in other editors, they will also read UTF-8 correctly.

The script is intentionally conservative with data handling:
- Blank rep names are converted to "House" to keep counts consistent.
- New 2025 customers are protected from the floor removal rule.

"""

import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime
from pathlib import Path


# =============================================================================
# CONFIGURATION
# =============================================================================

# Column names from input file
COL_ACCOUNT_ID = 'Account_ID'
COL_ACCOUNT_NAME = 'Account_Name'
COL_SEGMENT = 'Segment'
COL_SUB_SEGMENT = 'Sub_Segment'
COL_ASSIGNED_REP = 'Assigned_Rep_Name'
COL_REASSIGNED_REP = 'Re-Assigned_Rep_Name'
COL_REP_TYPE = 'Rep_Type'
COL_CUSTOMER_SINCE = 'Customer_Since_Date'
COL_REVENUE_TIER = 'Revenue_Tier'
COL_POTENTIAL_TIER = 'Potential_Tier'
COL_TOTAL_REV_2024 = 'Total_Rev_2024'
COL_TOTAL_REV_2025 = 'Total_Rev_2025'
COL_ORDERS_2024 = 'Orders_2024'
COL_ORDERS_2025 = 'Orders_2025'
COL_GROWTH_CLASS = 'Growth_Classification'
COL_COMPOSITE_SCORE = 'Composite_Score'
COL_FINAL_REP = 'Final_Rep_Name'

# Group mappings
REVENUE_TIER_TO_GROUP = {
    'HIGH_REVENUE': 'Group 1',
    'MID_REVENUE': 'Group 2',
    'LOW_REVENUE': 'Group 3'
}

POTENTIAL_TIER_TO_LABEL = {
    'HIGH_POTENTIAL': 'High Potential',
    'MID_POTENTIAL': 'Medium Potential',
    'LOW_POTENTIAL': 'Low Potential'
}

# Floor criteria
FLOOR_SEGMENT = 'DESIGN'
FLOOR_REVENUE_TIER = 'LOW_REVENUE'
PROTECTED_YEAR = 2025


# =============================================================================
# DATA LOADING AND PREPROCESSING
# =============================================================================

def load_and_preprocess_data(input_file: str) -> pd.DataFrame:
    """
    Load the input CSV and perform preprocessing:
    - Add Final_Rep_Name column if not present
    - Replace blank Assigned_Rep_Name with 'House'
    - Parse Customer_Since_Date and extract year
    - Add computed columns for grouping
    """
    print(f"Loading data from: {input_file}")
    
    # READ THE CSV FILE
    # This loads the account segmentation data into a pandas DataFrame
    # A DataFrame is like a spreadsheet in memory - rows are accounts, columns are data fields
    df = pd.read_csv(input_file)

    # COUNT HOW MANY ACCOUNTS WE LOADED
    # len(df) gives us the number of rows (accounts) in the DataFrame
    original_count = len(df)
    print(f"Loaded {original_count:,} accounts")

    # ADD FINAL_REP_NAME COLUMN IF IT DOESN'T EXIST
    # This column will eventually store the final rep assignment after all transformations
    # We initialize it as empty strings for now
    if COL_FINAL_REP not in df.columns:
        df[COL_FINAL_REP] = ''
        print("Added Final_Rep_Name column")

    # CLEAN UP BLANK/MISSING ASSIGNED REP NAMES
    # Some accounts might not have an assigned rep (blank or null values)
    # We replace these with 'House' which represents unassigned/house accounts
    # .fillna() replaces null/NaN values
    # .str.strip() removes leading/trailing spaces
    df[COL_ASSIGNED_REP] = df[COL_ASSIGNED_REP].fillna('House')
    df.loc[df[COL_ASSIGNED_REP].str.strip() == '', COL_ASSIGNED_REP] = 'House'

    # CLEAN UP BLANK/MISSING RE-ASSIGNED REP NAMES
    # Same logic as above - replace blanks/nulls with 'House'
    # Re-Assigned_Rep_Name is where Inside Sales accounts might be moving to
    df[COL_REASSIGNED_REP] = df[COL_REASSIGNED_REP].fillna('House')
    df.loc[df[COL_REASSIGNED_REP].str.strip() == '', COL_REASSIGNED_REP] = 'House'

    # PARSE CUSTOMER START DATE AND EXTRACT THE YEAR
    # Customer_Since_Date is stored as text like "01/15/2025"
    # pd.to_datetime() converts text to a date object
    # format='%m/%d/%Y' tells it the format is Month/Day/Year
    # errors='coerce' means if date is invalid, set it to NaN instead of crashing
    # .dt.year extracts just the year portion (e.g., 2025)
    df['Customer_Since_Year'] = pd.to_datetime(
        df[COL_CUSTOMER_SINCE], format='%m/%d/%Y', errors='coerce'
    ).dt.year

    # IDENTIFY NEW 2025 CUSTOMERS
    # Create a True/False flag: True if customer started in 2025, False otherwise
    # These accounts are "protected" from the floor removal later
    # PROTECTED_YEAR is defined as 2025 in the configuration section
    df['Is_New_2025'] = df['Customer_Since_Year'] == PROTECTED_YEAR

    # CREATE READABLE REVENUE GROUP LABELS
    # Revenue_Tier contains codes like 'HIGH_REVENUE', 'MID_REVENUE', 'LOW_REVENUE'
    # .map() converts these codes to readable labels: 'Group 1', 'Group 2', 'Group 3'
    # The mapping is defined in REVENUE_TIER_TO_GROUP dictionary
    # .fillna('Unknown') handles any unexpected values
    df['Revenue_Group'] = df[COL_REVENUE_TIER].map(REVENUE_TIER_TO_GROUP).fillna('Unknown')

    # CREATE READABLE POTENTIAL LABELS
    # Similar to above, converts codes like 'HIGH_POTENTIAL' to 'High Potential'
    # The mapping is defined in POTENTIAL_TIER_TO_LABEL dictionary
    df['Potential_Label'] = df[COL_POTENTIAL_TIER].map(POTENTIAL_TIER_TO_LABEL).fillna('Unknown')

    # CREATE COMBINED SEGMENT LABELS
    # Concatenates the group and potential labels together
    # Example: "Group 1" + " " + "High Potential" = "Group 1 High Potential"
    # This makes it easier to group and display accounts by their full segment
    df['Segment_Label'] = df['Revenue_Group'] + ' ' + df['Potential_Label']

    # FILL MISSING REVENUE AND ORDER VALUES WITH ZERO
    # Some accounts might have NaN (Not a Number) for revenue or orders
    # NaN means the value is missing or undefined
    # We replace NaN with 0 so calculations won't fail
    # This is safer than leaving NaN which can cause errors in math operations
    df[COL_TOTAL_REV_2024] = df[COL_TOTAL_REV_2024].fillna(0)
    df[COL_TOTAL_REV_2025] = df[COL_TOTAL_REV_2025].fillna(0)
    df[COL_ORDERS_2024] = df[COL_ORDERS_2024].fillna(0)
    df[COL_ORDERS_2025] = df[COL_ORDERS_2025].fillna(0)

    # CALCULATE AND DISPLAY TOTAL 2025 REVENUE
    # .sum() adds up all values in the column across all rows
    # This gives us a baseline total to validate against later
    # We'll check this matches the sum in our output reports
    total_rev_2025 = df[COL_TOTAL_REV_2025].sum()
    print(f"Total Rev 2025: ${total_rev_2025:,.2f}")

    return df


# =============================================================================
# TERRITORY ALIGNMENT
# =============================================================================

def apply_territory_alignment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply territory alignment:
    - For Inside Sales: account moves to Re-Assigned_Rep_Name
    - For all others: account stays with Assigned_Rep_Name

    Adds 'Aligned_Rep_Name' column with the post-alignment rep
    """
    # CREATE A COPY TO AVOID MODIFYING THE ORIGINAL DATAFRAME
    # .copy() creates a new independent DataFrame
    # This is good practice to prevent unexpected side effects
    df = df.copy()

    # DETERMINE THE ALIGNED REP FOR EACH ACCOUNT
    # This is the first major transformation: Territory Alignment
    # 
    # BUSINESS RULE: Inside Sales reps are getting their territories reorganized
    # - If Rep_Type is 'Inside Sales', the account moves to Re-Assigned_Rep_Name
    # - For all other rep types, the account stays with Assigned_Rep_Name
    # 
    # np.where() is like an IF-THEN-ELSE for every row:
    # - Condition: Is Rep_Type == 'Inside Sales'?
    # - If True: use Re-Assigned_Rep_Name
    # - If False: use Assigned_Rep_Name
    df['Aligned_Rep_Name'] = np.where(
        df[COL_REP_TYPE] == 'Inside Sales',
        df[COL_REASSIGNED_REP],
        df[COL_ASSIGNED_REP]
    )

    # TRACK WHICH ACCOUNTS CHANGED REPS
    # Create a True/False flag for each account
    # True = the account is moving to a different rep
    # False = the account is staying with the same rep
    # This helps us count and report on alignment impacts
    df['Alignment_Changed'] = df[COL_ASSIGNED_REP] != df['Aligned_Rep_Name']

    # COUNT HOW MANY ACCOUNTS ARE CHANGING REPS
    # .sum() on a True/False column counts the True values
    # True is treated as 1, False as 0, so sum gives us the count
    alignment_changes = df['Alignment_Changed'].sum()
    print(f"Accounts affected by alignment: {alignment_changes:,}")

    return df


# =============================================================================
# FLOOR LOGIC
# =============================================================================

def apply_floor_logic(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply floor removal logic:
    - Group 3 (LOW_REVENUE) + DESIGN segment + not new in 2025 = removed

    Adds 'Floor_Removed' column (True if account is removed)
    """
    # CREATE A COPY TO AVOID MODIFYING THE ORIGINAL DATAFRAME
    df = df.copy()

    # IDENTIFY ACCOUNTS TO REMOVE ("SETTING THE FLOOR")
    # This is the second major transformation: Floor Removal
    # 
    # BUSINESS RULE: Remove low-value DESIGN accounts that aren't brand new
    # An account is removed if ALL THREE conditions are true:
    # 
    # 1. Revenue_Tier == 'LOW_REVENUE' (Group 3 accounts)
    #    These are the smallest revenue accounts
    # 
    # 2. Segment == 'DESIGN' (FLOOR_SEGMENT)
    #    Only DESIGN segment accounts are eligible for removal
    #    Other segments like BUILD are protected
    # 
    # 3. NOT Is_New_2025 (~df['Is_New_2025'])
    #    The ~ symbol means NOT (flips True to False and vice versa)
    #    New 2025 customers are protected even if they're low revenue
    #    This gives new customers time to grow
    # 
    # The & symbol means AND (all conditions must be true)
    df['Floor_Removed'] = (
        (df[COL_REVENUE_TIER] == FLOOR_REVENUE_TIER) &
        (df[COL_SEGMENT] == FLOOR_SEGMENT) &
        (~df['Is_New_2025'])
    )

    # COUNT HOW MANY ACCOUNTS ARE BEING REMOVED
    # .sum() on True/False counts the True values
    floor_removed_count = df['Floor_Removed'].sum()
    
    # CALCULATE TOTAL REVENUE BEING REMOVED
    # df.loc[df['Floor_Removed'], COL_TOTAL_REV_2025] selects only rows where Floor_Removed is True
    # Then .sum() adds up the 2025 revenue for those accounts
    floor_removed_revenue = df.loc[df['Floor_Removed'], COL_TOTAL_REV_2025].sum()

    print(f"Accounts removed by floor: {floor_removed_count:,}")
    print(f"Revenue removed by floor: ${floor_removed_revenue:,.2f}")

    return df


# =============================================================================
# DATA TYPE VALIDATION
# =============================================================================

# Define expected data types for summary report columns
SUMMARY_COLUMN_TYPES = {
    # String columns
    'Assigned_Rep_Name': 'string',
    'Rep_Type': 'string',
    # Float columns (currency/revenue)
    '2024 Revenue': 'float',
    '2025 Revenue': 'float',
    '2025 Revenue (after Re-Assignment)': 'float',
    'Revenue Change (after Re-Assignment)': 'float',
    '2025 Revenue for The Floor (Design Only, Non New)': 'float',
    '2025 Final Revenue All Accounts (Floor Removed)': 'float',
    'Revenue Diff Between Start and Final 2025': 'float',
    # Integer columns (counts)
    '2025 Count of Accounts': 'int',
    'Account Changes due to Alignment': 'int',
    'Count of Accounts after Alignment': 'int',
    'The Floor (All Group 3 Accounts)': 'int',
    'The Floor (Design Only, Non New Accounts)': 'int',
    '2025 Final Count of Accounts (Floor Removed)': 'int',
    'Overall Count of Account Difference': 'int',
    'Count of 2025 Zero Rev New Accounts (Re-Assigned)': 'int',
}


def validate_and_enforce_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate and enforce data types for the summary report.
    Ensures numeric columns are properly typed before CSV export.
    """
    # CREATE A COPY TO AVOID MODIFYING THE ORIGINAL
    df = df.copy()
    
    # LIST TO COLLECT ANY ERRORS WE ENCOUNTER
    errors = []

    # LOOP THROUGH EACH COLUMN AND ITS EXPECTED DATA TYPE
    # SUMMARY_COLUMN_TYPES is a dictionary defined earlier that maps
    # column names to their expected types: 'float', 'int', or 'string'
    for col, expected_type in SUMMARY_COLUMN_TYPES.items():
        # SKIP THIS COLUMN IF IT DOESN'T EXIST IN THE DATAFRAME
        # Some columns might be optional
        if col not in df.columns:
            continue

        try:
            # HANDLE FLOAT COLUMNS (REVENUE, CURRENCY VALUES)
            if expected_type == 'float':
                # pd.to_numeric() converts values to numbers
                # errors='coerce' means invalid values become NaN instead of crashing
                # .astype(float) ensures the final data type is float
                df[col] = pd.to_numeric(df[col], errors='coerce').astype(float)
                
                # CHECK IF ANY VALUES FAILED TO CONVERT
                # .isna() returns True for NaN values
                # .sum() counts how many are True
                null_count = df[col].isna().sum()
                if null_count > 0:
                    errors.append(f"Column '{col}': {null_count} values could not be converted to float")

            # HANDLE INTEGER COLUMNS (COUNTS, WHOLE NUMBERS)
            elif expected_type == 'int':
                # First convert to numeric (handles various formats)
                # Then fill NaN with 0 (can't have NaN in integer column)
                # Finally convert to integer type
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

            # HANDLE STRING COLUMNS (TEXT VALUES LIKE REP NAMES)
            elif expected_type == 'string':
                # Convert everything to string type
                df[col] = df[col].astype(str)

        except Exception as e:
            # IF CONVERSION FAILS, RECORD THE ERROR
            errors.append(f"Column '{col}': Error converting to {expected_type} - {str(e)}")

    if errors:
        print("\nDATA TYPE VALIDATION WARNINGS:")
        for error in errors:
            print(f"  - {error}")
    else:
        print("Data type validation passed.")

    # Final verification - print dtypes for numeric columns
    print("\nColumn data types:")
    for col in df.columns:
        if col in SUMMARY_COLUMN_TYPES:
            print(f"  {col}: {df[col].dtype}")

    return df


# =============================================================================
# SUMMARY REPORT GENERATION
# =============================================================================

def generate_summary_report(df: pd.DataFrame, output_file: str) -> pd.DataFrame:
    """
    Generate summary report by rep matching the Power BI format.
    """
    print("\nGenerating summary report...")

    # CALCULATE STARTING METRICS FOR EACH REP (BEFORE ALIGNMENT)
    # .groupby() groups all accounts by their original assigned rep
    # .agg() performs aggregation functions on each group:
    # 
    # - COL_REP_TYPE: 'first' - take the first rep type value (they're all the same per rep)
    # - COL_TOTAL_REV_2024: 'sum' - add up all 2024 revenue for this rep's accounts
    # - COL_TOTAL_REV_2025: 'sum' - add up all 2025 revenue for this rep's accounts
    # - COL_ACCOUNT_ID: 'count' - count how many accounts this rep has
    # 
    # .reset_index() converts the groupby result back to a regular DataFrame
    # with Assigned_Rep_Name as a column instead of the index
    original_reps = df.groupby(COL_ASSIGNED_REP).agg({
        COL_REP_TYPE: 'first',
        COL_TOTAL_REV_2024: 'sum',
        COL_TOTAL_REV_2025: 'sum',
        COL_ACCOUNT_ID: 'count'
    }).reset_index()
    
    # RENAME COLUMNS TO MATCH OUTPUT FORMAT
    # The .columns assignment renames the columns to more readable names
    original_reps.columns = [
        'Assigned_Rep_Name', 'Rep_Type', '2024 Revenue', '2025 Revenue', '2025 Count of Accounts'
    ]

    # CALCULATE HOW MANY ACCOUNTS EACH REP IS GAINING/LOSING FROM ALIGNMENT
    # This tracks the impact of the territory realignment
    # 
    # ACCOUNTS LEAVING: Filter to only accounts that changed reps, then count
    # how many were originally assigned to each rep
    # df[df['Alignment_Changed']] = only rows where Alignment_Changed is True
    # .groupby(COL_ASSIGNED_REP).size() = count accounts by original rep
    accounts_leaving = df[df['Alignment_Changed']].groupby(COL_ASSIGNED_REP).size()
    
    # ACCOUNTS ARRIVING: Same filter, but count by where they're moving TO
    # .groupby('Aligned_Rep_Name').size() = count accounts by new rep
    accounts_arriving = df[df['Alignment_Changed']].groupby('Aligned_Rep_Name').size()

    # MAP THE COUNTS TO EACH REP'S ROW
    # .map() looks up each rep name in the accounts_leaving/arriving series
    # .fillna(0) replaces NaN with 0 (for reps with no changes)
    original_reps['Accounts_Leaving'] = original_reps['Assigned_Rep_Name'].map(accounts_leaving).fillna(0)
    original_reps['Accounts_Arriving'] = original_reps['Assigned_Rep_Name'].map(accounts_arriving).fillna(0)
    
    # CALCULATE NET CHANGE
    # Positive number = rep is gaining accounts overall
    # Negative number = rep is losing accounts overall
    # Zero = same number of accounts (but they might be different accounts!)
    original_reps['Account Changes due to Alignment'] = (
        original_reps['Accounts_Arriving'] - original_reps['Accounts_Leaving']
    ).astype(int)

    # CALCULATE STATS AFTER ALIGNMENT (WHO ACTUALLY HAS THE ACCOUNTS NOW)
    # Now we group by Aligned_Rep_Name instead of Assigned_Rep_Name
    # This shows the NEW reality after accounts have moved
    # 
    # For each rep's aligned accounts, calculate:
    # - Total 2025 revenue they now have
    # - Total count of accounts they now have
    after_alignment = df.groupby('Aligned_Rep_Name').agg({
        COL_TOTAL_REV_2025: 'sum',
        COL_ACCOUNT_ID: 'count'
    }).reset_index()
    after_alignment.columns = ['Rep_Name', '2025 Revenue (after Re-Assignment)', 'Count of Accounts after Alignment']

    # MERGE THE AFTER-ALIGNMENT STATS WITH THE ORIGINAL REP DATA
    # .merge() is like a SQL JOIN - it combines two DataFrames
    # - left_on='Assigned_Rep_Name': match on this column from original_reps
    # - right_on='Rep_Name': match on this column from after_alignment
    # - how='left': keep all rows from original_reps even if no match
    # This adds the after-alignment columns to our summary table
    original_reps = original_reps.merge(
        after_alignment,
        left_on='Assigned_Rep_Name',
        right_on='Rep_Name',
        how='left'
    )

    # HANDLE REPS WHO LOST ALL THEIR ACCOUNTS
    # If a rep lost all accounts in alignment, they won't be in after_alignment
    # The merge will create NaN values for them
    # We replace NaN with 0 to show they have zero accounts and zero revenue
    original_reps['Count of Accounts after Alignment'] = original_reps['Count of Accounts after Alignment'].fillna(0).astype(int)
    original_reps['2025 Revenue (after Re-Assignment)'] = original_reps['2025 Revenue (after Re-Assignment)'].fillna(0)

    # CALCULATE HOW MUCH REVENUE CHANGED DUE TO ALIGNMENT
    # Subtract the before revenue from the after revenue
    # Positive = rep gained revenue from alignment
    # Negative = rep lost revenue from alignment
    original_reps['Revenue Change (after Re-Assignment)'] = (
        original_reps['2025 Revenue (after Re-Assignment)'] - original_reps['2025 Revenue']
    )

    # COUNT ALL GROUP 3 (LOW REVENUE) ACCOUNTS AFTER ALIGNMENT
    # This is a reference number showing all low-revenue accounts
    # Filter to only LOW_REVENUE tier, then count by aligned rep
    # Not all of these will be removed - only the DESIGN segment ones that aren't new
    all_group3 = df[df[COL_REVENUE_TIER] == FLOOR_REVENUE_TIER].groupby('Aligned_Rep_Name').size()
    original_reps['The Floor (All Group 3 Accounts)'] = original_reps['Assigned_Rep_Name'].map(all_group3).fillna(0).astype(int)

    # COUNT AND CALCULATE REVENUE FOR ACCOUNTS ACTUALLY BEING REMOVED
    # Filter to only accounts where Floor_Removed is True
    # These are the DESIGN + Group 3 + non-2025 accounts
    # For each aligned rep, calculate:
    # - How many accounts are being removed from them
    # - How much 2025 revenue is being removed from them
    floor_removed = df[df['Floor_Removed']].groupby('Aligned_Rep_Name').agg({
        COL_ACCOUNT_ID: 'count',
        COL_TOTAL_REV_2025: 'sum'
    }).reset_index()
    floor_removed.columns = ['Rep_Name', 'The Floor (Design Only, Non New Accounts)', '2025 Revenue for The Floor (Design Only, Non New)']

    # MERGE FLOOR REMOVAL STATS INTO THE SUMMARY
    # Add the floor removal columns to our summary table
    # suffixes=('', '_floor') adds '_floor' to any duplicate column names
    original_reps = original_reps.merge(
        floor_removed,
        left_on='Assigned_Rep_Name',
        right_on='Rep_Name',
        how='left',
        suffixes=('', '_floor')
    )

    # HANDLE REPS WITH NO ACCOUNTS BEING REMOVED
    # If a rep has no floor removals, they won't be in floor_removed DataFrame
    # Replace NaN with 0 to show they have zero accounts/revenue being removed
    original_reps['The Floor (Design Only, Non New Accounts)'] = original_reps['The Floor (Design Only, Non New Accounts)'].fillna(0).astype(int)
    original_reps['2025 Revenue for The Floor (Design Only, Non New)'] = original_reps['2025 Revenue for The Floor (Design Only, Non New)'].fillna(0)

    # CALCULATE FINAL STATS AFTER FLOOR REMOVAL
    # This shows what each rep ends up with after BOTH transformations:
    # 1. Territory alignment (accounts moving between reps)
    # 2. Floor removal (low-value accounts being removed)
    # 
    # ~df['Floor_Removed'] means NOT Floor_Removed (the ~ flips True/False)
    # So this filters to only accounts that are NOT being removed
    # Then we group by aligned rep and calculate final counts and revenue
    final_stats = df[~df['Floor_Removed']].groupby('Aligned_Rep_Name').agg({
        COL_ACCOUNT_ID: 'count',
        COL_TOTAL_REV_2025: 'sum'
    }).reset_index()
    final_stats.columns = ['Rep_Name', '2025 Final Count of Accounts (Floor Removed)', '2025 Final Revenue All Accounts (Floor Removed)']

    # MERGE FINAL STATS INTO THE SUMMARY
    # Add the final account/revenue columns to our summary table
    original_reps = original_reps.merge(
        final_stats,
        left_on='Assigned_Rep_Name',
        right_on='Rep_Name',
        how='left',
        suffixes=('', '_final')
    )

    # HANDLE REPS WHO END UP WITH NO ACCOUNTS
    # If a rep lost all accounts (through alignment or floor removal),
    # they won't be in final_stats, so replace NaN with 0
    original_reps['2025 Final Count of Accounts (Floor Removed)'] = original_reps['2025 Final Count of Accounts (Floor Removed)'].fillna(0).astype(int)
    original_reps['2025 Final Revenue All Accounts (Floor Removed)'] = original_reps['2025 Final Revenue All Accounts (Floor Removed)'].fillna(0)

    # CALCULATE OVERALL DIFFERENCES FROM START TO FINISH
    # This shows the TOTAL impact of both alignment and floor removal combined
    # 
    # Account difference: Final count minus original count
    # Negative = rep lost accounts overall
    # Positive = rep gained accounts overall (rare, since we're removing accounts)
    original_reps['Overall Count of Account Difference'] = (
        original_reps['2025 Final Count of Accounts (Floor Removed)'] - original_reps['2025 Count of Accounts']
    ).astype(int)

    # Revenue difference: Final revenue minus original revenue
    # Shows total revenue impact from both transformations
    original_reps['Revenue Diff Between Start and Final 2025'] = (
        original_reps['2025 Final Revenue All Accounts (Floor Removed)'] - original_reps['2025 Revenue']
    )

    # COUNT NEW 2025 ACCOUNTS WITH ZERO REVENUE
    # These are brand new customers (started in 2025) who haven't generated revenue yet
    # They're important to track because:
    # 1. They're protected from floor removal (new customers get a grace period)
    # 2. They represent future potential revenue
    # 3. Reps need to know which accounts need nurturing
    # 
    # Filter criteria (both must be true with & symbol):
    # - Total_Rev_2025 == 0: No revenue in 2025
    # - Is_New_2025 == True: Started as a customer in 2025
    zero_rev_new_2025 = df[
        (df[COL_TOTAL_REV_2025] == 0) &
        (df['Is_New_2025'])
    ].groupby('Aligned_Rep_Name').size()
    original_reps['Count of 2025 Zero Rev New Accounts (Re-Assigned)'] = original_reps['Assigned_Rep_Name'].map(zero_rev_new_2025).fillna(0).astype(int)

    # Select and order final columns
    summary_columns = [
        'Assigned_Rep_Name',
        'Rep_Type',
        '2024 Revenue',
        '2025 Revenue',
        '2025 Count of Accounts',
        'Account Changes due to Alignment',
        'Count of Accounts after Alignment',
        '2025 Revenue (after Re-Assignment)',
        'Revenue Change (after Re-Assignment)',
        'The Floor (All Group 3 Accounts)',
        'The Floor (Design Only, Non New Accounts)',
        '2025 Revenue for The Floor (Design Only, Non New)',
        '2025 Final Count of Accounts (Floor Removed)',
        '2025 Final Revenue All Accounts (Floor Removed)',
        'Overall Count of Account Difference',
        'Revenue Diff Between Start and Final 2025',
        'Count of 2025 Zero Rev New Accounts (Re-Assigned)'
    ]

    summary_df = original_reps[summary_columns].copy()

    # Sort by rep name
    summary_df = summary_df.sort_values('Assigned_Rep_Name')

    # Validate and enforce data types before saving
    summary_df = validate_and_enforce_dtypes(summary_df)

    # Save to CSV using UTF-8 with BOM so Excel opens it correctly on Windows
    summary_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"Summary report saved to: {output_file}")

    return summary_df


# =============================================================================
# REP DETAIL REPORT GENERATION
# =============================================================================

def generate_rep_detail_report(df: pd.DataFrame, rep_name: str, output_file: str):
    """
    Generate detailed report for a single rep.
    Shows accounts they are keeping and losing, grouped by revenue tier.

    Output format is spreadsheet-friendly with:
    - Summary metrics as horizontal table (labels row, values row)
    - Breakdown by segment as matrix (row labels in col A, column headers in row 1)
    - Account details grouped by revenue tier
    """
    # GET ALL ACCOUNTS FOR THIS SPECIFIC REP (AFTER ALIGNMENT)
    # Filter the DataFrame to only rows where Aligned_Rep_Name matches this rep
    # .copy() creates an independent copy we can work with
    rep_accounts = df[df['Aligned_Rep_Name'] == rep_name].copy()

    # IF REP HAS NO ACCOUNTS, DON'T CREATE A REPORT
    # This can happen if a rep lost all accounts during alignment
    if len(rep_accounts) == 0:
        return

    # CALCULATE SUMMARY METRICS FOR THIS REP
    # These numbers will appear at the top of the detail report
    
    # Total number of accounts after alignment
    total_accounts = len(rep_accounts)
    
    # Sum up all 2025 revenue for this rep's accounts
    total_rev_2025 = rep_accounts[COL_TOTAL_REV_2025].sum()
    
    # Sum up all 2024 revenue (for comparison/context)
    total_rev_2024 = rep_accounts[COL_TOTAL_REV_2024].sum()

    # Count accounts being removed by floor logic
    # .sum() on True/False values counts the True values
    accounts_removed = int(rep_accounts['Floor_Removed'].sum())
    
    # Calculate total revenue from accounts being removed
    # .loc[condition, column] selects rows where condition is True
    revenue_removed = rep_accounts.loc[rep_accounts['Floor_Removed'], COL_TOTAL_REV_2025].sum()

    # Calculate what the rep will actually keep (final numbers)
    accounts_keeping = total_accounts - accounts_removed
    revenue_keeping = total_rev_2025 - revenue_removed

    # Count accounts with zero revenue
    # This helps identify accounts that need attention
    zero_rev_accounts = int((rep_accounts[COL_TOTAL_REV_2025] == 0).sum())
    
    # Count zero-revenue accounts that are also new in 2025
    # These are new customers who haven't generated revenue yet
    zero_rev_new_2025 = int(((rep_accounts[COL_TOTAL_REV_2025] == 0) & (rep_accounts['Is_New_2025'])).sum())

    # Account detail columns (no Growth_Classification per template)
    detail_columns = [
        COL_ACCOUNT_ID,
        COL_ACCOUNT_NAME,
        COL_SEGMENT,
        COL_SUB_SEGMENT,
        COL_TOTAL_REV_2024,
        COL_TOTAL_REV_2025,
        COL_ORDERS_2024,
        COL_ORDERS_2025,
        COL_REVENUE_TIER,
        COL_POTENTIAL_TIER,
        'Segment_Label',
        COL_CUSTOMER_SINCE,
        'Is_New_2025',
        COL_COMPOSITE_SCORE,
        'Floor_Removed'
    ]

    # DEFINE A HELPER FUNCTION TO FORMAT ACCOUNT DATA FOR CSV OUTPUT
    # This ensures all accounts are displayed consistently
    def format_for_output(accounts_df):
        """Format dataframe columns for clean CSV output."""
        # SELECT ONLY THE COLUMNS WE WANT IN THE OUTPUT
        # detail_columns is a list defined above with all the fields we want to show
        out_df = accounts_df[detail_columns].copy()
        
        # FORMAT REVENUE TO 2 DECIMAL PLACES
        # .round(2) rounds to 2 decimal places (dollars and cents)
        out_df[COL_TOTAL_REV_2024] = out_df[COL_TOTAL_REV_2024].round(2)
        out_df[COL_TOTAL_REV_2025] = out_df[COL_TOTAL_REV_2025].round(2)
        
        # FORMAT ORDER COUNTS AS WHOLE NUMBERS (INTEGERS)
        # Orders should never have decimals (you can't have 2.5 orders)
        # .fillna(0) replaces missing values with 0 first
        out_df[COL_ORDERS_2024] = out_df[COL_ORDERS_2024].fillna(0).astype(int)
        out_df[COL_ORDERS_2025] = out_df[COL_ORDERS_2025].fillna(0).astype(int)
        
        # FORMAT BOOLEAN (TRUE/FALSE) VALUES AS UPPERCASE TEXT
        # Python's True/False need to be converted to 'TRUE'/'FALSE' strings
        # .map() replaces values based on a dictionary mapping
        out_df['Is_New_2025'] = out_df['Is_New_2025'].map({True: 'TRUE', False: 'FALSE'})
        out_df['Floor_Removed'] = out_df['Floor_Removed'].map({True: 'TRUE', False: 'FALSE'})
        return out_df

    # CREATE PADDING FOR CSV ROWS
    # The detail report has 15 columns total
    # We need to pad shorter rows with empty columns for consistent formatting
    # ',' * 14 creates a string with 14 commas: ',,,,,,,,,,,,,,'
    empty_cols = ',' * 14

    # OPEN FILE WITH UTF-8 ENCODING
    # encoding='utf-8-sig' ensures special characters are handled properly
    # 'utf-8-sig' adds a BOM (Byte Order Mark) so Excel opens it correctly
    # newline='' prevents extra blank lines in the CSV on Windows
    with open(output_file, 'w', encoding='utf-8-sig', newline='') as f:
        # Write header section
        f.write(f"Rep Detail Report: {rep_name}{empty_cols}\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{empty_cols}\n")
        f.write(f"{empty_cols}\n")

        # Separator
        f.write(f"================================================================================{empty_cols}\n")
        f.write(f"SUMMARY METRICS{empty_cols}\n")
        f.write(f"================================================================================{empty_cols}\n")

        # Summary metrics as horizontal table - labels row then values row
        summary_labels = [
            "Total Accounts (after alignment):",
            "Total 2024 Revenue: ",
            "Total 2025 Revenue:",
            "Accounts Removed (Floor):",
            "Revenue Removed (Floor): ",
            "Final Accounts (Keeping):",
            "Final Revenue (Keeping):",
            "Zero Revenue Accounts:",
            "Zero Revenue New 2025 Accounts: "
        ]
        summary_values = [
            total_accounts,
            round(total_rev_2024, 2),
            round(total_rev_2025, 2),
            accounts_removed,
            round(revenue_removed, 2),
            accounts_keeping,
            round(revenue_keeping, 2),
            zero_rev_accounts,
            zero_rev_new_2025
        ]

        # Write labels row (pad to 15 columns)
        f.write(','.join(summary_labels) + ',' * (15 - len(summary_labels)) + '\n')
        # Write values row (pad to 15 columns)
        f.write(','.join(str(v) for v in summary_values) + ',' * (15 - len(summary_values)) + '\n')

        # Blank rows
        f.write(f"{empty_cols}\n")
        f.write(f"{empty_cols}\n")

        # WRITE BREAKDOWN BY SEGMENT SECTION
        # This shows a summary table of how accounts are distributed across segments
        f.write(f"================================================================================{empty_cols}\n")
        f.write(f"BREAKDOWN BY SEGMENT{empty_cols}\n")
        f.write(f"================================================================================{empty_cols}\n")

        # CALCULATE SEGMENT-LEVEL STATISTICS
        # Group accounts by their Segment_Label (e.g., "Group 1 High Potential")
        # For each segment, calculate:
        # - Count of accounts
        # - Total 2025 revenue
        # - Count being removed (sum of True values in Floor_Removed)
        segment_summary = rep_accounts.groupby('Segment_Label').agg({
            COL_ACCOUNT_ID: 'count',
            COL_TOTAL_REV_2025: 'sum',
            'Floor_Removed': 'sum'
        }).reset_index()
        
        # RENAME COLUMNS FOR READABILITY
        segment_summary.columns = ['Segment', 'Accounts', 'Revenue', 'Removed']
        
        # SORT BY REVENUE (HIGHEST FIRST)
        # ascending=False means sort from largest to smallest
        # This puts the most important segments at the top
        segment_summary = segment_summary.sort_values('Revenue', ascending=False)

        # WRITE HEADER ROW FOR THE SEGMENT TABLE
        # First cell is empty (for segment names), then column headers
        f.write(f",Accounts,Revenue,Removed{','.join([''] * 11)}\n")

        # WRITE DATA ROWS (ONE FOR EACH SEGMENT)
        # .iterrows() loops through each row of the segment_summary DataFrame
        # _ is the index (we don't need it), row contains the data
        for _, row in segment_summary.iterrows():
            f.write(f"{row['Segment']}: ,{int(row['Accounts'])},{round(row['Revenue'], 2)},{int(row['Removed'])}{','.join([''] * 11)}\n")

        # WRITE TOTALS ROW
        # Sum up all the columns to show grand totals
        f.write(f"Totals,{total_accounts},{round(total_rev_2025, 2)},{accounts_removed}{','.join([''] * 11)}\n")

        # Blank row
        f.write(f"{empty_cols}\n")

        # WRITE GROUP 1 ACCOUNTS (HIGH REVENUE - ALL KEEPING)
        # Filter to only HIGH_REVENUE tier accounts
        # These are the most valuable accounts - they're never removed
        group1 = rep_accounts[rep_accounts[COL_REVENUE_TIER] == 'HIGH_REVENUE'].copy()
        if len(group1) > 0:
            f.write(f"================================================================================{empty_cols}\n")
            f.write(f"GROUP 1 (HIGH REVENUE) - KEEPING: {len(group1)} accounts | {group1[COL_TOTAL_REV_2025].sum():.2f} revenue{empty_cols}\n")
            f.write(f"================================================================================{empty_cols}\n")
            # SORT BY REVENUE (HIGHEST FIRST)
            # ascending=False means largest revenue at the top
            group1_sorted = group1.sort_values(COL_TOTAL_REV_2025, ascending=False)
            # FORMAT AND WRITE ACCOUNT DETAILS TO CSV
            # index=False means don't write row numbers
            # encoding is inherited from the file object
            format_for_output(group1_sorted).to_csv(f, index=False, lineterminator='\n')
            f.write(f"{empty_cols}\n")

        # WRITE GROUP 2 ACCOUNTS (MEDIUM REVENUE - ALL KEEPING)
        # Filter to only MID_REVENUE tier accounts
        # These accounts are also protected from floor removal
        group2 = rep_accounts[rep_accounts[COL_REVENUE_TIER] == 'MID_REVENUE'].copy()
        if len(group2) > 0:
            f.write(f"================================================================================{empty_cols}\n")
            f.write(f"GROUP 2 (MID REVENUE) - KEEPING: {len(group2)} accounts | {group2[COL_TOTAL_REV_2025].sum():.2f} revenue{empty_cols}\n")
            f.write(f"================================================================================{empty_cols}\n")
            # SORT BY REVENUE (HIGHEST FIRST)
            group2_sorted = group2.sort_values(COL_TOTAL_REV_2025, ascending=False)
            # FORMAT AND WRITE ACCOUNT DETAILS TO CSV
            # lineterminator='\n' ensures consistent line endings
            format_for_output(group2_sorted).to_csv(f, index=False, lineterminator='\n')
            f.write(f"{empty_cols}\n")

        # WRITE GROUP 3 ACCOUNTS - KEEPING
        # Group 3 (LOW_REVENUE) is special - it's split into keeping and losing
        # 
        # KEEPING criteria (must be LOW_REVENUE AND not removed):
        # - Either NOT in DESIGN segment (BUILD accounts are protected)
        # - OR new in 2025 (new customers get protection)
        # 
        # Filter logic:
        # - Low revenue tier AND
        # - NOT Floor_Removed (the ~ means NOT)
        group3_keeping = rep_accounts[
            (rep_accounts[COL_REVENUE_TIER] == 'LOW_REVENUE') &
            (~rep_accounts['Floor_Removed'])
        ].copy()
        if len(group3_keeping) > 0:
            f.write(f"================================================================================{empty_cols}\n")
            f.write(f"GROUP 3 (LOW REVENUE) - KEEPING: {len(group3_keeping)} accounts | {group3_keeping[COL_TOTAL_REV_2025].sum():.2f} revenue{empty_cols}\n")
            f.write(f"(Non-DESIGN accounts or New 2025 customers - protected from floor){empty_cols}\n")
            f.write(f"================================================================================{empty_cols}\n")
            # SORT BY REVENUE (HIGHEST FIRST)
            group3_keeping_sorted = group3_keeping.sort_values(COL_TOTAL_REV_2025, ascending=False)
            # FORMAT AND WRITE ACCOUNT DETAILS TO CSV
            # lineterminator='\n' ensures consistent line endings
            format_for_output(group3_keeping_sorted).to_csv(f, index=False, lineterminator='\n')
            f.write(f"{empty_cols}\n")

        # WRITE GROUP 3 ACCOUNTS - LOSING (FLOOR REMOVED)
        # These are the accounts being removed from the rep
        # 
        # LOSING criteria (all three must be true):
        # - Low revenue tier (Group 3)
        # - DESIGN segment
        # - NOT new in 2025
        # 
        # This is the "floor" - the lowest-value accounts being removed
        group3_losing = rep_accounts[rep_accounts['Floor_Removed']].copy()
        if len(group3_losing) > 0:
            f.write(f"================================================================================{empty_cols}\n")
            f.write(f"GROUP 3 (LOW REVENUE) - LOSING (FLOOR REMOVED): {len(group3_losing)} accounts | {group3_losing[COL_TOTAL_REV_2025].sum():.2f} revenue{empty_cols}\n")
            f.write(f"(DESIGN segment + Not new in 2025){empty_cols}\n")
            f.write(f"================================================================================{empty_cols}\n")
            # SORT BY REVENUE (HIGHEST FIRST)
            group3_losing_sorted = group3_losing.sort_values(COL_TOTAL_REV_2025, ascending=False)
            # FORMAT AND WRITE ACCOUNT DETAILS TO CSV
            # lineterminator='\n' ensures consistent line endings
            format_for_output(group3_losing_sorted).to_csv(f, index=False, lineterminator='\n')
            f.write(f"{empty_cols}\n")

        # WRITE ZERO REVENUE ACCOUNTS SECTION
        # These accounts deserve special attention regardless of group
        # Zero revenue could mean:
        # - New customer who hasn't ordered yet
        # - Inactive account
        # - Account that needs nurturing
        zero_rev = rep_accounts[rep_accounts[COL_TOTAL_REV_2025] == 0].copy()
        if len(zero_rev) > 0:
            # COUNT HOW MANY ARE REMOVED VS KEEPING
            # Even zero-revenue accounts can be removed if they meet floor criteria
            zero_rev_removed = int(zero_rev['Floor_Removed'].sum())
            zero_rev_keeping = len(zero_rev) - zero_rev_removed
            
            f.write(f"================================================================================{empty_cols}\n")
            f.write(f"ZERO REVENUE ACCOUNTS: {len(zero_rev)} total | {zero_rev_keeping} keeping | {zero_rev_removed} removed{empty_cols}\n")
            f.write(f"================================================================================{empty_cols}\n")
            
            # SORT BY FLOOR_REMOVED (KEEPING FIRST), THEN NEW 2025 (NEW FIRST)
            # ascending=[True, False] means:
            # - Floor_Removed: False (keeping) before True (removed)
            # - Is_New_2025: True (new) before False (old)
            # This groups accounts by priority: keeping new customers first
            zero_rev_sorted = zero_rev.sort_values(['Floor_Removed', 'Is_New_2025'], ascending=[True, False])
            
            # FORMAT AND WRITE ACCOUNT DETAILS TO CSV
            # lineterminator='\n' ensures consistent line endings
            format_for_output(zero_rev_sorted).to_csv(f, index=False, lineterminator='\n')
            f.write(f"{empty_cols}\n")


def generate_all_rep_detail_reports(df: pd.DataFrame, output_dir: str) -> dict:
    """
    Generate detail reports for all reps.
    Returns a dictionary with rep name -> total revenue for validation.
    """
    print("\nGenerating rep detail reports...")

    # CREATE SUBDIRECTORY FOR REP DETAIL REPORTS
    # This keeps the output organized - all rep reports go in one folder
    # os.path.join() safely combines directory paths (works on Windows/Mac/Linux)
    rep_reports_dir = os.path.join(output_dir, 'rep_details')
    os.makedirs(rep_reports_dir, exist_ok=True)

    # GET LIST OF ALL UNIQUE REP NAMES (AFTER ALIGNMENT)
    # .unique() returns an array with each rep name appearing only once
    # This tells us which reps we need to create reports for
    reps = df['Aligned_Rep_Name'].unique()

    # CREATE DICTIONARY TO TRACK TOTALS FOR VALIDATION
    # Will be filled with: {rep_name: total_revenue}
    # Used later to verify all revenue is accounted for
    rep_totals = {}

    # LOOP THROUGH EACH REP AND CREATE THEIR DETAIL REPORT
    for rep in reps:
        # CREATE A SAFE FILENAME FROM THE REP NAME
        # Rep names might have special characters that aren't valid in filenames
        # This loop keeps only alphanumeric and safe characters (' ', '-', '_')
        # Everything else becomes an underscore
        # Example: "John O'Brien" becomes "John_O_Brien"
        safe_name = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in rep)
        
        # CLEAN UP THE FILENAME
        # .strip() removes leading/trailing spaces
        # .replace(' ', '_') converts spaces to underscores
        # Example: " John Doe " becomes "John_Doe"
        safe_name = safe_name.strip().replace(' ', '_')
        
        # BUILD FULL OUTPUT FILE PATH
        # Example: "./reports/rep_details/John_Doe_detail.csv"
        output_file = os.path.join(rep_reports_dir, f"{safe_name}_detail.csv")

        # GENERATE THE ACTUAL REPORT FOR THIS REP
        # This creates the CSV file with all account details
        generate_rep_detail_report(df, rep, output_file)

        # CALCULATE AND STORE THIS REP'S TOTAL REVENUE FOR VALIDATION
        # Filter to this rep's accounts and sum their 2025 revenue
        # Store in dictionary so we can check totals later
        rep_accounts = df[df['Aligned_Rep_Name'] == rep]
        rep_totals[rep] = rep_accounts[COL_TOTAL_REV_2025].sum()

    print(f"Generated {len(reps)} rep detail reports in: {rep_reports_dir}")

    # RETURN THE TOTALS DICTIONARY FOR VALIDATION
    # This allows the main function to verify all revenue is accounted for
    return rep_totals


# =============================================================================
# VALIDATION
# =============================================================================

def validate_totals(df: pd.DataFrame, summary_df: pd.DataFrame, rep_totals: dict):
    """
    Validate that totals match across input, summary, and detail reports.
    """
    print("\n" + "=" * 80)
    print("VALIDATION")
    print("=" * 80)

    # CALCULATE TOTAL 2025 REVENUE FROM INPUT DATA
    # This is our baseline - the original data we started with
    # Sum all 2025 revenue across all accounts in the input file
    input_total = df[COL_TOTAL_REV_2025].sum()
    print(f"Input Total Rev 2025: ${input_total:,.2f}")

    # CALCULATE TOTAL FROM SUMMARY REPORT
    # Sum the '2025 Revenue' column (before alignment)
    # This should equal the input total
    summary_total = summary_df['2025 Revenue'].sum()
    print(f"Summary Report Total (2025 Revenue): ${summary_total:,.2f}")

    # CALCULATE TOTAL FROM REP DETAIL REPORTS
    # rep_totals is a dictionary: {rep_name: total_revenue}
    # sum(dict.values()) adds up all the revenue values
    # This should also equal the input total
    rep_details_total = sum(rep_totals.values())
    print(f"Rep Details Total: ${rep_details_total:,.2f}")

    # CHECK FOR MISMATCHES BETWEEN THE THREE TOTALS
    # All three should be exactly the same (accounting for rounding)
    errors = []

    # ALLOW SMALL DIFFERENCES DUE TO FLOATING POINT ROUNDING
    # Computers can't perfectly represent all decimal numbers
    # 0.01 (1 cent) is a reasonable tolerance for currency
    tolerance = 0.01

    # CHECK IF INPUT MATCHES SUMMARY
    # abs() gets absolute value (distance between numbers)
    # If difference is more than 1 cent, something's wrong
    if abs(input_total - summary_total) > tolerance:
        errors.append(f"MISMATCH: Input ({input_total:,.2f}) != Summary ({summary_total:,.2f}), diff: {input_total - summary_total:,.2f}")

    # CHECK IF INPUT MATCHES REP DETAILS
    if abs(input_total - rep_details_total) > tolerance:
        errors.append(f"MISMATCH: Input ({input_total:,.2f}) != Rep Details ({rep_details_total:,.2f}), diff: {input_total - rep_details_total:,.2f}")

    # REPORT RESULTS
    if errors:
        print("\nVALIDATION ERRORS:")
        for error in errors:
            print(f"  - {error}")
        return False
    else:
        print("\nVALIDATION PASSED: All totals match!")
        return True


# =============================================================================
# MAIN
# =============================================================================

def main():
    # PARSE COMMAND LINE ARGUMENTS
    # When you run this script, you can optionally provide:
    # 1. Input file path
    # 2. Output directory path
    # 
    # sys.argv is a list: [script_name, arg1, arg2, ...]
    # sys.argv[0] is always the script name itself
    # len(sys.argv) tells us how many arguments were provided
    
    # GET INPUT FILE (FIRST ARGUMENT OR DEFAULT)
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    else:
        # DEFAULT: Look for file in current directory
        input_file = 'Account_Segmentation_V3_3_Final.csv'

    # GET OUTPUT DIRECTORY (SECOND ARGUMENT OR DEFAULT)
    if len(sys.argv) > 2:
        output_dir = sys.argv[2]
    else:
        # DEFAULT: Create 'reports' folder in current directory
        output_dir = './reports'

    # CHECK THAT INPUT FILE EXISTS
    # os.path.exists() returns True if file is found, False otherwise
    if not os.path.exists(input_file):
        print(f"ERROR: Input file not found: {input_file}")
        sys.exit(1)  # Exit with error code 1 (indicates failure)

    # CREATE OUTPUT DIRECTORY IF IT DOESN'T EXIST
    # exist_ok=True means don't error if directory already exists
    os.makedirs(output_dir, exist_ok=True)

    # PRINT HEADER AND CONFIGURATION
    print("=" * 80)
    print("ACCOUNT SEGMENTATION REPORT GENERATOR")
    print("=" * 80)
    print(f"Input file: {input_file}")
    print(f"Output directory: {output_dir}")
    print()

    # =========================================================================
    # STEP 1: LOAD AND PREPROCESS DATA
    # =========================================================================
    # Read CSV file and prepare it for processing:
    # - Add missing columns
    # - Clean up blank values
    # - Parse dates
    # - Create helper columns for grouping
    df = load_and_preprocess_data(input_file)

    # =========================================================================
    # STEP 2: APPLY TERRITORY ALIGNMENT
    # =========================================================================
    # Reassign accounts based on rep type:
    # - Inside Sales accounts move to Re-Assigned_Rep_Name
    # - All other accounts stay with Assigned_Rep_Name
    # Result: 'Aligned_Rep_Name' column shows where each account ends up
    df = apply_territory_alignment(df)

    # =========================================================================
    # STEP 3: APPLY FLOOR LOGIC
    # =========================================================================
    # Identify accounts to remove ("setting the floor"):
    # - Group 3 (low revenue)
    # - DESIGN segment
    # - Not new in 2025
    # Result: 'Floor_Removed' column marks accounts being removed
    df = apply_floor_logic(df)

    # =========================================================================
    # STEP 4: GENERATE SUMMARY REPORT
    # =========================================================================
    # Create one row per rep showing:
    # - Starting accounts and revenue
    # - Impact of alignment
    # - Impact of floor removal
    # - Final accounts and revenue
    summary_file = os.path.join(output_dir, 'Rep_Level_Impacts_2025_Summary.csv')
    summary_df = generate_summary_report(df, summary_file)

    # =========================================================================
    # STEP 5: GENERATE REP DETAIL REPORTS
    # =========================================================================
    # Create individual CSV file for each rep showing:
    # - Summary statistics
    # - Account breakdown by segment
    # - Full account details grouped by revenue tier
    # Returns dictionary of rep totals for validation
    rep_totals = generate_all_rep_detail_reports(df, output_dir)

    # =========================================================================
    # STEP 6: VALIDATE TOTALS
    # =========================================================================
    # Verify that revenue totals match across:
    # - Input data
    # - Summary report
    # - Rep detail reports
    # This ensures no accounts or revenue were lost/duplicated
    validation_passed = validate_totals(df, summary_df, rep_totals)

    # PRINT COMPLETION MESSAGE
    print("\n" + "=" * 80)
    print("COMPLETE")
    print("=" * 80)

    # EXIT WITH ERROR CODE IF VALIDATION FAILED
    # This is important for automation - other scripts can detect failure
    if not validation_passed:
        sys.exit(1)


# PYTHON IDIOM: ONLY RUN main() IF THIS FILE IS RUN DIRECTLY
# This check prevents main() from running if this file is imported as a module
# 
# __name__ is a special variable:
# - When file is run directly: __name__ == '__main__'
# - When file is imported: __name__ == 'generate_reports'
# 
# This pattern allows the file to be both:
# 1. A standalone script you can run
# 2. A module other scripts can import functions from
if __name__ == '__main__':
    main()
