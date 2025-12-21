"""
Account Segmentation Report Generator V2
========================================
Generates summary and rep detail reports for the "Set the Floor" initiative.

VERSION 2 CHANGES:
- Uses Final_Rep_Name column as the authoritative "after alignment" state
- Identifies floor exceptions (BTF='Y' but account stayed with original rep)
- Separate section for SF ex BI accounts (dormant accounts in Salesforce)
- Simplified alignment logic - Final_Rep_Name contains all reassignments

This script processes account segmentation data to:
1. Compare before (Assigned_Rep_Name) vs after (Final_Rep_Name) states
2. Identify accounts moved, floor exceptions, and SF ex BI accounts
3. Generate a summary report by rep
4. Generate individual rep detail reports with new sections
5. Validate totals match input data

Usage:
        python generate_reports_v2.py [input_file] [output_directory]

        Defaults:
                input_file: Account_Segmentation_V3_3_Final_SOT_12-21-25.csv
                output_directory: ./reports

Notes about encoding and Excel:
- All generated CSV files are written using UTF-8 with a BOM (`utf-8-sig`).
    This encoding preserves Unicode characters and helps Microsoft Excel
    recognize UTF-8 files correctly when opening on Windows.
- If you open the CSVs in other editors, they will also read UTF-8 correctly.

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
# This section defines all the column names and mappings used throughout the script.
# By centralizing these definitions, we can easily update them if the input file
# structure changes.

# Column names from input file
# These match the headers in the CSV input file exactly
COL_ACCOUNT_ID = 'Customer No.'           # Unique account identifier
COL_ACCOUNT_NAME = 'Account_Name'         # Human-readable account name
COL_SEGMENT = 'Segment'                   # DESIGN, BUILD, RETAIL, etc.
COL_SUB_SEGMENT = 'Sub_Segment'           # More detailed segment classification
COL_ASSIGNED_REP = 'Assigned_Rep_Name'    # BEFORE alignment - original rep assignment
COL_REASSIGNED_REP = 'Re-Assigned_Rep_Name'  # (Legacy - not used in V2)
COL_REP_TYPE = 'Rep_Type'                 # Inside Sales, Account Managers, Independent rep, etc.
COL_CUSTOMER_SINCE = 'Customer_Since_Date'   # When customer started (M/D/YYYY format)
COL_REVENUE_TIER = 'Revenue_Tier'         # HIGH_REVENUE, MID_REVENUE, or LOW_REVENUE
COL_POTENTIAL_TIER = 'Potential_Tier'     # HIGH_POTENTIAL, MID_POTENTIAL, or LOW_POTENTIAL
COL_TOTAL_REV_2024 = 'Total_Rev_2024'     # 2024 annual revenue
COL_TOTAL_REV_2025 = 'Total_Rev_2025'     # 2025 annual revenue
COL_ORDERS_2024 = 'Orders_2024'           # Number of orders placed in 2024
COL_ORDERS_2025 = 'Orders_2025'           # Number of orders placed in 2025
COL_FINAL_REP = 'Final_Rep_Name'          # AFTER alignment - final rep assignment (the authoritative field)
COL_BTF = 'BTF'                           # Below The Floor flag (Y/N) - accounts flagged for potential removal
COL_SF_EX_BI = 'In SF ex BI'              # In Salesforce but not in BI (Y/N) - dormant accounts

# MAPPINGS FOR READABLE OUTPUT
# These dictionaries convert internal codes to human-readable labels for reports

# Revenue tier to group name mapping
# Used to display "Group 1", "Group 2", "Group 3" in reports
REVENUE_TIER_TO_GROUP = {
    'HIGH_REVENUE': 'Group 1',        # High revenue accounts (most valuable)
    'MID_REVENUE': 'Group 2',         # Medium revenue accounts
    'LOW_REVENUE': 'Group 3'          # Low revenue accounts (below the floor candidates)
}

# Potential tier to readable label mapping
POTENTIAL_TIER_TO_LABEL = {
    'HIGH_POTENTIAL': 'High Potential',       # High growth potential
    'MID_POTENTIAL': 'Medium Potential',      # Medium growth potential
    'LOW_POTENTIAL': 'Low Potential'          # Low growth potential
}

# FLOOR REMOVAL CRITERIA
# These constants define which accounts are candidates for being removed from a rep
FLOOR_SEGMENT = 'DESIGN'              # Only DESIGN segment accounts can be removed
FLOOR_REVENUE_TIER = 'LOW_REVENUE'    # Only Group 3 (low revenue) accounts
PROTECTED_YEAR = 2025                 # New 2025 customers are protected from removal


# =============================================================================
# DATA LOADING AND PREPROCESSING
# =============================================================================

def load_and_preprocess_data(input_file: str) -> pd.DataFrame:
    """
    Load the input CSV and perform preprocessing:
    - Clean up rep names (replace blanks with 'House')
    - Parse Customer_Since_Date and extract year
    - Add computed columns for grouping
    - Identify floor exceptions and SF ex BI accounts
    
    This function is the first step in the report generation pipeline.
    It reads the raw CSV and prepares it for analysis by:
    1. Loading the CSV file into a pandas DataFrame
    2. Cleaning up missing/blank values
    3. Creating helper columns for grouping and analysis
    4. Identifying special account types (moved, exceptions, dormant)
    
    Args:
        input_file (str): Path to the input CSV file
        
    Returns:
        pd.DataFrame: Preprocessed data ready for report generation
    """
    print(f"Loading data from: {input_file}")
    
    # READ THE CSV FILE INTO MEMORY
    # pd.read_csv() creates a pandas DataFrame - essentially a spreadsheet in memory
    # Each row = one account, each column = one piece of data about that account
    df = pd.read_csv(input_file)

    # COUNT HOW MANY ACCOUNTS WE LOADED
    # This is a sanity check - tells us how much data we're working with
    original_count = len(df)
    print(f"Loaded {original_count:,} accounts")

    # CLEAN UP ASSIGNED REP NAMES (BEFORE state)
    # Some accounts might have no assigned rep (blank or null/NaN values)
    # .fillna() replaces None/NaN with the specified value
    # .str.strip() removes leading/trailing whitespace
    # We replace all blanks with "House" which represents unassigned/house accounts
    # This ensures every account has a rep assigned for grouping purposes
    df[COL_ASSIGNED_REP] = df[COL_ASSIGNED_REP].fillna('House')
    df.loc[df[COL_ASSIGNED_REP].str.strip() == '', COL_ASSIGNED_REP] = 'House'

    # CLEAN UP FINAL REP NAMES (AFTER state)
    # Same logic as above for the Final_Rep_Name column
    # This is the authoritative column that says where each account ends up after all moves
    df[COL_FINAL_REP] = df[COL_FINAL_REP].fillna('House')
    df.loc[df[COL_FINAL_REP].str.strip() == '', COL_FINAL_REP] = 'House'

    # PARSE CUSTOMER START DATE AND EXTRACT THE YEAR
    # The input file has dates in M/D/YYYY format (e.g., "01/15/2025")
    # pd.to_datetime() converts text strings to actual date objects
    # format='%m/%d/%Y' tells it how to parse the text
    # errors='coerce' means if a date is invalid, set it to NaN instead of crashing
    # .dt.year extracts just the year portion (e.g., 2025 from "01/15/2025")
    df['Customer_Since_Year'] = pd.to_datetime(
        df[COL_CUSTOMER_SINCE], format='%m/%d/%Y', errors='coerce'
    ).dt.year

    # IDENTIFY NEW 2025 CUSTOMERS (PROTECTED FROM REMOVAL)
    # Create a True/False column: True if customer started in 2025, False otherwise
    # New customers get special protection - they won't be removed from reps
    # even if they meet other criteria for floor removal
    df['Is_New_2025'] = df['Customer_Since_Year'] == PROTECTED_YEAR

    # CREATE READABLE REVENUE GROUP LABELS
    # .map() uses the REVENUE_TIER_TO_GROUP dictionary to convert codes to readable names
    # 'HIGH_REVENUE' becomes 'Group 1', etc.
    # .fillna('Unknown') handles any unexpected values that aren't in the dictionary
    df['Revenue_Group'] = df[COL_REVENUE_TIER].map(REVENUE_TIER_TO_GROUP).fillna('Unknown')

    # CREATE READABLE POTENTIAL LABELS
    # Same approach - convert codes like 'HIGH_POTENTIAL' to readable labels
    df['Potential_Label'] = df[COL_POTENTIAL_TIER].map(POTENTIAL_TIER_TO_LABEL).fillna('Unknown')

    # CREATE COMBINED SEGMENT LABELS FOR DISPLAY
    # Concatenate Revenue_Group + Potential_Label
    # Example: "Group 1" + " " + "High Potential" = "Group 1 High Potential"
    # This makes reports easier to read and accounts easier to categorize
    df['Segment_Label'] = df['Revenue_Group'] + ' ' + df['Potential_Label']

    # FILL MISSING REVENUE AND ORDER VALUES WITH ZERO
    # Some accounts might have NaN (Not a Number) values for revenue or orders
    # NaN causes problems in calculations, so we replace with 0
    # This is safe because missing revenue/orders = no revenue/orders
    df[COL_TOTAL_REV_2024] = df[COL_TOTAL_REV_2024].fillna(0)
    df[COL_TOTAL_REV_2025] = df[COL_TOTAL_REV_2025].fillna(0)
    df[COL_ORDERS_2024] = df[COL_ORDERS_2024].fillna(0)
    df[COL_ORDERS_2025] = df[COL_ORDERS_2025].fillna(0)

    # STANDARDIZE BTF COLUMN (Y/N flag for "Below The Floor")
    # BTF='Y' means this account was flagged for potential removal
    # .fillna('N') treats missing values as 'N' (not flagged)
    # .str.upper() converts 'y' to 'Y' and 'n' to 'N' for consistency
    df[COL_BTF] = df[COL_BTF].fillna('N').str.upper()
    
    # STANDARDIZE SF EX BI COLUMN (Y/N flag for "Salesforce but not in BI")
    # In SF ex BI='Y' means account exists in Salesforce but has no transaction data in BI
    # These are essentially dormant accounts we want to track separately
    df[COL_SF_EX_BI] = df[COL_SF_EX_BI].fillna('N').str.upper()

    # IDENTIFY ACCOUNT MOVEMENTS
    # Compare Assigned_Rep_Name (where it started) vs Final_Rep_Name (where it ended)
    # If they're different, the account moved during alignment
    # This creates a boolean column: True if moved, False if stayed
    df['Account_Moved'] = df[COL_ASSIGNED_REP] != df[COL_FINAL_REP]

    # IDENTIFY FLOOR EXCEPTIONS
    # Special case: BTF='Y' (flagged for removal) BUT account is staying with original rep
    # This happens when we made an exception and decided to keep the account despite the flag
    # Logic: BTF='Y' AND Assigned_Rep_Name == Final_Rep_Name
    # The & symbol means "AND" - both conditions must be true
    df['Floor_Exception'] = (df[COL_BTF] == 'Y') & (df[COL_ASSIGNED_REP] == df[COL_FINAL_REP])

    # IDENTIFY FLOOR REMOVED ACCOUNTS
    # These accounts were flagged for removal (BTF='Y') AND they moved away from original rep
    # Logic: BTF='Y' AND Assigned_Rep_Name != Final_Rep_Name
    # The account is literally being removed from the original rep's list
    df['Floor_Removed'] = (df[COL_BTF] == 'Y') & (df[COL_ASSIGNED_REP] != df[COL_FINAL_REP])

    # IDENTIFY SF EX BI ACCOUNTS (DORMANT ACCOUNTS)
    # These accounts exist in Salesforce but not in BI (no transaction data)
    # We track them separately because they're essentially inactive
    df['Is_SF_Ex_BI'] = df[COL_SF_EX_BI] == 'Y'

    # CALCULATE AND DISPLAY TOTAL 2025 REVENUE
    # .sum() adds up all values in the column
    # This gives us a baseline total to validate against at the end
    # We use :,.2f format to display with comma thousands separators and 2 decimal places
    total_rev_2025 = df[COL_TOTAL_REV_2025].sum()
    print(f"Total Rev 2025: ${total_rev_2025:,.2f}")
    
    # DISPLAY KEY STATISTICS FOR DEBUGGING/MONITORING
    # On boolean columns, .sum() counts the True values (True=1, False=0)
    # This tells us how many accounts fall into each category
    moved_count = df['Account_Moved'].sum()          # How many accounts changed reps?
    exception_count = df['Floor_Exception'].sum()    # How many exceptions were made?
    removed_count = df['Floor_Removed'].sum()        # How many accounts were removed?
    sf_ex_bi_count = df['Is_SF_Ex_BI'].sum()         # How many dormant accounts?
    
    print(f"Accounts moved: {moved_count:,}")
    print(f"Floor exceptions (BTF='Y' but kept): {exception_count:,}")
    print(f"Floor removed (BTF='Y' and moved): {removed_count:,}")
    print(f"SF ex BI accounts: {sf_ex_bi_count:,}")

    # RETURN PREPROCESSED DATA
    # At this point, the DataFrame is clean and has all helper columns added
    # Ready for the next phase: generating reports
    return df


# =============================================================================
# SUMMARY REPORT GENERATION
# =============================================================================
# This section creates a summary report showing before/after comparison for each rep.
# The report shows:
# - How many accounts/revenue each rep had BEFORE (Assigned_Rep_Name)
# - Account movements (moved in vs moved out)
# - How many accounts/revenue each rep has AFTER (Final_Rep_Name)
# - Special account types (floor exceptions, SF ex BI, zero revenue)

def generate_summary_report(df: pd.DataFrame, output_dir: str) -> pd.DataFrame:
    """
    Generate summary report showing before/after comparison by rep.
    
    This is the executive-level report that shows the impact of the alignment
    and floor logic for each rep.
    
    BEFORE STATE: Uses Assigned_Rep_Name (original assignments)
    AFTER STATE: Uses Final_Rep_Name (final assignments after moves)
    
    Args:
        df (pd.DataFrame): Preprocessed data from load_and_preprocess_data()
        output_dir (str): Directory where output files should be saved
        
    Returns:
        pd.DataFrame: Summary data (also saved to CSV)
    """
    print("\\nGenerating summary report...")
    
    # GET UNIQUE LIST OF ALL REPS (BOTH BEFORE AND AFTER STATES)
    # .unique() finds all distinct values in a column
    # | (pipe) is the "union" operator - combines two sets
    # We need both because:
    # - Some reps might have had accounts but gave them all away (lose all accounts)
    # - Some reps might not have had accounts but received some (new accounts)
    # .sorted() puts reps in alphabetical order for cleaner output
    all_reps = sorted(set(df[COL_ASSIGNED_REP].unique()) | set(df[COL_FINAL_REP].unique()))
    
# BUILD SUMMARY DATA ROW BY ROW
    # We'll create one row per rep showing their before/after state
    summary_rows = []
    
    for rep in all_reps:
        # ===== BEFORE STATE (Assigned_Rep_Name) =====
        # Filter to all accounts that were ORIGINALLY assigned to this rep
        # df[df[COL_ASSIGNED_REP] == rep] is like a WHERE clause: "where Assigned_Rep_Name = rep"
        before_accounts = df[df[COL_ASSIGNED_REP] == rep]
        before_count = len(before_accounts)                          # How many accounts before?
        before_rev_2024 = before_accounts[COL_TOTAL_REV_2024].sum()  # Total 2024 revenue before
        before_rev_2025 = before_accounts[COL_TOTAL_REV_2025].sum()  # Total 2025 revenue before
        
        # ===== AFTER STATE (Final_Rep_Name) =====
        # Filter to all accounts this rep has AFTER all the moves
        # These are their final accounts
        after_accounts = df[df[COL_FINAL_REP] == rep]
        after_count = len(after_accounts)                            # How many accounts after?
        after_rev_2024 = after_accounts[COL_TOTAL_REV_2024].sum()    # Total 2024 revenue after
        after_rev_2025 = after_accounts[COL_TOTAL_REV_2025].sum()    # Total 2025 revenue after

        # ===== MOVEMENT ANALYSIS =====
        # Show which accounts moved IN to this rep and which moved OUT
        
        # Accounts moved IN: Final_Rep = this rep AND Assigned_Rep != this rep
        # These are accounts from OTHER reps that came TO this rep
        accounts_moved_in = len(df[(df[COL_FINAL_REP] == rep) & (df[COL_ASSIGNED_REP] != rep)])
        
        # Accounts moved OUT: Assigned_Rep = this rep AND Final_Rep != this rep
        # These are accounts this rep gave AWAY to other reps
        accounts_moved_out = len(df[(df[COL_ASSIGNED_REP] == rep) & (df[COL_FINAL_REP] != rep)])
        
        # Net change: positive = gained accounts, negative = lost accounts
        net_account_change = accounts_moved_in - accounts_moved_out
        
        # Same logic but for 2025 revenue instead of account counts
        rev_moved_in = df[(df[COL_FINAL_REP] == rep) & (df[COL_ASSIGNED_REP] != rep)][COL_TOTAL_REV_2025].sum()
        rev_moved_out = df[(df[COL_ASSIGNED_REP] == rep) & (df[COL_FINAL_REP] != rep)][COL_TOTAL_REV_2025].sum()
        net_rev_change = rev_moved_in - rev_moved_out

        # ===== SPECIAL ACCOUNT TYPES (for this rep's FINAL accounts) =====
        
        # FLOOR EXCEPTIONS: BTF='Y' but account stayed with rep
        # These are accounts flagged for removal but we made exceptions to keep them
        floor_exceptions = after_accounts[after_accounts['Floor_Exception']]
        exception_count = len(floor_exceptions)
        exception_rev = floor_exceptions[COL_TOTAL_REV_2025].sum()
        
        # SF EX BI ACCOUNTS: Dormant accounts in Salesforce but not in BI
        # These have no transaction history but exist in the CRM
        sf_ex_bi = after_accounts[after_accounts['Is_SF_Ex_BI']]
        sf_ex_bi_count = len(sf_ex_bi)
        sf_ex_bi_rev = sf_ex_bi[COL_TOTAL_REV_2025].sum()
        
        # ZERO REVENUE ACCOUNTS: 2025 revenue = $0
        # These might be new customers or inactive accounts
        zero_rev_all = after_accounts[after_accounts[COL_TOTAL_REV_2025] == 0]
        zero_rev_count = len(zero_rev_all)
        zero_rev_new_2025 = len(zero_rev_all[zero_rev_all['Is_New_2025']])  # New 2025 customers with $0 revenue

        # ===== GET REP TYPE FOR DISPLAY =====
        # Rep type should be consistent for a given rep (all their accounts have same type)
        # We grab the first account's rep type as a representative sample
        # .iloc[0] gets the first row, if the rep had accounts before
        rep_type = ''
        if before_count > 0:
            # If this rep had accounts originally, get rep type from those accounts
            rep_type = before_accounts[COL_REP_TYPE].iloc[0]
        elif after_count > 0:
            # If this rep only has accounts after (received accounts), get rep type from those
            rep_type = after_accounts[COL_REP_TYPE].iloc[0]
        
        # BUILD ONE ROW OF SUMMARY DATA
        # Each key-value pair becomes a column in the output report
        # .round(2) ensures revenue values show only 2 decimal places
        summary_rows.append({
            'Rep_Name': rep,                                    # Rep identifier
            'Rep_Type': rep_type,                               # Account Managers, Inside Sales, etc.
            
            # ===== BEFORE STATE COLUMNS =====
            'Before_Accounts': before_count,                    # Accounts originally assigned to this rep
            'Before_Rev_2024': round(before_rev_2024, 2),       # Their 2024 revenue (before alignment)
            'Before_Rev_2025': round(before_rev_2025, 2),       # Their 2025 revenue (before alignment)
            
            # ===== MOVEMENT COLUMNS =====
            'Accounts_Moved_In': accounts_moved_in,             # Accounts this rep gained from others
            'Accounts_Moved_Out': accounts_moved_out,           # Accounts this rep gave away
            'Net_Account_Change': net_account_change,           # Positive = gained, negative = lost
            'Rev_Moved_In': round(rev_moved_in, 2),             # 2025 revenue of accounts moved in
            'Rev_Moved_Out': round(rev_moved_out, 2),           # 2025 revenue of accounts moved out
            'Net_Rev_Change': round(net_rev_change, 2),         # Net revenue impact from moves
            
            # ===== AFTER STATE COLUMNS =====
            'After_Accounts': after_count,                      # Final account count (after all moves)
            'After_Rev_2024': round(after_rev_2024, 2),         # Final 2024 revenue (for reference)
            'After_Rev_2025': round(after_rev_2025, 2),         # Final 2025 revenue (after all moves)
            
            # ===== FLOOR EXCEPTIONS COLUMNS =====
            'Floor_Exception_Accounts': exception_count,        # BTF='Y' but kept (exceptions)
            'Floor_Exception_Rev': round(exception_rev, 2),     # Revenue of those exceptions
            
            # ===== SF EX BI ACCOUNTS (DORMANT) COLUMNS =====
            'SF_Ex_BI_Accounts': sf_ex_bi_count,                # Accounts in Salesforce but not BI
            'SF_Ex_BI_Rev': round(sf_ex_bi_rev, 2),             # Revenue of those dormant accounts
            
            # ===== ZERO REVENUE COLUMNS =====
            'Zero_Rev_Accounts': zero_rev_count,                # Accounts with $0 2025 revenue
            'Zero_Rev_New_2025': zero_rev_new_2025              # Of those, how many are new 2025 customers
        })
    
    # CREATE DATAFRAME FROM ALL ROWS
    # pd.DataFrame(list_of_dicts) converts a list of dictionaries into a table
    summary_df = pd.DataFrame(summary_rows)
    
    # WRITE SUMMARY TO CSV FILE
    # os.path.join() builds the full file path across platforms (Windows/Mac/Linux)
    # .to_csv() writes the DataFrame to a CSV file
    # index=False means don't write row numbers as a column
    # encoding='utf-8-sig' ensures special characters work and Excel recognizes the file
    output_file = os.path.join(output_dir, 'Rep_Level_Impacts_2025_Summary_V2.csv')
    summary_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"Summary report saved to: {output_file}")
    
    # RETURN THE SUMMARY DATAFRAME
    # This is passed to validation() and stored for later reference
    return summary_df


# =============================================================================
# REP DETAIL REPORT GENERATION
# =============================================================================
# This section generates one detailed report file per rep.
# Each report shows all the accounts assigned to that rep (using Final_Rep_Name)
# and breaks them down by segment, group, and status.
# These are the detailed reports that reps use to understand their account list.

def generate_rep_detail_report(df: pd.DataFrame, rep_name: str, output_dir: str):
    """
    Generate detailed report for a single rep showing all their accounts.
    
    This function creates a detailed CSV report for one rep showing:
    - Summary metrics at the top
    - Breakdown by segment
    - Account listings organized by group (1, 2, 3)
    - Special sections for exceptions, dormant accounts, and zero-revenue accounts
    
    Uses Final_Rep_Name to determine which accounts belong to the rep (AFTER state).
    
    Args:
        df (pd.DataFrame): Full preprocessed dataset
        rep_name (str): The rep's name to generate a report for
        output_dir (str): Directory where the report should be saved
    """
    # FILTER TO ACCOUNTS ASSIGNED TO THIS REP (AFTER ALIGNMENT)
    # Final_Rep_Name is the authoritative source - these are the rep's actual accounts
    # .copy() creates a separate copy so we don't modify the original dataframe
    rep_accounts = df[df[COL_FINAL_REP] == rep_name].copy()
    
    if len(rep_accounts) == 0:
        return
    
    # CALCULATE SUMMARY METRICS
    total_accounts = len(rep_accounts)
    total_rev_2024 = rep_accounts[COL_TOTAL_REV_2024].sum()
    total_rev_2025 = rep_accounts[COL_TOTAL_REV_2025].sum()
    
    # ACCOUNTS THAT MOVED TO THIS REP
    accounts_moved_in = len(rep_accounts[rep_accounts[COL_ASSIGNED_REP] != rep_name])
    rev_moved_in = rep_accounts[rep_accounts[COL_ASSIGNED_REP] != rep_name][COL_TOTAL_REV_2025].sum()
    
    # FLOOR EXCEPTIONS
    floor_exceptions = rep_accounts[rep_accounts['Floor_Exception']]
    exception_count = len(floor_exceptions)
    exception_rev = floor_exceptions[COL_TOTAL_REV_2025].sum()
    
    # SF EX BI ACCOUNTS
    sf_ex_bi = rep_accounts[rep_accounts['Is_SF_Ex_BI']]
    sf_ex_bi_count = len(sf_ex_bi)
    sf_ex_bi_rev = sf_ex_bi[COL_TOTAL_REV_2025].sum()
    
    # ZERO REVENUE ACCOUNTS
    zero_rev = rep_accounts[rep_accounts[COL_TOTAL_REV_2025] == 0]
    zero_rev_count = len(zero_rev)
    zero_rev_new_2025 = len(zero_rev[zero_rev['Is_New_2025']])
    
    # CREATE OUTPUT DIRECTORY IF IT DOESN'T EXIST
    detail_dir = os.path.join(output_dir, 'rep_details')
    os.makedirs(detail_dir, exist_ok=True)
    
    # CREATE SAFE FILENAME (replace spaces and special chars)
    safe_name = rep_name.replace(' ', '_').replace('/', '_').replace('\\', '_')
    output_file = os.path.join(detail_dir, f'{safe_name}_detail_v2.csv')
    
    # HELPER FUNCTION TO FORMAT DATAFRAME FOR OUTPUT
    def format_for_output(out_df):
        """Format DataFrame columns for CSV output"""
        out_df = out_df.copy()
        
        # SELECT AND ORDER COLUMNS
        output_columns = [
            COL_ACCOUNT_ID, COL_ACCOUNT_NAME, COL_SEGMENT, COL_SUB_SEGMENT,
            'Segment_Label', COL_ASSIGNED_REP, COL_FINAL_REP,
            COL_TOTAL_REV_2024, COL_TOTAL_REV_2025,
            COL_ORDERS_2024, COL_ORDERS_2025,
            'Is_New_2025', COL_BTF, 'Floor_Exception', 'Floor_Removed',
            COL_SF_EX_BI
        ]
        
        # Keep only columns that exist
        output_columns = [col for col in output_columns if col in out_df.columns]
        out_df = out_df[output_columns]
        
        # FORMAT REVENUE TO 2 DECIMAL PLACES
        out_df[COL_TOTAL_REV_2024] = out_df[COL_TOTAL_REV_2024].round(2)
        out_df[COL_TOTAL_REV_2025] = out_df[COL_TOTAL_REV_2025].round(2)
        
        # FORMAT ORDER COUNTS AS WHOLE NUMBERS
        out_df[COL_ORDERS_2024] = out_df[COL_ORDERS_2024].fillna(0).astype(int)
        out_df[COL_ORDERS_2025] = out_df[COL_ORDERS_2025].fillna(0).astype(int)
        
        # FORMAT BOOLEAN VALUES AS UPPERCASE TEXT
        out_df['Is_New_2025'] = out_df['Is_New_2025'].map({True: 'TRUE', False: 'FALSE'})
        out_df['Floor_Exception'] = out_df['Floor_Exception'].map({True: 'TRUE', False: 'FALSE'})
        out_df['Floor_Removed'] = out_df['Floor_Removed'].map({True: 'TRUE', False: 'FALSE'})
        
        return out_df
    
    # CREATE PADDING FOR CSV ROWS (16 columns total in V2)
    empty_cols = ',' * 15
    
    # OPEN FILE WITH UTF-8 ENCODING
    with open(output_file, 'w', encoding='utf-8-sig', newline='') as f:
        # WRITE HEADER SECTION
        f.write(f"Rep Detail Report V2: {rep_name}{empty_cols}\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{empty_cols}\n")
        f.write(f"{empty_cols}\n")
        
        # SEPARATOR
        f.write(f"================================================================================{empty_cols}\n")
        f.write(f"SUMMARY METRICS{empty_cols}\n")
        f.write(f"================================================================================{empty_cols}\n")
        
        # SUMMARY METRICS
        summary_labels = [
            "Total Accounts:",
            "Total 2024 Revenue:",
            "Total 2025 Revenue:",
            "Accounts Moved In:",
            "Revenue Moved In:",
            "Floor Exceptions:",
            "Floor Exception Rev:",
            "SF ex BI Accounts:",
            "SF ex BI Rev:",
            "Zero Rev Accounts:",
            "Zero Rev New 2025:"
        ]
        summary_values = [
            total_accounts,
            round(total_rev_2024, 2),
            round(total_rev_2025, 2),
            accounts_moved_in,
            round(rev_moved_in, 2),
            exception_count,
            round(exception_rev, 2),
            sf_ex_bi_count,
            round(sf_ex_bi_rev, 2),
            zero_rev_count,
            zero_rev_new_2025
        ]
        
        # Write labels and values rows
        f.write(','.join(summary_labels) + ',' * (16 - len(summary_labels)) + '\n')
        f.write(','.join(str(v) for v in summary_values) + ',' * (16 - len(summary_values)) + '\n')
        
        # Blank rows
        f.write(f"{empty_cols}\n")
        f.write(f"{empty_cols}\n")
        
        # BREAKDOWN BY SEGMENT
        f.write(f"================================================================================{empty_cols}\n")
        f.write(f"BREAKDOWN BY SEGMENT{empty_cols}\n")
        f.write(f"================================================================================{empty_cols}\n")
        
        segment_summary = rep_accounts.groupby('Segment_Label').agg({
            COL_ACCOUNT_ID: 'count',
            COL_TOTAL_REV_2025: 'sum',
            'Floor_Exception': 'sum',
            'Floor_Removed': 'sum'
        }).reset_index()
        
        segment_summary.columns = ['Segment', 'Accounts', 'Revenue', 'Exceptions', 'Removed']
        segment_summary = segment_summary.sort_values('Revenue', ascending=False)
        
        # Write segment table header
        f.write(f",Accounts,Revenue,Exceptions,Removed{','.join([''] * 11)}\n")
        
        # Write segment data rows
        for _, row in segment_summary.iterrows():
            f.write(f"{row['Segment']}: ,{int(row['Accounts'])},{round(row['Revenue'], 2)},{int(row['Exceptions'])},{int(row['Removed'])}{','.join([''] * 11)}\n")
        
        # Write totals row
        f.write(f"Totals,{total_accounts},{round(total_rev_2025, 2)},{exception_count},0{','.join([''] * 11)}\n")
        
        # Blank row
        f.write(f"{empty_cols}\n")
        
        # GROUP 1 (HIGH REVENUE) - KEEPING
        group1 = rep_accounts[rep_accounts[COL_REVENUE_TIER] == 'HIGH_REVENUE'].copy()
        if len(group1) > 0:
            f.write(f"================================================================================{empty_cols}\n")
            f.write(f"GROUP 1 (HIGH REVENUE) - KEEPING: {len(group1)} accounts | ${group1[COL_TOTAL_REV_2025].sum():.2f} revenue{empty_cols}\n")
            f.write(f"================================================================================{empty_cols}\n")
            group1_sorted = group1.sort_values(COL_TOTAL_REV_2025, ascending=False)
            format_for_output(group1_sorted).to_csv(f, index=False, lineterminator='\n')
            f.write(f"{empty_cols}\n")
        
        # GROUP 2 (MID REVENUE) - KEEPING
        group2 = rep_accounts[rep_accounts[COL_REVENUE_TIER] == 'MID_REVENUE'].copy()
        if len(group2) > 0:
            f.write(f"================================================================================{empty_cols}\n")
            f.write(f"GROUP 2 (MID REVENUE) - KEEPING: {len(group2)} accounts | ${group2[COL_TOTAL_REV_2025].sum():.2f} revenue{empty_cols}\n")
            f.write(f"================================================================================{empty_cols}\n")
            group2_sorted = group2.sort_values(COL_TOTAL_REV_2025, ascending=False)
            format_for_output(group2_sorted).to_csv(f, index=False, lineterminator='\n')
            f.write(f"{empty_cols}\n")
        
        # GROUP 3 (LOW REVENUE) - KEEPING (NO EXCEPTION, NO FLOOR REMOVAL)
        group3_keeping = rep_accounts[
            (rep_accounts[COL_REVENUE_TIER] == 'LOW_REVENUE') &
            (~rep_accounts['Floor_Exception']) &
            (~rep_accounts['Floor_Removed'])
        ].copy()
        if len(group3_keeping) > 0:
            f.write(f"================================================================================{empty_cols}\n")
            f.write(f"GROUP 3 (LOW REVENUE) - KEEPING: {len(group3_keeping)} accounts | ${group3_keeping[COL_TOTAL_REV_2025].sum():.2f} revenue{empty_cols}\n")
            f.write(f"(Protected accounts - not flagged for floor removal){empty_cols}\n")
            f.write(f"================================================================================{empty_cols}\n")
            group3_keeping_sorted = group3_keeping.sort_values(COL_TOTAL_REV_2025, ascending=False)
            format_for_output(group3_keeping_sorted).to_csv(f, index=False, lineterminator='\n')
            f.write(f"{empty_cols}\n")
        
        # GROUP 3 (LOW REVENUE) - FLOOR EXCEPTIONS (BTF='Y' BUT KEPT)
        if len(floor_exceptions) > 0:
            f.write(f"================================================================================{empty_cols}\n")
            f.write(f"GROUP 3 (LOW REVENUE) - FLOOR EXCEPTIONS: {len(floor_exceptions)} accounts | ${floor_exceptions[COL_TOTAL_REV_2025].sum():.2f} revenue{empty_cols}\n")
            f.write(f"(BTF='Y' but account stayed with original rep - EXCEPTION MADE){empty_cols}\n")
            f.write(f"================================================================================{empty_cols}\n")
            exceptions_sorted = floor_exceptions.sort_values(COL_TOTAL_REV_2025, ascending=False)
            format_for_output(exceptions_sorted).to_csv(f, index=False, lineterminator='\n')
            f.write(f"{empty_cols}\n")
        
        # SF EX BI ACCOUNTS (DORMANT - IN SALESFORCE BUT NOT BI)
        if len(sf_ex_bi) > 0:
            f.write(f"================================================================================{empty_cols}\n")
            f.write(f"SF EX BI ACCOUNTS (DORMANT): {len(sf_ex_bi)} accounts | ${sf_ex_bi[COL_TOTAL_REV_2025].sum():.2f} revenue{empty_cols}\n")
            f.write(f"(In Salesforce but not in BI - no transaction data){empty_cols}\n")
            f.write(f"================================================================================{empty_cols}\n")
            sf_ex_bi_sorted = sf_ex_bi.sort_values(COL_TOTAL_REV_2025, ascending=False)
            format_for_output(sf_ex_bi_sorted).to_csv(f, index=False, lineterminator='\n')
            f.write(f"{empty_cols}\n")
        
        # ZERO REVENUE ACCOUNTS
        if len(zero_rev) > 0:
            f.write(f"================================================================================{empty_cols}\n")
            f.write(f"ZERO REVENUE ACCOUNTS: {len(zero_rev)} accounts ({zero_rev_new_2025} are new 2025 customers){empty_cols}\n")
            f.write(f"================================================================================{empty_cols}\n")
            zero_rev_sorted = zero_rev.sort_values(COL_TOTAL_REV_2024, ascending=False)
            format_for_output(zero_rev_sorted).to_csv(f, index=False, lineterminator='\n')
            f.write(f"{empty_cols}\n")
    
    print(f"  Detail report for {rep_name}: {output_file}")


# =============================================================================
# VALIDATION
# =============================================================================

def validate_reports(df: pd.DataFrame, summary_df: pd.DataFrame) -> bool:
    """
    Validate that totals match between input data and reports.
    Returns True if validation passes, False otherwise.
    """
    print("\n" + "="*80)
    print("VALIDATION")
    print("="*80)
    
    # INPUT TOTAL
    input_total = df[COL_TOTAL_REV_2025].sum()
    print(f"Input total Rev 2025: ${input_total:,.2f}")
    
    # SUMMARY TOTAL (After state)
    summary_total = summary_df['After_Rev_2025'].sum()
    print(f"Summary total Rev 2025 (After): ${summary_total:,.2f}")
    
    # CHECK IF THEY MATCH (within rounding tolerance)
    tolerance = 0.01
    if abs(input_total - summary_total) < tolerance:
        print("✓ Validation PASSED: Totals match")
        return True
    else:
        print("✗ Validation FAILED: Totals do not match")
        print(f"  Difference: ${abs(input_total - summary_total):,.2f}")
        return False


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function"""
    
    # PARSE COMMAND LINE ARGUMENTS
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    else:
        input_file = 'Account_Segmentation_V3_3_Final_SOT_12-21-25.csv'
    
    if len(sys.argv) > 2:
        output_dir = sys.argv[2]
    else:
        output_dir = './reports'
    
    print("="*80)
    print("ACCOUNT SEGMENTATION REPORT GENERATOR V2")
    print("="*80)
    print(f"Input file: {input_file}")
    print(f"Output directory: {output_dir}")
    print()
    
    # CREATE OUTPUT DIRECTORY IF IT DOESN'T EXIST
    os.makedirs(output_dir, exist_ok=True)
    
    # LOAD AND PREPROCESS DATA
    df = load_and_preprocess_data(input_file)
    
    # GENERATE SUMMARY REPORT
    summary_df = generate_summary_report(df, output_dir)
    
    # GENERATE REP DETAIL REPORTS
    print("\nGenerating rep detail reports...")
    unique_final_reps = sorted(df[COL_FINAL_REP].unique())
    
    for rep in unique_final_reps:
        generate_rep_detail_report(df, rep, output_dir)
    
    print(f"\nGenerated {len(unique_final_reps)} rep detail reports")
    
    # VALIDATE TOTALS
    validation_passed = validate_reports(df, summary_df)
    
    # EXIT WITH APPROPRIATE CODE
    if validation_passed:
        print("\n" + "="*80)
        print("REPORT GENERATION COMPLETE")
        print("="*80)
        sys.exit(0)
    else:
        print("\n" + "="*80)
        print("REPORT GENERATION COMPLETED WITH VALIDATION ERRORS")
        print("="*80)
        sys.exit(1)


if __name__ == '__main__':
    main()
