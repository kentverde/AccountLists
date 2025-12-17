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
    df = pd.read_csv(input_file)

    original_count = len(df)
    print(f"Loaded {original_count:,} accounts")

    # Add Final_Rep_Name column if not present
    if COL_FINAL_REP not in df.columns:
        df[COL_FINAL_REP] = ''
        print("Added Final_Rep_Name column")

    # Handle blank/null Assigned_Rep_Name - replace with 'House'
    df[COL_ASSIGNED_REP] = df[COL_ASSIGNED_REP].fillna('House')
    df.loc[df[COL_ASSIGNED_REP].str.strip() == '', COL_ASSIGNED_REP] = 'House'

    # Handle blank/null Re-Assigned_Rep_Name - replace with 'House'
    df[COL_REASSIGNED_REP] = df[COL_REASSIGNED_REP].fillna('House')
    df.loc[df[COL_REASSIGNED_REP].str.strip() == '', COL_REASSIGNED_REP] = 'House'

    # Parse Customer_Since_Date and extract year
    df['Customer_Since_Year'] = pd.to_datetime(
        df[COL_CUSTOMER_SINCE], format='%m/%d/%Y', errors='coerce'
    ).dt.year

    # Add is_new_2025 flag
    df['Is_New_2025'] = df['Customer_Since_Year'] == PROTECTED_YEAR

    # Add group label based on Revenue_Tier
    df['Revenue_Group'] = df[COL_REVENUE_TIER].map(REVENUE_TIER_TO_GROUP).fillna('Unknown')

    # Add potential label
    df['Potential_Label'] = df[COL_POTENTIAL_TIER].map(POTENTIAL_TIER_TO_LABEL).fillna('Unknown')

    # Add combined segment label (e.g., "Group 1 High Potential")
    df['Segment_Label'] = df['Revenue_Group'] + ' ' + df['Potential_Label']

    # Fill NaN revenue values with 0
    df[COL_TOTAL_REV_2024] = df[COL_TOTAL_REV_2024].fillna(0)
    df[COL_TOTAL_REV_2025] = df[COL_TOTAL_REV_2025].fillna(0)
    df[COL_ORDERS_2024] = df[COL_ORDERS_2024].fillna(0)
    df[COL_ORDERS_2025] = df[COL_ORDERS_2025].fillna(0)

    # Validate totals
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
    df = df.copy()

    # Determine aligned rep based on Rep_Type
    df['Aligned_Rep_Name'] = np.where(
        df[COL_REP_TYPE] == 'Inside Sales',
        df[COL_REASSIGNED_REP],
        df[COL_ASSIGNED_REP]
    )

    # Track if account changed reps due to alignment
    df['Alignment_Changed'] = df[COL_ASSIGNED_REP] != df['Aligned_Rep_Name']

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
    df = df.copy()

    # Floor criteria: Group 3 + DESIGN + not new in 2025
    df['Floor_Removed'] = (
        (df[COL_REVENUE_TIER] == FLOOR_REVENUE_TIER) &
        (df[COL_SEGMENT] == FLOOR_SEGMENT) &
        (~df['Is_New_2025'])
    )

    floor_removed_count = df['Floor_Removed'].sum()
    floor_removed_revenue = df.loc[df['Floor_Removed'], COL_TOTAL_REV_2025].sum()

    print(f"Accounts removed by floor: {floor_removed_count:,}")
    print(f"Revenue removed by floor: ${floor_removed_revenue:,.2f}")

    return df


# =============================================================================
# SUMMARY REPORT GENERATION
# =============================================================================

def generate_summary_report(df: pd.DataFrame, output_file: str) -> pd.DataFrame:
    """
    Generate summary report by rep matching the Power BI format.
    """
    print("\nGenerating summary report...")

    # Get unique reps from original assignment
    original_reps = df.groupby(COL_ASSIGNED_REP).agg({
        COL_REP_TYPE: 'first',
        COL_TOTAL_REV_2024: 'sum',
        COL_TOTAL_REV_2025: 'sum',
        COL_ACCOUNT_ID: 'count'
    }).reset_index()
    original_reps.columns = [
        'Assigned_Rep_Name', 'Rep_Type', '2024 Revenue', '2025 Revenue', '2025 Count of Accounts'
    ]

    # Calculate alignment changes per original rep
    # Negative = accounts leaving, Positive = accounts arriving
    accounts_leaving = df[df['Alignment_Changed']].groupby(COL_ASSIGNED_REP).size()
    accounts_arriving = df[df['Alignment_Changed']].groupby('Aligned_Rep_Name').size()

    # Build alignment change column
    original_reps['Accounts_Leaving'] = original_reps['Assigned_Rep_Name'].map(accounts_leaving).fillna(0)
    original_reps['Accounts_Arriving'] = original_reps['Assigned_Rep_Name'].map(accounts_arriving).fillna(0)
    original_reps['Account Changes due to Alignment'] = (
        original_reps['Accounts_Arriving'] - original_reps['Accounts_Leaving']
    ).astype(int)

    # After alignment stats (based on Aligned_Rep_Name)
    after_alignment = df.groupby('Aligned_Rep_Name').agg({
        COL_TOTAL_REV_2025: 'sum',
        COL_ACCOUNT_ID: 'count'
    }).reset_index()
    after_alignment.columns = ['Rep_Name', '2025 Revenue (after Re-Assignment)', 'Count of Accounts after Alignment']

    original_reps = original_reps.merge(
        after_alignment,
        left_on='Assigned_Rep_Name',
        right_on='Rep_Name',
        how='left'
    )

    # Handle reps that lost all accounts (no longer in after_alignment)
    original_reps['Count of Accounts after Alignment'] = original_reps['Count of Accounts after Alignment'].fillna(0).astype(int)
    original_reps['2025 Revenue (after Re-Assignment)'] = original_reps['2025 Revenue (after Re-Assignment)'].fillna(0)

    # Revenue change due to alignment
    original_reps['Revenue Change (after Re-Assignment)'] = (
        original_reps['2025 Revenue (after Re-Assignment)'] - original_reps['2025 Revenue']
    )

    # Floor statistics - All Group 3 accounts (after alignment)
    all_group3 = df[df[COL_REVENUE_TIER] == FLOOR_REVENUE_TIER].groupby('Aligned_Rep_Name').size()
    original_reps['The Floor (All Group 3 Accounts)'] = original_reps['Assigned_Rep_Name'].map(all_group3).fillna(0).astype(int)

    # Floor statistics - Design only, non-new accounts (the ones actually removed)
    floor_removed = df[df['Floor_Removed']].groupby('Aligned_Rep_Name').agg({
        COL_ACCOUNT_ID: 'count',
        COL_TOTAL_REV_2025: 'sum'
    }).reset_index()
    floor_removed.columns = ['Rep_Name', 'The Floor (Design Only, Non New Accounts)', '2025 Revenue for The Floor (Design Only, Non New)']

    original_reps = original_reps.merge(
        floor_removed,
        left_on='Assigned_Rep_Name',
        right_on='Rep_Name',
        how='left',
        suffixes=('', '_floor')
    )

    original_reps['The Floor (Design Only, Non New Accounts)'] = original_reps['The Floor (Design Only, Non New Accounts)'].fillna(0).astype(int)
    original_reps['2025 Revenue for The Floor (Design Only, Non New)'] = original_reps['2025 Revenue for The Floor (Design Only, Non New)'].fillna(0)

    # Final stats after floor removal
    final_stats = df[~df['Floor_Removed']].groupby('Aligned_Rep_Name').agg({
        COL_ACCOUNT_ID: 'count',
        COL_TOTAL_REV_2025: 'sum'
    }).reset_index()
    final_stats.columns = ['Rep_Name', '2025 Final Count of Accounts (Floor Removed)', '2025 Final Revenue All Accounts (Floor Removed)']

    original_reps = original_reps.merge(
        final_stats,
        left_on='Assigned_Rep_Name',
        right_on='Rep_Name',
        how='left',
        suffixes=('', '_final')
    )

    original_reps['2025 Final Count of Accounts (Floor Removed)'] = original_reps['2025 Final Count of Accounts (Floor Removed)'].fillna(0).astype(int)
    original_reps['2025 Final Revenue All Accounts (Floor Removed)'] = original_reps['2025 Final Revenue All Accounts (Floor Removed)'].fillna(0)

    # Overall differences
    original_reps['Overall Count of Account Difference'] = (
        original_reps['2025 Final Count of Accounts (Floor Removed)'] - original_reps['2025 Count of Accounts']
    ).astype(int)

    original_reps['Revenue Diff Between Start and Final 2025'] = (
        original_reps['2025 Final Revenue All Accounts (Floor Removed)'] - original_reps['2025 Revenue']
    )

    # Count of 2025 Zero Rev New Accounts (Re-Assigned)
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

    # Save to CSV
    summary_df.to_csv(output_file, index=False)
    print(f"Summary report saved to: {output_file}")

    return summary_df


# =============================================================================
# REP DETAIL REPORT GENERATION
# =============================================================================

def generate_rep_detail_report(df: pd.DataFrame, rep_name: str, output_file: str):
    """
    Generate detailed report for a single rep.
    Shows accounts they are keeping and losing, grouped by revenue tier.
    """
    # Get accounts aligned to this rep (after territory alignment)
    rep_accounts = df[df['Aligned_Rep_Name'] == rep_name].copy()

    if len(rep_accounts) == 0:
        return

    # Calculate summary metrics
    total_accounts = len(rep_accounts)
    total_rev_2025 = rep_accounts[COL_TOTAL_REV_2025].sum()
    total_rev_2024 = rep_accounts[COL_TOTAL_REV_2024].sum()

    accounts_removed = rep_accounts['Floor_Removed'].sum()
    revenue_removed = rep_accounts.loc[rep_accounts['Floor_Removed'], COL_TOTAL_REV_2025].sum()

    accounts_keeping = total_accounts - accounts_removed
    revenue_keeping = total_rev_2025 - revenue_removed

    zero_rev_accounts = (rep_accounts[COL_TOTAL_REV_2025] == 0).sum()
    zero_rev_new_2025 = ((rep_accounts[COL_TOTAL_REV_2025] == 0) & (rep_accounts['Is_New_2025'])).sum()

    # Account detail columns
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
        COL_GROWTH_CLASS,
        COL_COMPOSITE_SCORE,
        'Floor_Removed'
    ]

    with open(output_file, 'w') as f:
        # Write header section
        f.write(f"Rep Detail Report: {rep_name}\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("\n")
        f.write("=" * 80 + "\n")
        f.write("SUMMARY METRICS\n")
        f.write("=" * 80 + "\n")
        f.write(f"Total Accounts (after alignment): {total_accounts:,}\n")
        f.write(f"Total 2024 Revenue: ${total_rev_2024:,.2f}\n")
        f.write(f"Total 2025 Revenue: ${total_rev_2025:,.2f}\n")
        f.write("\n")
        f.write(f"Accounts Removed (Floor): {accounts_removed:,}\n")
        f.write(f"Revenue Removed (Floor): ${revenue_removed:,.2f}\n")
        f.write("\n")
        f.write(f"Final Accounts (Keeping): {accounts_keeping:,}\n")
        f.write(f"Final Revenue (Keeping): ${revenue_keeping:,.2f}\n")
        f.write("\n")
        f.write(f"Zero Revenue Accounts: {zero_rev_accounts:,}\n")
        f.write(f"Zero Revenue New 2025 Accounts: {zero_rev_new_2025:,}\n")
        f.write("\n")

        # Breakdown by segment
        f.write("=" * 80 + "\n")
        f.write("BREAKDOWN BY SEGMENT\n")
        f.write("=" * 80 + "\n")

        segment_summary = rep_accounts.groupby('Segment_Label').agg({
            COL_ACCOUNT_ID: 'count',
            COL_TOTAL_REV_2025: 'sum',
            'Floor_Removed': 'sum'
        }).reset_index()
        segment_summary.columns = ['Segment', 'Accounts', 'Revenue 2025', 'Removed']
        segment_summary = segment_summary.sort_values('Revenue 2025', ascending=False)

        for _, row in segment_summary.iterrows():
            f.write(f"{row['Segment']}: {row['Accounts']:,} accounts, ${row['Revenue 2025']:,.2f} revenue, {int(row['Removed']):,} removed\n")
        f.write("\n")

        # Group 1 accounts (keeping)
        group1 = rep_accounts[rep_accounts[COL_REVENUE_TIER] == 'HIGH_REVENUE'].copy()
        if len(group1) > 0:
            f.write("=" * 80 + "\n")
            f.write(f"GROUP 1 (HIGH REVENUE) - KEEPING: {len(group1):,} accounts, ${group1[COL_TOTAL_REV_2025].sum():,.2f}\n")
            f.write("=" * 80 + "\n")
            group1_sorted = group1.sort_values(COL_TOTAL_REV_2025, ascending=False)
            group1_sorted[detail_columns].to_csv(f, index=False)
            f.write("\n")

        # Group 2 accounts (keeping)
        group2 = rep_accounts[rep_accounts[COL_REVENUE_TIER] == 'MID_REVENUE'].copy()
        if len(group2) > 0:
            f.write("=" * 80 + "\n")
            f.write(f"GROUP 2 (MID REVENUE) - KEEPING: {len(group2):,} accounts, ${group2[COL_TOTAL_REV_2025].sum():,.2f}\n")
            f.write("=" * 80 + "\n")
            group2_sorted = group2.sort_values(COL_TOTAL_REV_2025, ascending=False)
            group2_sorted[detail_columns].to_csv(f, index=False)
            f.write("\n")

        # Group 3 accounts - KEEPING (non-DESIGN, or new 2025, protected from floor)
        group3_keeping = rep_accounts[
            (rep_accounts[COL_REVENUE_TIER] == 'LOW_REVENUE') &
            (~rep_accounts['Floor_Removed'])
        ].copy()
        if len(group3_keeping) > 0:
            f.write("=" * 80 + "\n")
            f.write(f"GROUP 3 (LOW REVENUE) - KEEPING: {len(group3_keeping):,} accounts, ${group3_keeping[COL_TOTAL_REV_2025].sum():,.2f}\n")
            f.write("(Non-DESIGN accounts or New 2025 customers - protected from floor)\n")
            f.write("=" * 80 + "\n")
            group3_keeping_sorted = group3_keeping.sort_values(COL_TOTAL_REV_2025, ascending=False)
            group3_keeping_sorted[detail_columns].to_csv(f, index=False)
            f.write("\n")

        # Group 3 accounts - LOSING (DESIGN + non-2025 = floor removed)
        group3_losing = rep_accounts[rep_accounts['Floor_Removed']].copy()
        if len(group3_losing) > 0:
            f.write("=" * 80 + "\n")
            f.write(f"GROUP 3 (LOW REVENUE) - LOSING (FLOOR REMOVED): {len(group3_losing):,} accounts, ${group3_losing[COL_TOTAL_REV_2025].sum():,.2f}\n")
            f.write("(DESIGN segment + Not new in 2025)\n")
            f.write("=" * 80 + "\n")
            group3_losing_sorted = group3_losing.sort_values(COL_TOTAL_REV_2025, ascending=False)
            group3_losing_sorted[detail_columns].to_csv(f, index=False)
            f.write("\n")

        # Zero Revenue accounts (separate visibility)
        zero_rev = rep_accounts[rep_accounts[COL_TOTAL_REV_2025] == 0].copy()
        if len(zero_rev) > 0:
            zero_rev_removed = zero_rev['Floor_Removed'].sum()
            zero_rev_keeping = len(zero_rev) - zero_rev_removed
            f.write("=" * 80 + "\n")
            f.write(f"ZERO REVENUE ACCOUNTS: {len(zero_rev):,} total ({zero_rev_keeping:,} keeping, {zero_rev_removed:,} removed)\n")
            f.write("=" * 80 + "\n")
            zero_rev_sorted = zero_rev.sort_values(['Floor_Removed', 'Is_New_2025'], ascending=[True, False])
            zero_rev_sorted[detail_columns].to_csv(f, index=False)
            f.write("\n")


def generate_all_rep_detail_reports(df: pd.DataFrame, output_dir: str) -> dict:
    """
    Generate detail reports for all reps.
    Returns a dictionary with rep name -> total revenue for validation.
    """
    print("\nGenerating rep detail reports...")

    # Create output directory for rep reports
    rep_reports_dir = os.path.join(output_dir, 'rep_details')
    os.makedirs(rep_reports_dir, exist_ok=True)

    # Get unique aligned reps
    reps = df['Aligned_Rep_Name'].unique()

    rep_totals = {}

    for rep in reps:
        # Create safe filename
        safe_name = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in rep)
        safe_name = safe_name.strip().replace(' ', '_')
        output_file = os.path.join(rep_reports_dir, f"{safe_name}_detail.csv")

        generate_rep_detail_report(df, rep, output_file)

        # Track total for validation
        rep_accounts = df[df['Aligned_Rep_Name'] == rep]
        rep_totals[rep] = rep_accounts[COL_TOTAL_REV_2025].sum()

    print(f"Generated {len(reps)} rep detail reports in: {rep_reports_dir}")

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

    # Input total
    input_total = df[COL_TOTAL_REV_2025].sum()
    print(f"Input Total Rev 2025: ${input_total:,.2f}")

    # Summary report total (original 2025 revenue)
    summary_total = summary_df['2025 Revenue'].sum()
    print(f"Summary Report Total (2025 Revenue): ${summary_total:,.2f}")

    # Rep details total
    rep_details_total = sum(rep_totals.values())
    print(f"Rep Details Total: ${rep_details_total:,.2f}")

    # Check for discrepancies
    errors = []

    tolerance = 0.01  # Allow 1 cent tolerance for floating point

    if abs(input_total - summary_total) > tolerance:
        errors.append(f"MISMATCH: Input ({input_total:,.2f}) != Summary ({summary_total:,.2f}), diff: {input_total - summary_total:,.2f}")

    if abs(input_total - rep_details_total) > tolerance:
        errors.append(f"MISMATCH: Input ({input_total:,.2f}) != Rep Details ({rep_details_total:,.2f}), diff: {input_total - rep_details_total:,.2f}")

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
    # Parse command line arguments
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    else:
        input_file = 'Account_Segmentation_V3_3_Final.csv'

    if len(sys.argv) > 2:
        output_dir = sys.argv[2]
    else:
        output_dir = './reports'

    # Validate input file exists
    if not os.path.exists(input_file):
        print(f"ERROR: Input file not found: {input_file}")
        sys.exit(1)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 80)
    print("ACCOUNT SEGMENTATION REPORT GENERATOR")
    print("=" * 80)
    print(f"Input file: {input_file}")
    print(f"Output directory: {output_dir}")
    print()

    # Step 1: Load and preprocess data
    df = load_and_preprocess_data(input_file)

    # Step 2: Apply territory alignment
    df = apply_territory_alignment(df)

    # Step 3: Apply floor logic
    df = apply_floor_logic(df)

    # Step 4: Generate summary report
    summary_file = os.path.join(output_dir, 'Rep_Level_Impacts_2025_Summary.csv')
    summary_df = generate_summary_report(df, summary_file)

    # Step 5: Generate rep detail reports
    rep_totals = generate_all_rep_detail_reports(df, output_dir)

    # Step 6: Validate totals
    validation_passed = validate_totals(df, summary_df, rep_totals)

    print("\n" + "=" * 80)
    print("COMPLETE")
    print("=" * 80)

    if not validation_passed:
        sys.exit(1)


if __name__ == '__main__':
    main()
