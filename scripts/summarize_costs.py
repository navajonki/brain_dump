#!/usr/bin/env python3
"""
Summarize Replicate API costs from the CSV log file.
"""

import os
import csv
import sys
from datetime import datetime
from pathlib import Path
from collections import defaultdict

# Default path to the CSV file
DEFAULT_CSV_PATH = "logs/replicate_costs.csv"

def format_cost(cost):
    """Format cost with appropriate units (dollars or cents)."""
    if cost < 0.01:
        return f"{cost * 100:.4f}¢"  # Show in cents with 4 decimal places
    else:
        return f"${cost:.4f}"  # Show in dollars with 4 decimal places

def summarize_costs(csv_path=DEFAULT_CSV_PATH):
    """
    Read the CSV file and summarize the costs.
    
    Args:
        csv_path: Path to the CSV file
    """
    if not os.path.exists(csv_path):
        print(f"Error: Cost log file not found at {csv_path}")
        return
    
    # Initialize counters
    total_cost = 0.0
    total_input_cost = 0.0
    total_output_cost = 0.0
    total_input_tokens = 0
    total_output_tokens = 0
    total_calls = 0
    session_count = 0
    
    # Track costs by date and command
    costs_by_date = defaultdict(float)
    costs_by_command = defaultdict(float)
    
    # Read the CSV file
    with open(csv_path, 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        
        for row in reader:
            session_count += 1
            
            # Extract data
            input_tokens = int(row['Input Tokens'])
            output_tokens = int(row['Output Tokens'])
            input_cost = float(row['Input Cost ($)'])
            output_cost = float(row['Output Cost ($)'])
            total_session_cost = float(row['Total Cost ($)'])
            call_count = int(row['Call Count'])
            
            # Add to totals
            total_input_tokens += input_tokens
            total_output_tokens += output_tokens
            total_input_cost += input_cost
            total_output_cost += output_cost
            total_cost += total_session_cost
            total_calls += call_count
            
            # Track by date
            date_str = row['Start Time'].split()[0]
            costs_by_date[date_str] += total_session_cost
            
            # Track by command
            command = row['Command'].split()[0]  # Just the script name
            costs_by_command[command] += total_session_cost
    
    # Print summary
    print("\n" + "=" * 60)
    print(f"REPLICATE API COST SUMMARY")
    print("=" * 60)
    
    print(f"\nTotal Sessions: {session_count}")
    print(f"Total API Calls: {total_calls}")
    print(f"Total Input Tokens: {total_input_tokens:,} ({format_cost(total_input_cost)})")
    print(f"Total Output Tokens: {total_output_tokens:,} ({format_cost(total_output_cost)})")
    print(f"Total Cost: {format_cost(total_cost)}")
    
    # Print costs by date
    print("\nCosts by Date:")
    for date, cost in sorted(costs_by_date.items()):
        print(f"  {date}: {format_cost(cost)}")
    
    # Print costs by command
    print("\nCosts by Command:")
    for command, cost in sorted(costs_by_command.items(), key=lambda x: x[1], reverse=True):
        print(f"  {command}: {format_cost(cost)}")
    
    print("\n" + "=" * 60)
    print(f"Cost data from: {csv_path}")
    print("=" * 60)

def main():
    """Main function to parse arguments and run the summary."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Summarize Replicate API costs")
    parser.add_argument("--csv", default=DEFAULT_CSV_PATH, help="Path to the CSV cost log file")
    
    args = parser.parse_args()
    summarize_costs(args.csv)

if __name__ == "__main__":
    main() 