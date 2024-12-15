import csv
import argparse

def filter_csv(input_file, output_file, filter_function):
    """
    Filters rows from the input CSV file based on a filter function and writes
    the filtered rows to the output CSV file.

    Args:
        input_file (str): Path to the input CSV file.
        output_file (str): Path to the output CSV file.
        filter_function (function): A function that takes a row (dict) and
                                     returns True if the row should be included.
    """
    with open(input_file, mode='r', newline='', encoding='utf-8') as infile:
        reader = csv.DictReader(infile)
        rows = [row for row in reader if filter_function(row)]

        if rows:
            with open(output_file, mode='w', newline='', encoding='utf-8') as outfile:
                writer = csv.DictWriter(outfile, fieldnames=reader.fieldnames)
                writer.writeheader()
                writer.writerows(rows)
        else:
            print("No rows matched the filter criteria.")

def held_out_filter(row):
    """Example filter that includes rows where column 'age' > 30."""
    return (not 'distribute_nine' in row['filepath']) and (not 'up_center_single_down_center_single' in row['filepath'])
    # return '_val' in row['filepath']

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter a CSV file based on a custom predicate.")
    parser.add_argument("input_file", help="Path to the input CSV file")
    parser.add_argument("output_file", help="Path to the output CSV file")
    
    args = parser.parse_args()

    # Call the filter function with the provided input and output files
    filter_csv(args.input_file, args.output_file, held_out_filter)