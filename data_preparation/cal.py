import csv


def analyze_csv_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()  # Read the entire file content
            character_count = len(content)  # Count characters

        # Count rows
        with open(file_path, 'r', encoding='utf-8') as file:
            reader = csv.reader(file)
            row_count = sum(1 for row in reader)  # Count rows

        print(f"{file_path} contains {character_count} characters and {row_count} rows.")
        return character_count, row_count
    except Exception as e:
        print(f"An error occurred: {e}")


# Analyze CSV files
analyze_csv_file("SuicideWatch_submissions_cleaned.csv")
analyze_csv_file("SuicideWatch_comments_cleaned.csv")
analyze_csv_file("depression_submissions_cleaned.csv")
analyze_csv_file("depression_comments_cleaned.csv")
