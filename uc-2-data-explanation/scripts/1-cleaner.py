import pandas as pd

# Load the CSV file
file_path = 'C:/Users/USER/PycharmProjects/alfie/uc-2-data-explanation/csv-json/frames.csv'  # Change this to your actual file
df = pd.read_csv(file_path)

# Drop the second-last column
df.drop(df.columns[-2], axis=1, inplace=True)

# Save the modified CSV
output_path = 'C:/Users/USER/PycharmProjects/alfie/uc-2-data-explanation/csv-json/frames-cleaned.csv'  # Change if needed
df.to_csv(output_path, index=False)

print(f"Modified CSV saved to {output_path}")
