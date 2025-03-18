import pandas as pd
import base64
import os

# Define input and output file paths
input_csv = "C:/Users/USER/PycharmProjects/alfie/data-processing/csv-files/frames.csv"  # Change this to the actual input CSV file path
output_csv = "C:/Users/USER/PycharmProjects/alfie/data-processing/csv-files/frames-decoded.csv"
output_folder = "C:/Users/USER/PycharmProjects/alfie/data-processing/csv-files/"

# Create a folder to store decoded images if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Read the CSV file
df = pd.read_csv(input_csv)

# Function to decode base64 and save image
def decode_and_save(base64_string, frame_id):
    image_path = os.path.join(output_folder, f"frame_{frame_id}.png")
    try:
        image_data = base64.b64decode(base64_string)
        with open(image_path, "wb") as img_file:
            img_file.write(image_data)
        return image_path
    except Exception as e:
        print(f"Error decoding frame {frame_id}: {e}")
        return "Error"

# Process each row
df["frame_data"] = df.apply(lambda row: decode_and_save(row["frame_data"], row["id"]), axis=1)

# Save the modified CSV
df.to_csv(output_csv, index=False)

print(f"Processed CSV saved as {output_csv}")