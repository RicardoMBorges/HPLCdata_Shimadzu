"""
process_chromatography_data.py

This script is designed to process chromatography data from text files.
It contains functions to extract relevant data from files and combine them into a single dataset.

Functions:
- extract_data(file_path): Reads a text file, extracts data starting from a line that begins with "R.Time (min)", 
  and stores it in a list.
- main(): Iterates through all text files in a given directory, applies the extract_data function to each, 
  and saves the results in a new file with a modified name.
- Additional code: After the main function, the script contains code to combine data from multiple files into a 
  single DataFrame, filter this DataFrame based on specific retention times, and save the final combined data 
  to a CSV file.

Usage:
- The script is intended to be run as a standalone Python script or imported as a module in other scripts 
  or Jupyter Notebooks.
- Modify the input and output directory paths as needed before running the script.
"""


def extract_data(file_path):
    data = []
    start_extraction = False

    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith("R.Time (min)"):
                start_extraction = True
                continue
            if start_extraction:
                columns = line.strip().split()
                if len(columns) == 2:
                    # Replace commas with dots in each column
                    columns = [col.replace(',', '.') for col in columns]
                    data.append(columns)

    return data

def main():
    input_folder = directory_path
    output_folder = directory_path

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for file_name in os.listdir(input_folder):
        if file_name.endswith(".txt"):
            file_path = os.path.join(input_folder, file_name)
            data = extract_data(file_path)

            # Save the data into a new file
            output_file_path = os.path.join(output_folder, f"{os.path.splitext(file_name)[0]}_table.csv")
            with open(output_file_path, 'w') as output_file:
                for row in data:
                    output_file.write('\t'.join(row) + '\n')

if __name__ == "__main__":
    main()


# Get a list of all files matching the pattern *_table.txt
file_list = glob.glob('*_table.csv')

# Initialize an empty DataFrame to store the combined data
combined_df = pd.DataFrame()

# Loop through each file and read its data into a DataFrame
for file in file_list:
    # Extracting the file name (excluding the extension) to use as a column header
    column_name = file.split('_table.csv')[0]
    
    # Assuming the files are tab-delimited, you can adjust the delimiter accordingly
    df = pd.read_csv(file, delimiter='\t')
    
    # Add the data to the combined DataFrame with the file name as the column header
    combined_df[column_name] = df.iloc[:, 1]  # Assuming you want the second column, adjust as needed

# Assuming the files are tab-delimited, you can adjust the delimiter accordingly
df1 = pd.read_csv(file, delimiter='\t', header=None)
axis = df1.iloc[:, 0]

# Concatenate 'axis' DataFrame with 'combined_df'
combined_df2 = pd.concat([axis, combined_df], axis=1)
combined_df2.rename(columns={0:"RT(min)"},inplace=True)    

######## Select parts to remove - remove ends.
retention_time_start = 5 #(min)
retention_time_end = 30  #(min)

# Step 1: Find the index of the row closest to retention_time_start
start_index = (combined_df2["RT(min)"] - retention_time_start).abs().idxmin()

# Step 2: Find the index of the row closest to retention_time_end
end_index = (combined_df2["RT(min)"] - retention_time_end).abs().idxmin()

# Step 3: Slice the DataFrame to keep only the rows within the selected range
selected_range_df = combined_df2.loc[start_index:end_index]

# Step 4: Delete the rows before retention_time_start and after retention_time_end in the original DataFrame
combined_df2 = combined_df2.loc[start_index:end_index].copy()

# Save the combined DataFrame to a CSV file
combined_df2.to_csv('combined_data.csv', sep=";",index=False)