import pandas as pd

# Define the path to your input CSV file
input_csv = 'dataset.csv'

# Define the path for the new output CSV file
output_csv = 'background_labels.csv'

# Define the columns that represent autistic traits.
# These columns are expected to contain 1s and 0s.
autistic_traits_columns = [
    'Absence or Avoidance of Eye Contact',
    'Aggressive Behavior',
    'Hyper- or Hyporeactivity to Sensory Input',
    'Non-Responsiveness to Verbal Interaction',
    'Non-Typical Language',
    'Object Lining-Up',
    'Self-Hitting or Self-Injurious Behavior',
    'Self-Spinning or Spinning Objects',
    'Upper Limb Stereotypies'
]



# --- THE RULE ---
# This is a simple threshold-based rule.
# You can change the threshold value to fit your criteria.
# The rule is: 'is_autistic' will be 1 if the number of traits present is >= this threshold.
threshold = 3 

# --- Script Logic ---
def add_autism_labels(input_file, output_file, traits, threshold):
    """
    Reads a CSV, applies a rule to add an 'is_autistic' column, and saves the new file.
    
    Args:
        input_file (str): The path to the original CSV file.
        output_file (str): The path to save the new CSV file.
        traits (list): A list of column names representing autistic traits.
        threshold (int): The minimum number of traits to be labeled as autistic.
    """
    try:
        # Load the CSV file into a pandas DataFrame
        df = pd.read_csv(input_file)
        
        # Check if all the required trait columns exist in the DataFrame
        if not all(col in df.columns for col in traits):
            print("Error: One or more autistic trait columns are missing from the CSV.")
            print("Please ensure the CSV contains all of the following columns:")
            print(traits)
            return

        # Add the 'is_autistic' column based on the rule.
        # We sum the values (1s) across the trait columns for each row.
        # If the sum is greater than or equal to the threshold, we assign a 1.
        df['is_autistic'] = (df[traits].sum(axis=1) >= threshold).astype(int)
        
        # Save the new DataFrame to a CSV file.
        # index=False prevents pandas from writing the row index as a new column.
        df.to_csv(output_file, index=False)
        
        print(f"Successfully added the 'is_autistic' column based on the threshold of {threshold}.")
        print(f"The new CSV file has been saved to '{output_csv}'.")
        print("\n--- First 5 rows of the new DataFrame: ---")
        print(df.head())
        
    except FileNotFoundError:
        print(f"Error: The file '{input_file}' was not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# --- Execute the function ---
if __name__ == "__main__":
    add_autism_labels(input_csv, output_csv, autistic_traits_columns, threshold)