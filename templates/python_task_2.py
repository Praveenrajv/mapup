import pandas as pd


def calculate_distance_matrix(df)->pd.DataFrame():
    """
    Calculate a distance matrix based on the dataframe, df.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Distance matrix
    """
   distance_matrix = distance_matrix.fillna(0)
    distance_matrix.values[[range(distance_matrix.shape[0])]*2] = 0

    # Ensure the matrix is symmetric by adding the transposed matrix
    distance_matrix += distance_matrix.T

    # Perform matrix addition for cumulative distances along known routes
    for col in distance_matrix.columns:
        for idx in distance_matrix.index:
            if pd.notna(distance_matrix.at[idx, col]):
                for k in distance_matrix.columns:
                    if pd.notna(distance_matrix.at[col, k]):
                        distance_matrix.at[idx, k] = distance_matrix.at[idx, col] + distance_matrix.at[col, k]

    return distance_matrix

# Example usage
# Assuming 'dataset-3.csv' is the name of your CSV file
dataset_path = 'dataset-3.csv'  # Replace with the actual path to your CSV file
df = pd.read_csv(dataset_path)
result_matrix = calculate_distance_matrix(df)

# Display the resulting distance matrix
print(result_matrix)

    return df


def unroll_distance_matrix(df)->pd.DataFrame():
    """
    Unroll a distance matrix to a DataFrame in the style of the initial dataset.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Unrolled DataFrame containing columns 'id_start', 'id_end', and 'distance'.
    """
    
for id_start in unique_pairs:
        for id_end in unique_pairs:
            if id_start != id_end:
                # Check if the distance is available in the distance_matrix
                if id_start in distance_matrix.index and id_end in distance_matrix.columns:
                    distance = distance_matrix.at[id_start, id_end]
                    unrolled_df = unrolled_df.append({'id_start': id_start, 'id_end': id_end, 'distance': distance},
                                                     ignore_index=True)

    return unrolled_df

# Example usage
# Assuming 'result_matrix' is the DataFrame from Question 1
result_unrolled = unroll_distance_matrix(result_matrix)

# Display the resulting unrolled DataFrame
print(result_unrolled)
    return df


def find_ids_within_ten_percentage_threshold(df, reference_id)->pd.DataFrame():
    """
    Find all IDs whose average distance lies within 10% of the average distance of the reference ID.

    Args:
        df (pandas.DataFrame)
        reference_id (int)

    Returns:
        pandas.DataFrame: DataFrame with IDs whose average distance is within the specified percentage threshold
                          of the reference ID's average distance.
    """
 reference_rows = unrolled_df[unrolled_df['id_start'] == reference_value]

    # Calculate the average distance for the reference value
    average_distance = reference_rows['distance'].mean()

    # Calculate the threshold range (10% of the average distance)
    threshold_range = 0.1 * average_distance

    # Filter rows where the distance is within the threshold range
    filtered_rows = reference_rows[(reference_rows['distance'] >= (average_distance - threshold_range)) &
                                   (reference_rows['distance'] <= (average_distance + threshold_range))]

    # Extract unique values from id_start column
    result_list = sorted(filtered_rows['id_start'].unique())

    return result_list

# Example usage
# Assuming 'result_unrolled' is the DataFrame from the previous step
reference_id = 123  # Replace with the desired reference value
result_ids_within_threshold = find_ids_within_ten_percentage_threshold(result_unrolled, reference_id)

# Display the resulting list of values within the threshold
print(result_ids_within_threshold)

    return df


def calculate_toll_rate(df)->pd.DataFrame():
    """
    Calculate toll rates for each vehicle type based on the unrolled DataFrame.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    
    # Define rate coefficients for each vehicle type
    rate_coefficients = {'moto': 0.8, 'car': 1.2, 'rv': 1.5, 'bus': 2.2, 'truck': 3.6}

    # Multiply distance by rate coefficients for each vehicle type
    for vehicle_type, rate_coefficient in rate_coefficients.items():
        unrolled_df[vehicle_type] = unrolled_df['distance'] * rate_coefficient

    return unrolled_df

# Example usage
# Assuming 'result_unrolled' is the DataFrame from the previous step
result_with_toll_rate = calculate_toll_rate(result_unrolled)

# Display the resulting DataFrame with toll rates
print(result_with_toll_rate)

    return df


def calculate_time_based_toll_rates(df)->pd.DataFrame():
    """
    Calculate time-based toll rates for different time intervals within a day.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
  time_ranges = [(time(0, 0, 0), time(10, 0, 0)), (time(10, 0, 0), time(18, 0, 0)), (time(18, 0, 0), time(23, 59, 59))]
    weekday_discount_factors = [0.8, 1.2, 0.8]
    weekend_discount_factor = 0.7

    # Create a dictionary to map days of the week
    days_of_week = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'}

    # Initialize empty lists to store new columns
    start_day_list, start_time_list, end_day_list, end_time_list = [], [], [], []

    # Iterate over each row in the DataFrame
    for _, row in unrolled_df.iterrows():
        # Extract start and end times
        start_time = row['start_time']
        end_time = row['end_time']

        # Determine discount factor based on time range
        if start_time.weekday() < 5:  # Weekdays
            discount_factors = weekday_discount_factors
        else:  # Weekends
            discount_factors = [weekend_discount_factor] * len(time_ranges)

        # Iterate over time ranges and apply discount factors
        for time_range, discount_factor in zip(time_ranges, discount_factors):
            if time_range[0] <= start_time <= time_range[1] and time_range[0] <= end_time <= time_range[1]:
                row['moto':'truck'] *= discount_factor
                break  # Exit loop once the applicable time range is found

        # Append values to the lists
        start_day_list.append(days_of_week[start_time.weekday()])
        start_time_list.append(start_time)
        end_day_list.append(days_of_week[end_time.weekday()])
        end_time_list.append(end_time)

    # Add new columns to the DataFrame
    unrolled_df['start_day'] = start_day_list
    unrolled_df['start_time'] = start_time_list
    unrolled_df['end_day'] = end_day_list
    unrolled_df['end_time'] = end_time_list

    return unrolled_df

# Example usage
# Assuming 'result_unrolled' is the DataFrame from the previous step
result_with_time_based_toll_rates = calculate_time_based_toll_rates(result_unrolled)

# Display the resulting DataFrame with time-based toll rates
print(result_with_time_based_toll_rates)

    return df
