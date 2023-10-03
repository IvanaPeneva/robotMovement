import csv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from scipy import stats
import math

output_file = 'filtered_output.csv'
global filtered_file
global needed_points_interpolation
global interpolated_file
global interpolated_continuous_file


def process_data(input_file, dimension):
    calculated_dimension = dimension + 1

    with open(input_file, 'r') as file:
        reader = csv.reader(file, delimiter='\t')
        rows = [row for row in reader]

    new_rows = []
    for i, row in enumerate(rows, start=1):
        identifier = row[0]
        array_values = row[1].strip('[]').split(', ')
        new_row = [str(i)] + [identifier] + array_values
        new_rows.append(new_row)

    with open(output_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(new_rows)

    data = pd.read_csv(output_file)
    data = data[data.iloc[:, 1] != 1]
    data_filtered = data.iloc[:, [0, calculated_dimension]]
    data_filtered.to_csv(output_file, index=False, header=None)

    global filtered_file
    filtered_file = output_file

    # Save the filtered data without header
    data_filtered.to_csv(output_file, index=False, header=None)

    # Read the content of the new output file
    with open(output_file, 'r') as file:
        lines = file.readlines()

    # Write all lines except the first one
    with open(output_file, 'w') as file:
        file.writelines(lines[1:])


def interpolation(needed_points):
    data = pd.read_csv(filtered_file)

    x = data.iloc[:, 0].values
    y = data.iloc[:, 1].values

    f = interp1d(x, y, kind='linear')
    new_x = np.array(needed_points)
    new_y = f(new_x)

    interpolated_data_output_file = 'interpolated_data.csv'
    interpolated_data_df = pd.DataFrame({'x': new_x, 'y': new_y})
    interpolated_data_df.to_csv(interpolated_data_output_file, index=False, header=False)
    global interpolated_file
    interpolated_file = interpolated_data_output_file


def make_data_continuous():
    data = pd.read_csv(interpolated_file, header=None, names=['x', 'y'])
    data = data.sort_values(by='x')
    first_x = data['x'].min() - 1
    first_y = data['y'].iloc[0]
    data = pd.concat([pd.DataFrame({'x': [first_x], 'y': [first_y]}), data])

    interpolator = interp1d(data['x'], data['y'], kind='linear', fill_value='extrapolate')

    continuous_x = range(int(data['x'].min()), int(data['x'].max()) + 1)
    continuous_y = interpolator(continuous_x)
    continuous_data = pd.DataFrame({'x': continuous_x, 'y': continuous_y})
    continuous = 'interpolated_data_continuous.csv'
    continuous_data.to_csv(continuous, index=False, header=False)
    print(len(continuous_data))

    global interpolated_continuous_file
    interpolated_continuous_file = continuous


def calculate_area():
    data1 = pd.read_csv(filtered_file, header=None)
    data2 = pd.read_csv(interpolated_file, header=None)

    make_data_continuous()

    continuous_data = pd.read_csv(interpolated_continuous_file, header=None)
    y_values = continuous_data.iloc[:, 1].values
    y_diff = np.abs(data1.iloc[:,
                    1] - y_values)
    # The area between the two graphs is defined as the integral of the absolute
    # difference between their y-values with respect to x.

    area = np.trapz(y_diff, x=data1.iloc[:, 0])
    # x-values are used to determine the width of each trapezoid in our case=1 because each timestep

    fig, ax = plt.subplots()
    ax.plot(data1.iloc[:, 0], data1.iloc[:, 1], label=filtered_file)
    ax.plot(data2.iloc[:, 0], data2.iloc[:, 1], label=interpolated_file)
    ax.plot(data2.iloc[:, 0], data2.iloc[:, 1], 'o', color='red')
    ax.fill_between(data1.iloc[:, 0], data1.iloc[:, 1], y_values, color='green')
    ax.legend()
    plt.title(f'The area is {area}')
    plt.show()
    print(f'The area between the two graphs is {area}')

    return area


def distance_from_point_to_regression_line(slope, intercept, x_segment, y_segment):
    x_intersect = (x_segment + slope * (y_segment - intercept)) / (1 + slope ** 2)
    y_intersect = slope * x_intersect + intercept

    distance = abs(((x_intersect - x_segment) ** 2 + (y_intersect - y_segment) ** 2) ** 0.5)

    return distance, (x_intersect, y_intersect)


def calculate_distance():
    distances = []

    split_points = []
    with open(interpolated_file, 'r') as file:
        for line in file:
            time_step = int(line.split(',')[0])
            split_points.append(time_step)
    print(split_points)

    with open(filtered_file, "r") as filtered:
        lines = filtered.readlines()

    for i in range(len(split_points) - 1):
        start_point = split_points[i]
        end_point = split_points[i + 1]

        with open(f"segment{start_point}_{end_point}.txt", "w") as output:
            for line in lines:
                parts = line.strip().split(',')
                x = int(parts[0])

                if start_point <= x < end_point:
                    output.write(line)
                elif x >= end_point:
                    break

    max_distance = -1
    x_max_distance = None

    x_intersect = None
    y_intersect = None

    x1, y1 = None, None

    with open(interpolated_file, 'r') as interpolated_file_read:
        for line_interpolated in interpolated_file_read:
            parts_interpolated = line_interpolated.strip().split(',')
            x2, y2 = float(parts_interpolated[0]), float(parts_interpolated[1])

            if x1 is not None and y1 is not None:
                slope = (y2 - y1) / (x2 - x1)
                intercept = y1 - slope * x1

            for i in range(len(split_points) - 1):
                start_point = split_points[i]
                end_point = split_points[i + 1]

                if start_point == x1:
                    segment_name = f"segment{start_point}_{end_point}.txt"
                    with open(segment_name, 'r') as segment_file:
                        for line in segment_file:
                            parts = line.strip().split(',')
                            x_segment = int(parts[0])
                            y_segment = float(parts[1])

                            distance, (x_intersect1, y_intersect1) = distance_from_point_to_regression_line(slope,
                                                                                                            intercept,
                                                                                                            x_segment,
                                                                                                            y_segment)
                            distances.append(distance)

                            if distance > max_distance:
                                max_distance = distance
                                x_max_distance = x_segment
                                x_intersect = x_intersect1
                                y_intersect = y_intersect1

            x1, y1 = x2, y2

    filtered_df = pd.read_csv(filtered_file, header=None, names=['x', 'y'])
    corresponding_y = filtered_df.loc[filtered_df['x'] == x_max_distance, 'y'].values[0]

    print(
        f"The biggest distance is: {max_distance} and is between point x: {x_max_distance} "
        f"y:{corresponding_y} and intersects at x: {x_intersect} and y: {y_intersect} ")
    fig, ax = plt.subplots()

    data1 = pd.read_csv(filtered_file, header=None)

    ax.plot(data1.iloc[:, 0], data1.iloc[:, 1], label=filtered_file)

    df1 = pd.read_csv(interpolated_file, header=None)
    ax.plot(df1.iloc[:, 0], df1.iloc[:, 1], label=interpolated_file)

    plt.scatter(x_max_distance, corresponding_y, color='green', marker='o')

    plt.scatter(x_intersect, y_intersect, color='green', marker='x')

    plt.scatter(x_max_distance, corresponding_y, color='green', marker='o',
                label=f'({x_max_distance}, {corresponding_y})')

    plt.scatter(x_intersect, y_intersect, color='green', marker='x',
                label=f'({x_intersect}, {y_intersect})')

    plt.plot([x_max_distance, x_intersect], [corresponding_y, y_intersect], color='green')

    ax.plot(df1.iloc[:, 0], df1.iloc[:, 1], 'o', color='red')
    ax.legend()
    plt.title(f'The largest distance is: {max_distance}')
    plt.show()
    return max_distance


def residuals_plot():
    data1 = pd.read_csv(filtered_file, header=None)
    data2 = pd.read_csv(interpolated_continuous_file, header=None)

    if len(data1) != len(data2):
        raise ValueError("Data1 and Data2 must have the same number of data points.")

    residuals: object = np.array(data2)[:, 1] - np.array(data1)[:, 1]

    plt.scatter(np.array(data1)[:, 0], residuals, label='Residuals', color='blue', alpha=0.7)
    plt.axhline(y=0, color='red', linestyle='--', label='Zero Residual Line')

    plt.xlabel('X Values')
    plt.ylabel('Residuals')
    plt.legend()

    plt.title('Residual Analysis')
    plt.grid(True)
    plt.show()

    return residuals


def mae(residuals: object):
    # MAE measures the average absolute difference between the observed and expected values.
    mae_calculation = np.mean(np.abs(residuals))
    print(f'Mean Absolute Error (MAE): {mae_calculation}')


def mse_and_rmse(residuals: object):
    # MSE measures the average of the squared differences between the observed and expected values.
    # MSE tends to penalize larger errors more severely than smaller errors.
    mse = np.mean(residuals ** 2)
    print(f'Mean Squared Error (MSE): {mse}')
    # RMSE is the square root of MSE and provides a
    # measure of the typical magnitude of errors in the same units as the data.
    rmse = np.sqrt(mse)
    print(f'Root Mean Squared Error (RMSE): {rmse}')


def std_dev_residuals(residuals: object):
    # The standard deviation of residuals measures the spread or variability of residuals around zero.
    std_dev_residuals_calculation = np.std(residuals)
    print(f'Standard Deviation of Residuals: {std_dev_residuals_calculation}')


def r_squared(residuals: object):
    data2 = pd.read_csv(interpolated_continuous_file, header=None)
    y_observed = data2.iloc[:, 1]

    y_mean = y_observed.mean()

    ss_total = ((y_observed - y_mean) ** 2).sum()

    ss_residual = (residuals ** 2).sum()

    r_squared_calculation = 1 - (ss_residual / ss_total)

    print(f'Coefficient of Determination (R-squared): {r_squared_calculation}')


def residual_analysis_methods():
    residuals = residuals_plot()
    mae(residuals)
    mse_and_rmse(residuals)
    std_dev_residuals(residuals)
    r_squared(residuals)


def t_testing():
    # Read the data using pandas
    data1 = pd.read_csv(filtered_file, header=None)
    data2 = pd.read_csv(interpolated_continuous_file, header=None)

    # Extract the relevant columns
    y_expected = data1.iloc[:, 1]
    y_observed = data2.iloc[:, 1]

    # Perform the t-test
    t_statistic, p_value = stats.ttest_ind(y_expected, y_observed)

    alpha = 0.1

    print("T-statistic:", t_statistic)
    print("P-value:", p_value)
    print(f'Alpha: {alpha}')

    if p_value < alpha:
        print("The p-value is less than the chosen significance level (alpha).")
        print("We reject the null hypothesis.")
    else:
        print("The p-value is greater than or equal to the chosen significance level (alpha).")
        print("We fail to reject the null hypothesis.")


def calculate_length():
    total_length = 0
    prev_point = None

    with open(filtered_file, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            x, y = map(float, row)
            if prev_point is not None:
                distance = math.sqrt((x - prev_point[0]) ** 2 + (y - prev_point[1]) ** 2)
                total_length += distance
            prev_point = (x, y)

    print(f'Total length of the curve: {total_length}')