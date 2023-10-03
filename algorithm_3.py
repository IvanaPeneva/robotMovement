import random
import pandas as pd
import numpy as np
import os
import statistics
import recurrent_functions
import csv
import matplotlib.pyplot as plt

global filtered_file
global interpolated_file

processed_files = set()

children = set()

point_when_exceeds = set()

file_with_angles = []


def calculate_vector_angle(vector1, vector2):
    dot_product = np.dot(vector1, vector2)
    magnitude1 = np.linalg.norm(vector1)
    magnitude2 = np.linalg.norm(vector2)
    cosine_angle = dot_product / (magnitude1 * magnitude2)
    angle_rad = np.arccos(cosine_angle)
    angle_deg = np.degrees(angle_rad)
    return angle_deg


def make_angles_list(file):
    angles = [file]
    with open(file, 'r') as f:
        mylist = [tuple(map(float, line.strip().split(','))) for line in f]
        angles.append(mylist)

        first_point_vector1 = mylist[0]
        second_point_vector1 = mylist[1]

        x_1 = first_point_vector1[0]
        y_1 = first_point_vector1[1]

        x_2 = second_point_vector1[0]
        y_2 = second_point_vector1[1]

        for i in range(1, len(mylist) - 2):
            first_point_vector2 = mylist[i]
            second_point_vector2 = mylist[i + 1]

            x_3 = first_point_vector2[0]
            y_3 = first_point_vector2[1]

            x_4 = second_point_vector2[0]
            y_4 = second_point_vector2[1]

            vector1 = np.array([x_2 - x_1, y_2 - y_1])
            vector2 = np.array([x_4 - x_3, y_4 - y_3])

            angle_deg = calculate_vector_angle(vector1, vector2)

            angles.append(angle_deg)

    return angles


def calculate_angle(filename, threshold: float):
    if filename in processed_files:
        # print(f"{filename} has already been processed. Skipping.")
        return

    processed_files.add(filename)

    angles = make_angles_list(filename)
    file_with_angles.append(angles)
    mylist = angles[1]
    first_point_vector1 = mylist[0]
    x_1 = first_point_vector1[0]

    only_angles = angles[2:]


    for k in range (len(only_angles)):
        if only_angles[k] > threshold:
            timestep = k + x_1 + 1
            point_when_exceeds.add(timestep)

            half_len = len(mylist) // 2
            first_half = mylist[:half_len]
            second_half = mylist[half_len:]

            base_filename, ext = os.path.splitext(filename)
            first_filename = base_filename + '_first' + ext
            second_filename = base_filename + '_second' + ext

            with open(first_filename, 'w') as h:
                for point in first_half:
                    h.write(','.join(str(x) for x in point) + '\n')

            with open(second_filename, 'w') as g:
                for point in second_half:
                    g.write(','.join(str(x) for x in point) + '\n')
            calculate_angle(first_filename, threshold)
            calculate_angle(second_filename, threshold)
            return

    # kids that never have angle more than threshold
    children.add(filename)


def split_file(filename, threshold, size, split_technique):
    with open(filename, "r") as f:
        lines = f.readlines()

    num_lines = len(lines)
    lines_per_file = num_lines // size
    lines_written = set()

    for i in range(size):
        start_idx = i * lines_per_file
        end_idx = (i + 1) * lines_per_file if i < (size - 1) else num_lines
        part_lines = lines[start_idx:end_idx]
        part_filename = f"{filename}_{i}.csv"

        part_lines = [line for line in part_lines if line not in lines_written]

        with open(part_filename, "w") as f:
            f.writelines(part_lines)

        lines_written.update(part_lines)

        calculate_angle(part_filename, threshold)


    files_no_more_reparation = sorted(find_filenames_with_different_last_parts(children))

    time_steps = []

    for no_separation_files in files_no_more_reparation:
        for files in file_with_angles:
            file_name = files[0]
            mylist = files[1]
            first_point_vector1 = mylist[0]
            x_1 = int(first_point_vector1[0])

            angles=files[2:]
            if file_name == no_separation_files:
                if split_technique == 'median':
                    searched_angle = statistics.median(angles)
                elif split_technique == 'mean' or split_technique == 'first point':
                    searched_angle = statistics.mean(angles)
                elif split_technique == 'max':
                    searched_angle = max(angles)
                elif split_technique == 'random':
                    random_index = random.randint(0, len(angles) - 1)
                    searched_angle = angles[random_index]
                else:
                    raise ValueError("Invalid string value: " + split_technique)

                median_angle_index = find_closest_index(angles, searched_angle)

                actual_timestep = median_angle_index + 1 + x_1

                time_steps.append(actual_timestep)

    data = pd.read_csv(filtered_file)

    x = data.iloc[:, 0].values

    first_value = x[0] * 1.0
    last_value = x[-1] * 1.0

    if split_technique != 'first point':
        if first_value not in time_steps:
            time_steps.append(first_value)
        if last_value not in time_steps:
            time_steps.append(last_value)
        needed_points = sorted(list(set(time_steps)))

    else:
        if first_value not in point_when_exceeds:
            point_when_exceeds.add(first_value)
        if last_value not in point_when_exceeds:
            point_when_exceeds.add(last_value)

        needed_points = sorted(list(set(point_when_exceeds)))
    needed_points = [int(index) for index in needed_points]
    print(f'The number of needed points is {len(needed_points)}')
    recurrent_functions.interpolation(needed_points)

    print(needed_points)
    return needed_points


def find_closest_index(lst, target):
    return min(range(len(lst)), key=lambda i: abs(lst[i] - target))


def find_filenames_with_different_last_parts(filenames):
    parts_dict = {}

    for filename in filenames:
        base_filename, ext = os.path.splitext(filename)
        parts = base_filename.split('_')
        parent_part = "_".join(parts[:-1])

        if parent_part in parts_dict:
            parts_dict[parent_part].add(filename)
        else:
            parts_dict[parent_part] = {filename}

    result = {f for filenames_set in parts_dict.values() if len(filenames_set) > 1 for f in filenames_set}

    result_without_last_part = {f.rsplit('_', 1)[0] for f in result}

    for filename in filenames:
        if filename.rsplit('_', 1)[0] not in result_without_last_part:
            result_without_last_part.add(filename)

    result_with_csv_ext = set()
    for filename in result_without_last_part:
        if not filename.endswith('.csv'):
            result_with_csv_ext.add(filename + '.csv')
        else:
            result_with_csv_ext.add(filename)

    return result_with_csv_ext


def analysis_bar_graphs(input_file, size, split_pieces, split_technique):
    analysis_bar_graph_area(input_file, size, split_pieces, split_technique)
    analysis_bar_graph_distance(input_file, size, split_pieces, split_technique)


def analysis_bar_graph_area(input_file, size, split_pieces, split_technique):
    results = []
    for value in range(1, size, 1):
        decimal_value = value / 100.0
        print(decimal_value)
        split_file(filtered_file, decimal_value, split_pieces, split_technique)
        area = recurrent_functions.calculate_area()
        global processed_files
        processed_files = set()
        global children
        children = set()
        global point_when_exceeds
        point_when_exceeds = set()
        results.append([value, area])
    with open('results.csv', 'w') as file:
        writer = csv.writer(file)
        writer.writerows(results)

    data = pd.read_csv('results.csv', header=None)

    plt.close('all')
    plt.xticks(range(0, size))

    plt.bar(data.iloc[:, 0], data.iloc[:, 1])

    for i in range(len(data.iloc[:, 0])):
        plt.text(x=data.iloc[i, 0], y=data.iloc[i, 1] + 0.5, s=round(data.iloc[i, 1], 2), ha='center')

    plt.xlabel('Number of points')
    plt.ylabel('Area between graphs')
    plt.title(input_file)

    plt.show()


def analysis_bar_graph_distance(input_file, size, split_pieces, split_technique):
    results = []
    for value in range(1, size, 1):
        decimal_value = value / 100.0
        print(decimal_value)
        split_file(filtered_file, decimal_value, split_pieces, split_technique)
        distance = recurrent_functions.calculate_distance()
        global processed_files
        processed_files = set()
        global children
        children = set()
        global point_when_exceeds
        point_when_exceeds = set()
        results.append([value, distance])
    with open('results.csv', 'w') as file:
        writer = csv.writer(file)
        writer.writerows(results)

    data = pd.read_csv('results.csv', header=None)

    plt.close('all')
    plt.xticks(range(0, size))

    plt.bar(data.iloc[:, 0], data.iloc[:, 1])

    for i in range(len(data.iloc[:, 0])):
        plt.text(x=data.iloc[i, 0], y=data.iloc[i, 1] + 0.5, s=round(data.iloc[i, 1], 2), ha='center')

    plt.xlabel('Number of points')
    plt.ylabel('Max Distance between graphs')
    plt.title(input_file)

    plt.show()


def main():
    input_file = 'TCPREAL/bigU001kg.csv'
    angle_threshold = 0.1
    split_pieces = 3
    dimension = 1
    split_technique = 'median'

    recurrent_functions.process_data(input_file, dimension)
    global filtered_file
    filtered_file = recurrent_functions.filtered_file
    recurrent_functions.calculate_length()

    time_steps = split_file(filtered_file, angle_threshold, split_pieces, split_technique)
    print(time_steps)

    recurrent_functions.calculate_area()
    recurrent_functions.calculate_distance()
    recurrent_functions.residual_analysis_methods()
    recurrent_functions.t_testing()

    # decomment to see full analysis
    #size=21
    #analysis_bar_graphs(input_file, size, split_pieces, split_technique)


if __name__ == "__main__":
    main()
