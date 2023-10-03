import pandas as pd
import numpy as np
from scipy.signal import find_peaks_cwt
from scipy.signal import find_peaks
import recurrent_functions
import csv
import matplotlib.pyplot as plt

global filtered_file
global interpolated_file


def finding_valleys_and_peaks(threshold, cwt):
    data = pd.read_csv(filtered_file)

    x = data.iloc[:, 0].values
    y = data.iloc[:, 1].values

    first_value = x[0] * 1.0
    last_value = x[-1] * 1.0

    if cwt:
        peaks = find_peaks_cwt(y, np.arange(1, threshold))
        valleys = find_peaks_cwt(-y, np.arange(1, threshold))
    else:
        peaks, _ = find_peaks(y)
        valleys, _ = find_peaks(-y)

    needed_points = []

    for p in peaks:
        needed_points.append(p + first_value)

    for v in valleys:
        needed_points.append(v + first_value)

    if first_value not in needed_points:
        needed_points.append(first_value)
    if last_value not in needed_points:
        needed_points.append(last_value)

    needed_points = sorted(list(set(needed_points)))
    needed_points = [int(index) for index in needed_points]
    return needed_points


def analysis_bar_graphs(input_file, width, cwt):
    analysis_bar_graph_area(input_file, width, cwt)
    analysis_bar_graph_distance(input_file, width, cwt)


def analysis_bar_graph_area(input_file, width, cwt):
    results = []
    for i in range(10, width):
        needed_points = finding_valleys_and_peaks(i, cwt)
        recurrent_functions.interpolation(needed_points)
        area = recurrent_functions.calculate_area()
        results.append([i, area])
    with open('results.csv', 'w') as file:
        writer = csv.writer(file)
        writer.writerows(results)

    data = pd.read_csv('results.csv', header=None)

    plt.close('all')
    plt.xticks(range(0, width))

    plt.bar(data.iloc[:, 0], data.iloc[:, 1])

    for i in range(len(data.iloc[:, 0])):
        plt.text(x=data.iloc[i, 0], y=data.iloc[i, 1] + 0.5, s=round(data.iloc[i, 1], 2), ha='center')

    plt.xlabel('Number of points')
    plt.ylabel('Area between graphs')
    plt.title(input_file)

    plt.show()


def analysis_bar_graph_distance(input_file, width, cwt):
    plt.close('all')
    results = []
    for i in range(10, width):
        needed_points = finding_valleys_and_peaks(i, cwt)
        recurrent_functions.interpolation(needed_points)
        distance = recurrent_functions.calculate_distance()
        results.append([i, distance])
    with open('results.csv', 'w') as file:
        writer = csv.writer(file)
        writer.writerows(results)
    with open('results.csv', 'w') as file:
        writer = csv.writer(file)
        writer.writerows(results)

    data = pd.read_csv('results.csv', header=None)

    plt.close('all')
    plt.xticks(range(0, width))

    plt.bar(data.iloc[:, 0], data.iloc[:, 1])

    for i in range(len(data.iloc[:, 0])):
        plt.text(x=data.iloc[i, 0], y=data.iloc[i, 1] + 0.5, s=round(data.iloc[i, 1], 2), ha='center')

    plt.xlabel('Number of points')
    plt.ylabel('Distance between graphs')
    plt.title(input_file)

    plt.show()


def main():
    input_file = 'TCPREAL/slalom001kg.csv'
    width = 60
    dimension = 1
    cwt = True

    recurrent_functions.process_data(input_file, dimension)
    global filtered_file
    filtered_file = recurrent_functions.filtered_file
    recurrent_functions.calculate_length()

    needed_points = finding_valleys_and_peaks(width, cwt)
    recurrent_functions.interpolation(needed_points)

    recurrent_functions.calculate_area()
    recurrent_functions.calculate_distance()
    recurrent_functions.residual_analysis_methods()
    recurrent_functions.t_testing()

    # decomment to see full analysis
    # width = 50
    # analysis_bar_graphs(input_file, width, cwt)


if __name__ == "__main__":
    main()
