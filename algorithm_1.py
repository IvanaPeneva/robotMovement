import pandas as pd
import numpy as np
import recurrent_functions
import csv
import matplotlib.pyplot as plt

global filtered_file
global interpolated_file


def find_points(points: int):
    data = pd.read_csv(filtered_file)
    x = data.iloc[:, 0].values
    new_x = np.linspace(x.min(), x.max(), num=points)
    rounded_x = np.round(new_x).astype(int)

    return rounded_x


def analysis_bar_graphs(input_file, size):
    analysis_bar_graph_area(input_file, size)
    analysis_bar_graph_distance(input_file, size)


def analysis_bar_graph_area(input_file, size):
    results = []
    for i in range(2, size):
        needed_points = find_points(i)
        recurrent_functions.interpolation(needed_points)
        area = recurrent_functions.calculate_area()
        results.append([i, area])
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


def analysis_bar_graph_distance(input_file, size):
    results = []
    for i in range(2, size):
        needed_points = find_points(i)
        recurrent_functions.interpolation(needed_points)
        distance = recurrent_functions.calculate_distance()
        results.append([i, distance])
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
    plt.ylabel('Distance between graphs')
    plt.title(input_file)

    plt.show()


def main():
    input_file = 'TCPREAL/fineU.csv'
    points = 8
    dimension = 1
    recurrent_functions.process_data(input_file, dimension)
    global filtered_file
    filtered_file = recurrent_functions.filtered_file
    needed_points = find_points(points)
    recurrent_functions.interpolation(needed_points)
    recurrent_functions.calculate_area()
    recurrent_functions.calculate_distance()
    recurrent_functions.residual_analysis_methods()
    recurrent_functions.t_testing()

    # decomment to see full analysis
    # size=16
    # analysis_bar_graphs(input_file,size)


if __name__ == "__main__":
    main()
