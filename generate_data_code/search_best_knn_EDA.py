import generate_train_data
import matplotlib.pyplot as plt
import csv
import configparser

# Load config
config = configparser.ConfigParser()
config.read('../config/config.txt')

# Constants
class_labels = {0: 'B', 1: 'P', 2: 'R', 3: 'G'}
barriers = {
    'R': {'horizontal': 500, 'vertical': 500},
    'G': {'horizontal': 500, 'vertical': -500},
    'B': {'horizontal': -500, 'vertical': 500},
    'P': {'horizontal': -500, 'vertical': -500},
}
seed = config['generate']['seed']


def analyze_points(iteration):
    points = []
    labels = []

    with open(f'EDA/trainKNN{iteration}_data{seed}.csv', 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            x, y, label = float(row[0]), float(row[1]), int(row[2])
            points.append((x, y))
            labels.append(label)

    # Count points outside barr.
    out_of_bounds = {color: 0 for color in barriers.keys()}
    total_points = len(points)

    for (x, y), label in zip(points, labels):
        color = class_labels[label]
        if color in barriers:
            if (color == 'R' and (x >= barriers['R']['vertical'] or y >= barriers['R']['horizontal'])) or \
               (color == 'G' and (x <= barriers['G']['vertical'] or y >= barriers['G']['horizontal'])) or \
               (color == 'B' and (x >= barriers['B']['vertical'] or y <= barriers['B']['horizontal'])) or \
               (color == 'P' and (x <= barriers['P']['vertical'] or y <= barriers['P']['horizontal'])):
                out_of_bounds[color] += 1

    print(f"\nKNN {iteration}:")
    for color, count in out_of_bounds.items():
        print(f"{color}: {count} points out")

        # Count points for the spec. color
        total_color_points = labels.count([k for k, v in class_labels.items() if v == color][0])
        if total_color_points > 0:
            out_of_bounds_percentage = (count / total_color_points) * 100
            print(f"{color} %: {out_of_bounds_percentage:.2f}%")
        else:
            print(f"No points for color {color} to analyze")

    total_percentage = (sum(out_of_bounds.values()) / total_points) * 100
    print(f"TOTAL %: {total_percentage:.2f}%\n")
    visualize_points(points, labels)


def visualize_points(points, labels):
    color_map = {0: 'blue', 1: 'purple', 2: 'red', 3: 'green'}
    darker_color_map = {0: 'darkblue', 1: 'darkviolet', 2: 'darkred', 3: 'darkgreen'}

    in_bounds_x, in_bounds_y, in_bounds_labels = [], [], []
    out_bounds_x, out_bounds_y, out_bounds_labels = [], [], []

    for (x, y), label in zip(points, labels):
        color = class_labels[label]
        if (color == 'R' and (x >= barriers['R']['vertical'] or y >= barriers['R']['horizontal'])) or \
           (color == 'G' and (x <= barriers['G']['vertical'] or y >= barriers['G']['horizontal'])) or \
           (color == 'B' and (x >= barriers['B']['vertical'] or y <= barriers['B']['horizontal'])) or \
           (color == 'P' and (x <= barriers['P']['vertical'] or y <= barriers['P']['horizontal'])):
            out_bounds_x.append(x)
            out_bounds_y.append(y)
            out_bounds_labels.append(label)
        else:
            in_bounds_x.append(x)
            in_bounds_y.append(y)
            in_bounds_labels.append(label)

    plt.figure(figsize=(10, 10))

    in_bounds_colors = [color_map[label] for label in in_bounds_labels]
    out_bounds_colors = [darker_color_map[label] for label in out_bounds_labels]

    plt.scatter(in_bounds_x, in_bounds_y, color=in_bounds_colors, alpha=0.6)
    plt.scatter(out_bounds_x, out_bounds_y, color=out_bounds_colors, alpha=0.6)

    plt.xlim(-5000, 5000)
    plt.ylim(-5000, 5000)
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.legend(handles=[plt.Line2D([0], [0], marker='o', color='w', label='B', markerfacecolor='blue', markersize=10),
                        plt.Line2D([0], [0], marker='o', color='w', label='P', markerfacecolor='purple', markersize=10),
                        plt.Line2D([0], [0], marker='o', color='w', label='R', markerfacecolor='red', markersize=10),
                        plt.Line2D([0], [0], marker='o', color='w', label='G', markerfacecolor='green', markersize=10),
                        plt.Line2D([0], [0], marker='o', color='w', label='B (Out of Bounds)', markerfacecolor='darkblue', markersize=10),
                        plt.Line2D([0], [0], marker='o', color='w', label='P (Out of Bounds)', markerfacecolor='darkviolet', markersize=10),
                        plt.Line2D([0], [0], marker='o', color='w', label='R (Out of Bounds)', markerfacecolor='darkred', markersize=10),
                        plt.Line2D([0], [0], marker='o', color='w', label='G (Out of Bounds)', markerfacecolor='darkgreen', markersize=10)],
               title='Class Labels')
    plt.grid()
    plt.axhline(0, color='gray', lw=0.5, ls='--')
    plt.axvline(0, color='gray', lw=0.5, ls='--')
    plt.show()


def main():
    for i in range(5, 16):
         generate_train_data.main(i)
       # analyze_points(i)
    return 0


if __name__ == "__main__":
    main()
