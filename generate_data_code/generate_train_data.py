import os
import random
import sys
import numpy as np
import configparser
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import time

# Load config
config = configparser.ConfigParser()
config.read('../config/config.txt')

# Constants
class_labels = {'B': 0, 'P': 1, 'R': 2, 'G': 3}
random_ranges = {
    'R': ((-5000, 500), (-500, 5000)),
    'G': ((-500, 5000), (-500, 5000)),
    'B': ((-5000, 500), (-5000, 500)),
    'P': ((500, 5000), (-5000, 500))
}

# Initialize starting points from the config file
def init_start_points(config):
    try:
        start_points = {label: eval(config['start-bods'][label]) for label in class_labels.keys()}
    except KeyError as e:
        print(f"Error in config file: missing key {e}")
        sys.exit(1)

    points = []
    labels = []
    for label, coords in start_points.items():
        for x, y in coords:
            points.append([x, y])
            labels.append(class_labels[label])
    return np.array(points), np.array(labels)

# KNN method using scikit-learn's KNeighborsClassifier
def KNN_method(points, colors, new_point, k):
    # Init the classifier
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(points, colors)
    # Predict the color for the new point
    predicted_color = knn.predict([new_point])[0]
    return predicted_color


# Generate random points
def generate_random_points(num_points, k, start_points, start_colors, class_classes):
    points = np.vstack([start_points, np.empty((0, 2), dtype=int)])  # Initialize with start points
    colors = np.append(start_colors, np.empty(0, dtype=int))  # Initialize with start colors
    generated_points = set(map(tuple, start_points.tolist()))

    class_classes_list = list(class_classes)  # Convert dict_keys to list

    for class_type in class_classes_list:  # Use class_classes_list
        while num_points[class_classes_list.index(class_type)] > 0:
            x_range, y_range = random_ranges[class_type]
            new_point = (random.randint(*x_range), random.randint(*y_range))

            if new_point not in generated_points:
                generated_points.add(new_point)
                predicted_color = KNN_method(points, colors, np.array(new_point), k)

                predicted_label = {v: k for k, v in class_labels.items()}[predicted_color]

                if predicted_label not in class_classes_list:
                    predicted_label = class_type

                points = np.vstack([points, new_point])
                colors = np.append(colors, class_labels[predicted_label])
                num_points[class_classes_list.index(class_type)] -= 1
    return points, colors

# Save data to a file
def save_into_file(points, colors, k, random_seed,x):
    if x == 0:
        output_dir = f"data/knn{k}"
        os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir = f"EDA"
        os.makedirs(output_dir, exist_ok=True)

    file_path = f"{output_dir}/trainKNN{k}_data{random_seed}.csv"
    with open(file_path, "w") as f:
        for point, color in zip(points, colors):
            f.write(f"{point[0]},{point[1]},{color}\n")

# Plot points
def plot_points(points, labels, title='Generated Points'):
    color_map = {0: 'red', 1: 'green', 2: 'blue', 3: 'purple'}
    plt.figure(figsize=(20, 15))

    # Определяем цвета для точек
    colors = [color_map[label] for label in labels]

    # Отрисовка точек
    plt.scatter(points[:, 0], points[:, 1], c=colors, alpha=0.8)

    # Определяем границы для каждого класса и рисуем линии
    random_ranges = {
        'R': ((-5000, 500), (-500, 5000)),
        'G': ((-500, 5000), (-500, 5000)),
        'B': ((-5000, 500), (-5000, 500)),
        'P': ((500, 5000), (-5000, 500))
    }

    # red
    plt.plot([-5000, 500], [-500, -500], color='#8b0000', linewidth=4)  # Горизонтальная черная линия по оси X
    plt.plot([500, 500], [-500, 5000], color='#8b0000', linewidth=4)  # Вертикальная черная линия по оси Y

    # blue
    plt.plot([-5000, 500], [500, 500], color='#00008b', linewidth=4)  # Горизонтальная черная линия по оси X
    plt.plot([500, 500], [500, -5000], color='#00008b', linewidth=4)  # Вертикальная черная линия по оси Y

    # green
    plt.plot([-500, 5000], [-500, -500], color='#008b2d', linewidth=4)  # Горизонтальная черная линия по оси X
    plt.plot([-500, -500], [-500, 5000], color='#008b2d', linewidth=4)  # Вертикальная черная линия по оси Y

    # purple
    plt.plot([-500, 5000], [500, 500], color='#ef00fb', linewidth=4)  # Горизонтальная черная линия по оси X
    plt.plot([-500, -500], [500, -5000], color='#ef00fb', linewidth=4)  # Вертикальная черная линия по оси Y

    # Заголовок и параметры графика
    plt.title(title)
    plt.xlim(-5000, 5000)
    plt.ylim(-5000, 5000)
    plt.grid(True)
    plt.axhline(0, color='white', linewidth=1)  # Отрисовка оси X
    plt.axvline(0, color='white', linewidth=1)  # Отрисовка оси Y

    plt.show()

# Main function
def main(x):
    start_time = time.time()  # Start the timer

    random_seed = int(sys.argv[1]) if len(sys.argv) > 1 else int(config['generate']['seed'])
    random.seed(random_seed)
    np.random.seed(random_seed)

    if x == 0:
        k = int(config['generate']['knn'])
        start_points, start_colors = init_start_points(config)
    else:
        k = x
        start_points, start_colors = init_start_points(config)

    num_points = [int(config['generate'][color]) for color in class_labels.keys()]
    points, colors = generate_random_points(num_points, k, start_points, start_colors, class_labels.keys())

    save_into_file(points, colors, k, random_seed,x)
    plot_points(points, colors, title='Generated Points')

    elapsed_time = time.time() - start_time
    print(f"Execution time: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    main(0)
