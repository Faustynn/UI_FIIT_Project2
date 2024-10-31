import os
import random
import sys
import numpy as np
import configparser
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import time

config = configparser.ConfigParser()
if len(sys.argv) > 1:
    config.read('../config/config.txt')
else:
    config.read('config/config.txt')

class_labels = {'B': 0, 'P': 1, 'R': 2, 'G': 3}
random_ranges = {
    'R': ((-5000, 500), (-500, 5000)),
    'G': ((-500, 5000), (-500, 5000)),
    'B': ((-5000, 500), (-5000, 500)),
    'P': ((500, 5000), (-5000, 500))
}
barriers = {
    'R': {'horizontal': -500, 'vertical': 500},
    'G': {'horizontal': -500, 'vertical': -500},
    'B': {'horizontal': 500, 'vertical': 500},
    'P': {'horizontal': 500, 'vertical': -500},
}

def init_start_points(config):
    start_points = {label: eval(config['start-bods'][label]) for label in class_labels.keys()}
    points = []
    labels = []

    for label, coords in start_points.items():
        for x, y in coords:
            points.append([x, y])
            labels.append(class_labels[label])
    return np.array(points), np.array(labels)

def KNN_method(points, colors, new_point, k):
    knn = KNeighborsClassifier(n_neighbors=k, algorithm='kd_tree')
    knn.fit(points, colors)
    predicted_color = knn.predict([new_point])[0]
    return predicted_color

def generate_random_points(num_points, k, start_points, start_colors, class_classes):
    points = np.vstack([start_points, np.empty((0, 2), dtype=int)])
    colors = np.append(start_colors, np.empty(0, dtype=int))

    generated_points = set(map(tuple, start_points.tolist()))
    new_points = []
    new_colors = []

    class_classes_list = list(class_classes)
    for class_type in class_classes_list:
        attempts = 0
        while num_points[class_classes_list.index(class_type)] > 0 and attempts < 10000:
            new_point = (random.randint(-5000,5000), random.randint(-5000,5000))

            if new_point not in generated_points:
                generated_points.add(new_point)
                new_points.append(new_point)

                predicted_color = KNN_method(points, colors, np.array(new_point), k)
                new_colors.append(predicted_color)
                num_points[class_classes_list.index(class_type)] -= 1
            attempts += 1

        if attempts >= 10000:
            print(f"Max atempts for {class_type}.")

    points = np.vstack([points, np.array(new_points)])
    colors = np.append(colors, new_colors)
    return points, colors

def save_into_file(points, colors, k, random_seed):
    if len(sys.argv) > 2:
        output_dir = f"../data"
    elif len(sys.argv) > 1:
        output_dir = f"EDA_KNN"
    else:
        output_dir = f"generate_data_code/EDA_KNN"
    os.makedirs(output_dir, exist_ok=True)

    file_path = f"{output_dir}/trainKNN{k}_data{random_seed}.csv"
    with open(file_path, "w") as f:
        for point, color in zip(points, colors):
            f.write(f"{point[0]},{point[1]},{color}\n")

def plot_points(points, labels):
    color_map = {0: 'red', 1: 'green', 2: 'blue', 3: 'purple'}

    plt.figure(figsize=(20, 15))
    colors = [color_map[label] for label in labels]
    plt.scatter(points[:, 0], points[:, 1], c=colors, alpha=0.8)

    plt.plot([-5000, 500], [-500, -500], color='#8b0000', linewidth=4)  # X
    plt.plot([500, 500], [-500, 5000], color='#8b0000', linewidth=4)  # Y

    plt.plot([-5000, 500], [500, 500], color='#00008b', linewidth=4)
    plt.plot([500, 500], [500, -5000], color='#00008b', linewidth=4)

    plt.plot([-500, 5000], [-500, -500], color='#008b2d', linewidth=4)
    plt.plot([-500, -500], [-500, 5000], color='#008b2d', linewidth=4)

    plt.plot([-500, 5000], [500, 500], color='#ef00fb', linewidth=4)
    plt.plot([-500, -500], [500, -5000], color='#ef00fb', linewidth=4)

    plt.xlim(-5000, 5000)
    plt.ylim(-5000, 5000)
    plt.grid(True)
    plt.axhline(0, color='white', linewidth=1)  # X
    plt.axvline(0, color='white', linewidth=1)  # Y
    plt.show()

def main():
    start_time = time.time()

    if len(sys.argv) > 2:
        random_seed = int(sys.argv[1])
    else:
        random_seed = int(config['generate']['seed'])
    random.seed(random_seed)
    np.random.seed(random_seed)

    if len(sys.argv) > 2:
        k = int(sys.argv[2])
    elif len(sys.argv) > 1:
        k = int(sys.argv[1])
    else:
        k = int(config['generate']['knn'])

    start_points, start_colors = init_start_points(config)

    num_points = [int(config['generate'][color]) for color in class_labels.keys()]
    points, colors = generate_random_points(num_points, k, start_points, start_colors, class_labels.keys())

    save_into_file(points, colors, k, random_seed)
  #  plot_points(points, colors)

    elapsed_time = time.time() - start_time
    print(f"Time: {elapsed_time:.2f} sec.")

if __name__ == "__main__":
    main()
