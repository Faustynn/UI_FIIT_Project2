import os
import csv
import configparser
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

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


def analyze_points(file_path):
    points = []
    labels = []

    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            x, y, label = float(row[0]), float(row[1]), int(row[2])
            points.append((x, y))
            labels.append(label)
    return points, labels


def optimal_k_selection(points, labels, k_values, cv_folds=5):
    best_k = None
    best_score = 0
    scores = []

    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k, algorithm='kd_tree')
        cv_scores = cross_val_score(knn, points, labels, cv=cv_folds, scoring='accuracy')
        mean_score = np.mean(cv_scores)
        scores.append((k, mean_score))

        if mean_score > best_score:
            best_score = mean_score
            best_k = k

    print(f"Cross-Validation results for k_values: {scores}")
    print(f"Best k: {best_k}, Average accuracy: {best_score:.4f}")
    return best_k, best_score


def main():
    k_values = range(1, 20)

    base_dir = 'EDA'
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.startswith("trainKNN") and file.endswith(".csv"):
                file_path = os.path.join(root, file)

                points, labels = analyze_points(file_path)
                best_k, best_score = optimal_k_selection(points, labels, k_values)
                print(f"Optimal k for file {file}: {best_k}, Average accuracy: {best_score:.4f}\n")


if __name__ == "__main__":
    main()
