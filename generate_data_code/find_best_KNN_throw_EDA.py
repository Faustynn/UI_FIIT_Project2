import numpy as np
import configparser
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import os

config = configparser.ConfigParser()
config.read('config/config.txt')
class_labels = {'B': 0, 'P': 1, 'R': 2, 'G': 3}

def tune_k(points, colors, max_k=20):
    accuracies = []
    k_values = range(1, max_k + 1)

    X_train, X_test, y_train, y_test = train_test_split(points, colors, test_size=0.2, random_state=42)

    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        accuracy = knn.score(X_test, y_test)
        accuracies.append(accuracy)

    best_k = k_values[np.argmax(accuracies)]
    return best_k, accuracies

def plot_accuracies(k_values, accuracies):
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, accuracies, marker='o')
    plt.title('KNN acuracy')
    plt.xlabel('k')
    plt.ylabel('Accuracy')
    plt.xticks(k_values)
    plt.grid()
    plt.show()

def main():
    data_directory = 'EDA_KNN/'
    best_k_overall = None
    best_accuracy_overall = 0

    for filename in os.listdir(data_directory):
        if filename.endswith('.csv'):
            data_file = os.path.join(data_directory, filename)
            data = np.loadtxt(data_file, delimiter=',')

            points = data[:, :2]
            colors = data[:, 2]

            best_k, accuracies = tune_k(points, colors)
            avg_accuracy = np.max(accuracies)

            if avg_accuracy > best_accuracy_overall:
                best_accuracy_overall = avg_accuracy
                best_k_overall = best_k

    with open('EDA_KNN/best_knn.txt', 'w') as f:
        f.write(f'BEST k: {best_k_overall}\n')
        f.write(f'Midle accuracy: {best_accuracy_overall:.4f}\n')

    print(f'Best k: {best_k_overall}, Midle accuracy: {best_accuracy_overall:.4f}')

if __name__ == "__main__":
    main()
