import os
import random
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import configparser
import time
from sklearn.model_selection import train_test_split

# Load config
config = configparser.ConfigParser()
config.read('config/config.txt')


# NN definition
class ClassifierNN(nn.Module):
    def __init__(self):
        super(ClassifierNN, self).__init__()
        self.fc1 = nn.Linear(2, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 4)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        out = self.leaky_relu(self.fc1(x))
        out = self.dropout(out)
        out = self.leaky_relu(self.fc2(out))
        out = self.dropout(out)
        return self.fc3(out)

# Dataset
class PointsDataset(Dataset):
    def __init__(self, points, labels):
        self.points = torch.tensor(points, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __getitem__(self, idx):
        return self.points[idx], self.labels[idx]

    def __len__(self):
        return len(self.points)
# Load data from directory
def load_data_from_directory(directory_path):
    points = []
    labels = []
    for filename in os.listdir(directory_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(directory_path, filename)
            with open(file_path, 'r') as f:
                for line in f:
                    x, y, label = line.strip().split(',')
                    points.append([float(x), float(y)])
                    labels.append(int(label))
    return np.array(points), np.array(labels)


# Initialize start points
def init_start_points():
    start_points = {
        'R': eval(config['start-bods']['R']),
        'G': eval(config['start-bods']['G']),
        'B': eval(config['start-bods']['B']),
        'P': eval(config['start-bods']['P']),
    }
    points = []
    labels = []
    for label, coords in start_points.items():
        for x, y in coords:
            points.append([x, y])
            labels.append({'R': 0, 'G': 1, 'B': 2, 'P': 3}[label])
    return np.array(points), np.array(labels)
# Generate random points using the model
def generate_random_points(num_points, points, labels, model):
    points = points.tolist()
    labels = labels.tolist()
    generated_points = set(tuple(point) for point in points)

    zones = {
        0: lambda: (random.randint(-5000, 500), random.randint(-5000, 500)),  # R: X < +500, Y < +500
        1: lambda: (random.randint(-500, 5000), random.randint(-5000, 500)),  # G: X > -500, Y < +500
        2: lambda: (random.randint(-5000, 500), random.randint(-500, 5000)),  # B: X < +500, Y > -500
        3: lambda: (random.randint(-500, 5000), random.randint(-500, 5000))   # P: X > -500, Y > -500
    }

    sum_num_points = num_points + len(points)
    while len(points) < sum_num_points:
        new_point = None
        label = random.choice([0, 1, 2, 3])  # Случайный выбор класса (R, G, B, P)

        while True:
            new_point = zones[label]()
            if new_point not in generated_points:
                generated_points.add(new_point)
                break

        points.append(new_point)

        model.eval()
        with torch.no_grad():
            input_tensor = torch.tensor(new_point, dtype=torch.float32).unsqueeze(0)
            output = model(input_tensor)
            _, predicted = torch.max(output, 1)

        labels.append(predicted.item())

    return np.array(points), np.array(labels)


# Train the model
def train_model(model, train_loader, optimizer, criterion, early_stopping_patience, test_loader,knn):
    best_accuracy = 0.0
    epochs_without_improvement = 0
    epoch = 0
    while True:
        model.train()
        running_loss = 0.0
        for batch_points, batch_labels in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_points)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        epoch += 1
        print(f'Epoch {epoch}, Loss: {running_loss / len(train_loader)}')

        # Evaluate on the test set after each epoch
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_points, batch_labels in test_loader:
                outputs = model(batch_points)
                _, predicted = torch.max(outputs, 1)
                total += batch_labels.size(0)
                correct += (predicted == batch_labels).sum().item()

        accuracy = correct / total
        print(f'Test Accuracy: {accuracy * 100:.2f}%')

        # Check if accuracy improved
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            epochs_without_improvement = 0
            # Save the model if it improves
            torch.save(model.state_dict(), f'model/model_knn{knn}.pth')
            print(f'Model saved to model_knn{knn}.pth')
        else:
            epochs_without_improvement += 1

        # If no improvement for 'early_stopping_patience' epochs, stop training
        if epochs_without_improvement >= early_stopping_patience:
            print(f'Early stopping triggered. No improvement for {early_stopping_patience} epochs.')
            break


# Main function
def main():
    # Start timer
    start_time = time.time()

    # Set random seed
    random_seed = int(config['generate']['seed']) if len(sys.argv) == 1 else int(sys.argv[1])
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    # Initialize start points
    start_points, start_labels = init_start_points()

    # Initialize the model
    model = ClassifierNN()

    # Check if model file exists
    knn = int(config['generate']['knn'])
    model_path = f'model/model_knn{knn}.pth'

    if os.path.exists(model_path):
        print(f'Loading existing model from {model_path}')
        model.load_state_dict(torch.load(model_path))
    else:
        # Load all data for training
        directory_path = f'data/knn{knn}'
        points, labels = load_data_from_directory(directory_path)

        # Combine start points with loaded points
        all_points = np.vstack([start_points, points])
        all_labels = np.append(start_labels, labels)

        # Split data into training and test sets (80% train, 20% test)
        train_points, test_points, train_labels, test_labels = train_test_split(all_points, all_labels, test_size=0.2,
                                                                                random_state=random_seed)

        # Create datasets and data loaders
        train_dataset = PointsDataset(train_points, train_labels)
        train_loader = DataLoader(train_dataset, batch_size=int(config['neuron-settings']['batch-size']), shuffle=True)

        test_dataset = PointsDataset(test_points, test_labels)
        test_loader = DataLoader(test_dataset, batch_size=int(config['neuron-settings']['batch-size']), shuffle=False)

        # Initialize criterion and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=float(config['neuron-settings']['learning-rate']))

        # Train the model with early stopping
        early_stopping_patience = int(config['neuron-settings']['early-stopping-patience'])
        train_model(model, train_loader, optimizer, criterion, early_stopping_patience, test_loader,knn)

    # Generate and classify new points
    num_points_to_generate = int(config['generate']['num_generate'])
    new_points, new_labels = generate_random_points(num_points_to_generate, start_points, start_labels, model)

    # Visualize the results
    fig, axs = plt.subplots(1, 2, figsize=(15, 7))
    axs[0].scatter(start_points[:, 0], start_points[:, 1], c=start_labels, alpha=0.7)
    axs[0].set_title('Start Points')
    axs[1].scatter(new_points[:, 0], new_points[:, 1], c=new_labels, alpha=0.7)
    axs[1].set_title('Generated Points')
    plt.show()
    fig.savefig(f'results/plot_{random_seed}.png')

    # Calculate time
    print(f'Process time: {time.time() - start_time:.2f} sec.')
if __name__ == '__main__':
    main()
