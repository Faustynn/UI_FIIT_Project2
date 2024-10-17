import random
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import configparser

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

# DataSet
class PointsDataset(Dataset):
    def __init__(self, points, labels):
        self.points = torch.tensor(points, dtype=torch.float32)
        self.colors = torch.tensor(labels, dtype=torch.long)

    def __getitem__(self, idx):
        return self.points[idx], self.colors[idx]

    def __len__(self):
        return len(self.points)







def visualize_points(points, labels, ax, title):
    ax.scatter(points[:, 0], points[:, 1], c=labels, alpha=0.7)
    ax.set_xlim([-5000, 5000])
    ax.set_ylim([-5000, 5000])
    ax.grid(True)
    ax.set_title(title)
def visualize_points_s(points, labels, ax, title):
    ax.scatter(points[:, 0], points[:, 1], c=labels, alpha=0.7)
    ax.set_xlim([-500, 500])
    ax.set_ylim([-500, 500])
    ax.grid(True)
    ax.set_title(title)
def visualize_classification_surface(model, ax):
    x_min, x_max = -5000, 5000
    y_min, y_max = -5000, 5000
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 100), np.arange(y_min, y_max, 100))
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    model.eval()
    with torch.no_grad():
        output = model(torch.tensor(grid_points, dtype=torch.float32))
        _, predicted = torch.max(output, 1)

    ax.scatter(grid_points[:, 0], grid_points[:, 1], c=predicted.numpy(), alpha=1, s=5)
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    ax.grid(True)
    ax.set_title('2D Surface')






# load start points
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
# generate random points alg.
def generate_random_points(num_points, points, labels):
    points = points.tolist()
    labels = labels.tolist()
    classes = ['R', 'G', 'B', 'P']
    last_class = None
    generated_points = set(tuple(point) for point in points)

    while len(points) < num_points:
        if last_class is None or random.random() < 0.99:
            if last_class is None or last_class != 'R':
                class_type = 'R'
            else:
                class_type = random.choice([c for c in classes if c != last_class])
        else:
            class_type = random.choice(classes)

        if class_type == 'R':
            x = random.randint(-500, 500)
            y = random.randint(-500, 500)
        elif class_type == 'G':
            x = random.randint(-500, 500)
            y = random.randint(0, 500)
        elif class_type == 'B':
            x = random.randint(-500, 500)
            y = random.randint(0, 500)
        else:
            x = random.randint(-500, 500)
            y = random.randint(-500, 500)

        new_point = (x, y)
        if new_point not in generated_points:
            points.append(new_point)
            labels.append({'R': 0, 'G': 1, 'B': 2, 'P': 3}[class_type])
            generated_points.add(new_point)
            last_class = class_type

    return np.array(points), np.array(labels)




# classify point alg.
def classify(model, point):
    model.eval()
    with torch.no_grad():
        input_tensor = torch.tensor(point, dtype=torch.float32).unsqueeze(0)
        output = model(input_tensor)
        pos, predicted = torch.max(output, 1)
        return predicted.item()




# Main function
def main():
    # Set random seed
    if len(sys.argv) > 1:
        random_seed = int(sys.argv[1])
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
    else:
        random_seed = int(config['generate']['seed'])
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)



    # Initialize start points
    start_points, start_labels = init_start_points()


    # Initialize the model, criterion, and optimizer
    model = ClassifierNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=float(config['neuron-settings']['learning-rate']))

    # Create dataset and data loader
    dataset = PointsDataset(start_points, start_labels)
    data_loader = DataLoader(dataset, batch_size=int(config['neuron-settings']['batch-size']), shuffle=True)

    best_accuracy = 0.0
    patience = int(config['neuron-settings']['early-stopping-patience'])
    epochs_without_improvement = 0
    epoch_count = 0

    # Training model
    while True:
        model.train()
        for batch_points, batch_labels in data_loader:
            optimizer.zero_grad()
            outputs = model(batch_points)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()



        model.eval()
        with torch.no_grad():
            outputs = model(torch.tensor(start_points, dtype=torch.float32))
            _, predicted = torch.max(outputs, 1)
            accuracy = (predicted.numpy() == start_labels).mean() * 100
            print(f'Gen {epoch_count + 1}, Accuracy: {accuracy:.2f}%')
            epoch_count += 1


            if accuracy > best_accuracy:
                best_accuracy = accuracy
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    print("Stop training")
                    break
    # Save the best model
    torch.save(model.state_dict(), 'model/model.pth')



    # Generate and classify new points
    new_points, new_labels = generate_random_points(int(config['generate']['Red']), start_points, start_labels)
    classified_labels = []

    for point in new_points:
        predicted_class = classify(model, point)
        classified_labels.append(predicted_class)

    classified_labels = np.array(classified_labels)


    # Accuracy for new points
    accuracy_new = np.mean(classified_labels == new_labels)
    print(f'Accuracy for generated points: {accuracy_new * 100:.2f}%')


    fig, axs = plt.subplots(2, 2, figsize=(15, 15))
    visualize_points(start_points, start_labels, axs[0, 0], 'Start Points')
    visualize_points(new_points, classified_labels, axs[0, 1], 'Generate Points Full')
    visualize_points_s(new_points, classified_labels, axs[1, 0], 'Generate Points 500x500')
    visualize_classification_surface(model, axs[1, 1])
    plt.show()
    fig.savefig(f'results/plot_{random_seed}.png')

if __name__ == '__main__':
    main()
