1.DECISION TREE ALGORITHM
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn import preprocessing
data = {
    "Day": ["D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9", "D10", "D11", "D12", "D13", "D14"],
    "Outlook": ["Sunny", "Sunny", "Overcast", "Rain", "Rain", "Rain", "Overcast", "Sunny", "Sunny", "Rain", "Sunny", "Overcast", "Overcast", "Rain"],
    "Temperature": ["Hot", "Hot", "Hot", "Mild", "Cool", "Cool", "Cool", "Mild", "Cool", "Mild", "Mild", "Mild", "Hot", "Mild"],
    "Humidity": ["High", "High", "High", "Mild", "Normal", "Normal", "Normal", "High", "Normal", "Normal", "High", "High", "Normal", "High"],
    "Wind": ["Weak", "Strong", "Weak", "Weak", "Weak", "Strong", "Strong", "Weak", "Weak", "Weak", "Strong", "Strong", "Weak", "Strong"],
    "PlayTennis": ["No", "No", "Yes", "Yes", "Yes", "No", "Yes", "No", "Yes", "Yes", "Yes", "Yes", "Yes", "No"]
}
df = pd.DataFrame(data)
df.to_csv('tennis_data.csv', index=False)
label_encoder = preprocessing.LabelEncoder()
df['Outlook'] = label_encoder.fit_transform(df['Outlook'])
df['Temperature'] = label_encoder.fit_transform(df['Temperature'])
df['Humidity'] = label_encoder.fit_transform(df['Humidity'])
df['Wind'] = label_encoder.fit_transform(df['Wind'])
df['PlayTennis'] = label_encoder.fit_transform(df['PlayTennis'])
X = df[['Outlook', 'Temperature', 'Humidity', 'Wind']]
y = df['PlayTennis']
clf = DecisionTreeClassifier()
clf = clf.fit(X, y)
tree_rules = export_text(clf, feature_names=['Outlook', 'Temperature', 'Humidity', 'Wind'])
print(tree_rules)



2.K MEANS CLUSTERING
import pandas as pd
import numpy as np
data = {
    "Individual": [1, 2, 3, 4, 5, 6, 7],
    "Variable 1": [1.0, 1.5, 3.0, 5.0, 3.5, 4.5, 3.5],
    "Variable 2": [1.0, 2.0, 4.0, 7.0, 5.0, 5.0, 4.5]
}
df = pd.DataFrame(data)
df.to_csv('kmeans_data.csv', index=False)
data_points = np.array(list(zip(df["Variable 1"], df["Variable 2"])))
k = 2  # Number of clusters
centroids = data_points[:k]
def euclidean_distance(point1, point2):
    return np.sqrt(sum((point1 - point2) ** 2))
for _ in range(10):  # Run the algorithm for a fixed number of iterations
    clusters = {i: [] for i in range(k)}
    for point in data_points:
        distances = [euclidean_distance(point, centroid) for centroid in centroids]
        cluster_index = distances.index(min(distances))
        clusters[cluster_index].append(point)
    for cluster_index in clusters:
        if clusters[cluster_index]:  # Avoid division by zero
            centroids[cluster_index] = np.mean(clusters[cluster_index], axis=0)
for cluster_index in clusters:
    print(f"Cluster {cluster_index + 1}: {clusters[cluster_index]}")


3LINEAR REGRESSION
x_values = [0, 1, 2, 3, 4]
y_values = [2, 3, 5, 4, 6]
n = len(x_values)
sum_x = sum_y = sum_xy = sum_x_squared = 0
for i in range(n):
    sum_x += x_values[i]
    sum_y += y_values[i]
    sum_xy += x_values[i] * y_values[i]
    sum_x_squared += x_values[i] ** 2
mean_x = sum_x / n
mean_y = sum_y / n
numerator = sum_xy - n * mean_x * mean_y
denominator = sum_x_squared - n * mean_x ** 2
a = numerator / denominator
b = mean_y - a * mean_x
print(f"Linear regression line: y = {a:.2f}x + {b:.2f}")
x_new = 10
y_new = a * x_new + b
print(f"Estimated value of y when x = 10: {y_new:.2f}")
error = 0
for i in range(n):
    y_pred = a * x_values[i] + b
    error += (y_values[i] - y_pred) ** 2
print(f"Total error: {error:.2f}")


4 S ALGORITHM
data = [
    ["Sunny", "Warm", "Normal", "Strong", "Warm", "Same", "Yes"],
    ["Sunny", "Warm", "High", "Strong", "Warm", "Same", "Yes"],
    ["Rainy", "Cold", "High", "Strong", "Warm", "Change", "No"],
    ["Sunny", "Warm", "High", "Strong", "Cool", "Change", "Yes"]
]
hypothesis = ["0", "0", "0", "0", "0", "0"]
for instance in data:
    if instance[-1] == "Yes":  # Consider only positive examples
        for i in range(len(hypothesis)):
            if hypothesis[i] == "0":  
                hypothesis[i] = instance[i]
            elif hypothesis[i] != instance[i]:  
                hypothesis[i] = "?"
print("Maximally Specific Hypothesis:", hypothesis)



5. GMM
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
iris = datasets.load_iris()
X = iris.data[:, :2]  # Using only the first two features for easy visualization
gmm = GaussianMixture(n_components=3, random_state=42)
gmm.fit(X)
labels = gmm.predict(X)
colors = ['red', 'green', 'yellow']
for i in range(3):
    plt.scatter(X[labels == i, 0], X[labels == i, 1], color=colors[i], label=f'Gaussian {i + 1}')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.title('Gaussian Mixture Model on Iris Dataset')
plt.legend()
plt.show()




6 SVM
import numpy as np
import matplotlib.pyplot as plt
X = np.array([1, 5, 1.5, 8, 1, 9, 7, 8.7, 2.3, 5.5, 7.7, 6.1])
Y = np.array([2, 8, 1.8, 8, 0.6, 11, 10, 9.4, 4, 3, 8.8, 7.5])
labels = np.array([-1, -1, -1, 1, -1, 1, 1, 1, -1, -1, 1, 1])
plt.scatter(X[labels == 1], Y[labels == 1], color='b', label='Class 1')
plt.scatter(X[labels == -1], Y[labels == -1], color='r', label='Class -1')
w = np.array([0.5, 0.5])
b = -4
def plot_hyperplane(w, b, color):
    x_min, x_max = 0, 10
    y_min, y_max = 0, 10
    xx = np.linspace(x_min, x_max, 100)
    yy = (-w[0] * xx - b) / w[1]
    plt.plot(xx, yy, color=color)
plot_hyperplane(w, b, 'k')
plot_hyperplane(w, b - 1, 'g--')
plot_hyperplane(w, b + 1, 'g--')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.title('SVM Model with Optimal Hyperplane and Marginal Planes')
plt.show()



7.KNN
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
data = [[158, 58, 'M'], [158, 59, 'M'], [158, 63, 'M'], [160, 59, 'M'], 
        [160, 60, 'M'], [163, 60, 'M'], [163, 61, 'M'], [160, 64, 'L'], 
        [163, 64, 'L'], [165, 61, 'L'], [165, 62, 'L'], [165, 65, 'L'], 
        [168, 62, 'L'], [168, 63, 'L'], [168, 66, 'L'], [170, 63, 'L'], 
        [170, 64, 'L'], [170, 68, 'L']]
data = np.array(data)
heights = data[:, 0].astype(float)
weights = data[:, 1].astype(float)
labels = data[:, 2]
def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))
def k_nearest_neighbors(new_point, k=3):
    distances = []
    for i in range(len(data)):
        distance = euclidean_distance(new_point, np.array([heights[i], weights[i]]))
        distances.append((distance, labels[i]))
    distances.sort(key=lambda x: x[0])
    neighbors = distances[:k]
    output = [neighbor[1] for neighbor in neighbors]
    prediction = Counter(output).most_common(1)[0][0]
    return prediction
new_customer = np.array([161, 61])
predicted_size = k_nearest_neighbors(new_customer, k=3)
print(f"The predicted T-shirt size for the new customer is: {predicted_size}")
plt.scatter(heights[labels == 'M'], weights[labels == 'M'], color='r', label='M')
plt.scatter(heights[labels == 'L'], weights[labels == 'L'], color='b', label='L')
plt.scatter(new_customer[0], new_customer[1], color='g', marker='x', label='New Customer')
plt.xlabel('Height (in cms)')
plt.ylabel('Weight (in kgs)')
plt.legend()
plt.title('K-Nearest Neighbors')
plt.show()