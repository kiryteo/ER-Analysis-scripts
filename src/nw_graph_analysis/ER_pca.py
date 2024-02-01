import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.datasets import fetch_openml
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import glob

# Assuming you have a list of image paths, you can use a library like PIL to load the images
# Make sure to install PIL using: pip install pillow
from PIL import Image

def pca_image():
    # Load your 128x128 images
    image_paths = ["path/to/image1.jpg", "path/to/image2.jpg", ...]  # Replace with your image paths

    image_paths = glob.glob('/localhome/asa420/MIAL/data/classification_data/PCA/atl/*')

    images = []

    for path in image_paths:
        img = Image.open(path).convert('L')  # Convert to grayscale
        img = np.array(img).astype(float)
        images.append(img)

    # Stack images into a single matrix
    images_matrix = np.stack(images)

    # Flatten the images
    images_flat = images_matrix.reshape((images_matrix.shape[0], -1))


    import numpy as np
    from sklearn.decomposition import PCA

    # Assuming you have already loaded and flattened your images into images_flat

    # Choose a range of components to evaluate
    num_components_range = np.arange(1, 26, 1)  # Adjust the step size as needed

    explained_variances = []

    for num_components in num_components_range:
        pca = PCA(n_components=num_components)
        pca.fit(images_flat)
        explained_variances.append(np.sum(pca.explained_variance_ratio_))

    # Set a threshold for cumulative explained variance
    threshold_variance = 0.95

    # Find the index where the cumulative explained variance crosses the threshold
    num_components_threshold = np.argmax(np.array(explained_variances) >= threshold_variance) + 1

    # Use the selected number of components in your PCA model
    num_components = num_components_range[num_components_threshold - 1]

    print(f"Selected number of components: {num_components}")


    # # Plot the cumulative explained variance
    # plt.plot(num_components_range, explained_variances, marker='o')
    # plt.xlabel('Number of Components')
    # plt.ylabel('Cumulative Explained Variance')
    # plt.title('Cumulative Explained Variance vs. Number of Components')
    # plt.grid(True)
    # plt.show()


    # exit()



    # Choose the number of principal components
    # num_components = 20

    # Perform PCA
    pca = PCA(n_components=num_components)
    images_pca = pca.fit_transform(images_flat)

    # Project the data back to the original space
    images_reconstructed = pca.inverse_transform(images_pca)

    # Reshape the reconstructed images
    images_reconstructed = images_reconstructed.reshape(images_matrix.shape)

    # Display original and reconstructed images
    fig, axes = plt.subplots(2, 5, figsize=(10, 4))
    for i in range(5):
        axes[0, i].imshow(images[i], cmap='gray')
        axes[1, i].imshow(images_reconstructed[i], cmap='gray')
        axes[0, i].axis('off')
        axes[1, i].axis('off')

    axes[0, 0].set_title('Original')
    axes[1, 0].set_title('Reconstructed')

    plt.show()


import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from PIL import Image
import os
import matplotlib.pyplot as plt

def kmeans_clustering():

    # Specify the path to the folder containing your 128x128 images
    image_folder_path = "/localhome/asa420/MIAL/data/classification_data/PCA"

    # Load images and convert them to grayscale
    image_files = os.listdir(image_folder_path)
    images = []

    for file in image_files:
        image_path = os.path.join(image_folder_path, file)
        img = Image.open(image_path).convert('L')  # Convert to grayscale
        img = np.array(img).astype(float)
        images.append(img)

    # Stack images into a single matrix
    images_matrix = np.stack(images)

    # Flatten the images
    images_flat = images_matrix.reshape((images_matrix.shape[0], -1))

    # Standardize the data (optional, but can be helpful for k-means)
    scaler = StandardScaler()
    images_flat_standardized = scaler.fit_transform(images_flat)

    # Perform PCA for dimensionality reduction

    num_components_range = np.arange(1, 100, 1)  # Adjust the step size as needed

    explained_variances = []

    for num_components in num_components_range:
        pca = PCA(n_components=num_components)
        pca.fit(images_flat)
        explained_variances.append(np.sum(pca.explained_variance_ratio_))

    # Set a threshold for cumulative explained variance
    threshold_variance = 0.95

    # Find the index where the cumulative explained variance crosses the threshold
    num_components_threshold = np.argmax(np.array(explained_variances) >= threshold_variance) + 1

    # Use the selected number of components in your PCA model
    num_pca_components = num_components_range[num_components_threshold - 1]

    # num_pca_components = 22  # Adjust as needed
    pca = PCA(n_components=num_pca_components)
    images_pca = pca.fit_transform(images_flat_standardized)

    # Apply k-means clustering
    num_clusters = 4
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(images_pca)

    # Visualize the clustering results
    fig, axes = plt.subplots(2, 5, figsize=(10, 4))
    for i in range(num_clusters):
        cluster_indices = np.where(cluster_labels == i)[0]
        sample_index = cluster_indices[0]
        image = images[sample_index]

        axes[0, i].imshow(image, cmap='gray')
        axes[0, i].axis('off')
        axes[0, i].set_title(f'Cluster {i}')

        axes[1, i].hist(cluster_labels[cluster_indices], bins=range(num_clusters), align='left', rwidth=0.8)
        axes[1, i].set_xticks(range(num_clusters))
        axes[1, i].set_title('Cluster Distribution')
        axes[1, i].set_xlabel('Cluster')
        axes[1, i].set_ylabel('Count')

    plt.tight_layout()
    plt.show()


import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from PIL import Image
import os
import matplotlib.pyplot as plt

def spectral_clustering():
    # Load images and convert them to grayscale
    image_folder_path = "/localhome/asa420/MIAL/data/classification_data/PCA"
    image_files = os.listdir(image_folder_path)
    images = []

    for file in image_files:
        image_path = os.path.join(image_folder_path, file)
        img = Image.open(image_path).convert('L')  # Convert to grayscale
        img = np.array(img).astype(float)
        images.append(img)

    # Stack images into a single matrix
    images_matrix = np.stack(images)

    # Flatten the images
    images_flat = images_matrix.reshape((images_matrix.shape[0], -1))

    # Standardize the data (optional, but can be helpful for Spectral Clustering)
    scaler = StandardScaler()
    images_flat_standardized = scaler.fit_transform(images_flat)

    num_components_range = np.arange(1, 100, 1)  # Adjust the step size as needed

    explained_variances = []

    for num_components in num_components_range:
        pca = PCA(n_components=num_components)
        pca.fit(images_flat)
        explained_variances.append(np.sum(pca.explained_variance_ratio_))

    # Set a threshold for cumulative explained variance
    threshold_variance = 0.95

    # Find the index where the cumulative explained variance crosses the threshold
    num_components_threshold = np.argmax(np.array(explained_variances) >= threshold_variance) + 1

    # Use the selected number of components in your PCA model
    num_pca_components = num_components_range[num_components_threshold - 1]

    # Perform PCA for dimensionality reduction
    # num_pca_components = 50  # Adjust as needed
    pca = PCA(n_components=num_pca_components)
    images_pca = pca.fit_transform(images_flat_standardized)

    # Apply Spectral Clustering
    num_clusters = 4  # Adjust as needed
    spectral = SpectralClustering(n_clusters=num_clusters, affinity='nearest_neighbors')
    cluster_labels = spectral.fit_predict(images_pca)

    # Visualize the clustering results
    fig, axes = plt.subplots(2, 4, figsize=(10, 4))
    for i in range(len(np.unique(cluster_labels))):
        cluster_indices = np.where(cluster_labels == i)[0]
        sample_index = cluster_indices[0]
        image = images[sample_index]

        axes[0, i].imshow(image, cmap='gray')
        axes[0, i].axis('off')
        axes[0, i].set_title(f'Cluster {i}')

        axes[1, i].hist(cluster_labels[cluster_indices], bins=range(num_clusters + 1), align='left', rwidth=0.8)
        axes[1, i].set_xticks(range(num_clusters))
        axes[1, i].set_title('Cluster Distribution')
        axes[1, i].set_xlabel('Cluster')
        axes[1, i].set_ylabel('Count')

    plt.tight_layout()
    plt.show()


# spectral_clustering()
    
import numpy as np
from sklearn.decomposition import PCA
from PIL import Image
import os
import matplotlib.pyplot as plt

# Load images and convert them to grayscale
image_folder_path = "/localhome/asa420/MIAL/data/classification_data/PCA/0"
image_files = os.listdir(image_folder_path)
images = []

# Assuming you have class labels for each image (modify as needed)
# class_labels = [0, 1, 2, 3] * 30  # Replace with your actual class labels
class_labels = [0] * 26# + [1] * 30 + [2] * 31 + [3] * 29

for file, label in zip(image_files, class_labels):
    image_path = os.path.join(image_folder_path, file)
    img = Image.open(image_path).convert('L')  # Convert to grayscale
    img = np.array(img).astype(float)
    images.append((img, label))

# Stack images into a single matrix
images_matrix = np.stack([img for img, label in images])

# Flatten the images
images_flat = images_matrix.reshape((images_matrix.shape[0], -1))

# Perform PCA for dimensionality reduction
num_pca_components = 20  # Adjust as needed
pca = PCA(n_components=num_pca_components)
images_pca = pca.fit_transform(images_flat)

# Separate the data by class
class_indices = [np.where(class_labels == i)[0] for i in range(1)]

# Calculate the mean image for each class
mean_images_per_class = []
for indices in class_indices:
    mean_image = np.mean(images_pca[indices], axis=0)
    mean_images_per_class.append(mean_image)

# Project the mean images back to the original space
mean_images_per_class_original_space = pca.inverse_transform(mean_images_per_class)

# Display the mean images for each class
fig, axes = plt.subplots(1, 4, figsize=(12, 3))
for i, mean_image in enumerate(mean_images_per_class_original_space):
    axes[i].imshow(mean_image.reshape(images_matrix.shape[1:3]), cmap='gray')
    axes[i].axis('off')
    axes[i].set_title(f'Mean Image\nClass {i}')

plt.show()
