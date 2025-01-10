# Function to plot the class distribution
import torch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def plot_confusion_matrix(model, loader, device, num_classes, title="Confusion Matrix", normalize=True):
    """
    Plots a confusion matrix for the given model and dataset.

    Args:
        model: Trained PyTorch model.
        loader: DataLoader for the dataset (train, val, or test).
        device: Device ('cuda' or 'cpu') to perform computations.
        num_classes: Number of classes in the dataset.
        title: Title for the confusion matrix plot.
        normalize: If True, normalize the confusion matrix by row.
    """
    model.eval()  # Set the model to evaluation mode
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            preds = outputs.argmax(1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds, labels=np.arange(num_classes))
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
        cm = np.nan_to_num(cm)  # Handle division by zero for rows with no samples

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.arange(num_classes))
    
    # Plot confusion matrix
    fig, ax = plt.subplots(figsize=(10, 10))
    cmap = "Blues" if normalize else None
    values_format = ".2f" if normalize else "d"
    disp.plot(ax=ax, cmap=cmap, values_format=values_format)
    plt.title(title)
    plt.savefig(f"{title.replace(' ', '_').lower()}.png")
    plt.show()

def plot_class_distribution(df, class_column='label'):
    """
    Plot the distribution of classes in the dataset.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        class_column (str): Name of the column containing class labels.
    """
    # Count the occurrences of each class
    class_counts = df[class_column].value_counts()

    # Plot the class distribution as a bar chart
    plt.figure(figsize=(10, 6))
    class_counts.plot(kind='bar', color='skyblue', edgecolor='black')
    
    # Add labels and title
    plt.title('Class Distribution', fontsize=16)
    plt.xlabel('Class', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.xticks(rotation=45, fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Show the plot
    plt.tight_layout()
    plt.show()


# Function to plot random samples with labels
def plot_random_samples(df, num_samples=5):
    """
    Display a random selection of images from the given DataFrame along with their corresponding labels.

    Args:
        df (pd.DataFrame): DataFrame containing image paths and labels. Must have columns 'path' and 'label'.
        num_samples (int, optional): Number of random samples to display. Defaults to 5.

    Returns:
        None: Displays the images and their labels in a plot.
    """
    # Randomly sample 'num_samples' rows from the dataframe
    sampled_df = df.sample(n=num_samples)

    # Set up the plot grid (e.g., 1 row, num_samples columns)
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 5))

    # Loop through the sampled images and display them
    for i, (index, row) in enumerate(sampled_df.iterrows()):
        image_path = row['path']
        label = row['label']

        # Load and display the image
        image = Image.open(image_path).convert('RGB')
        axes[i].imshow(image)
        axes[i].axis('off')  # Hide axis

        # Set title with label
        axes[i].set_title(f"Label: {label}")

    plt.tight_layout()
    plt.show()
