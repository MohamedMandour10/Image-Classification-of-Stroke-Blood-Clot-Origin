import os
import zipfile
import pandas as pd

def zip_directory(directory_path, output_zip_file):
    """
    Zips the contents of a directory and removes the original files.

    Args:
        directory_path (str): Path of the directory to be zipped.
        output_zip_file (str): Path of the resulting zip file.
    """
    with zipfile.ZipFile(output_zip_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Walk through the directory
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                # Get the file's full path
                file_path = os.path.join(root, file)
                # Add file to zip (with relative path)
                arcname = os.path.relpath(file_path, start=directory_path)
                zipf.write(file_path, arcname)
                # Remove the original file after adding to the zip
                os.remove(file_path)
    
    print(f"Directory '{directory_path}' successfully zipped into '{output_zip_file}'")


def delete_directory(directory_path):
    try:
        if os.path.exists(directory_path):
            for root, dirs, files in os.walk(directory_path, topdown=False):
                for file in files:
                    os.remove(os.path.join(root, file))
                for dir in dirs:
                    os.rmdir(os.path.join(root, dir))
            os.rmdir(directory_path)
            print(f"Deleted directory: {directory_path}")
        else:
            print(f"Directory does not exist: {directory_path}")
    except Exception as e:
        print(f"Error deleting directory {directory_path}: {e}")



def create_file_label_dataframe(base_dir):
    """
    Creates a DataFrame with file paths and labels by traversing a base directory.

    Args:
        base_dir (str): The base directory containing subdirectories for each label.

    Returns:
        pd.DataFrame: A DataFrame with two columns: 'path' and 'label'.
    """
    # Initialize lists to store paths and labels
    paths = []
    labels = []

    # Traverse the directory
    for label in os.listdir(base_dir):
        label_dir = os.path.join(base_dir, label)
        if os.path.isdir(label_dir):
            for file in os.listdir(label_dir):
                file_path = os.path.join(label_dir, file)
                if os.path.isfile(file_path):
                    paths.append(file_path)
                    labels.append(label)

    # Create the DataFrame
    df = pd.DataFrame({"path": paths, "label": labels})
    return df
