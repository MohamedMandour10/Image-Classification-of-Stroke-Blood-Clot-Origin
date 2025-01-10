import os
import zipfile
from huggingface_hub import HfApi, hf_hub_download

class HuggingFaceRepoManager:
    """
    A utility class for managing uploads and downloads with Hugging Face repositories.
    """
    def __init__(self):
        self.api = HfApi()
        self.namespace = self._get_user_namespace()

    def _get_user_namespace(self):
        """
        Retrieves the current user's namespace from the Hugging Face API.
        """
        user_info = self.api.whoami()
        return user_info['name']

    def upload_to_huggingface(self, zip_file_path, repo_name, repo_type="dataset"):
        """
        Uploads a zip file to a Hugging Face repository.

        Args:
            zip_file_path (str): Path to the zip file to upload.
            repo_name (str): Name of the Hugging Face repository.
            repo_type (str): Type of the repository ('model', 'dataset', or 'space').
        """
        # Check if the zip file exists
        if not os.path.exists(zip_file_path):
            raise FileNotFoundError(f"The file {zip_file_path} does not exist.")

        # Form the correct repo_id
        repo_id = f"{self.namespace}/{repo_name}"

        # Upload the zip file
        try:
            self.api.upload_file(
                path_or_fileobj=zip_file_path,
                path_in_repo=os.path.basename(zip_file_path),
                repo_id=repo_id,
                repo_type=repo_type,
            )
            print(f"Successfully uploaded '{zip_file_path}' to '{repo_id}' on Hugging Face.")
        except Exception as e:
            print(f"Failed to upload file: {e}")

    def download_and_extract_zip(self, repo_id, file_name, output_dir, repo_type="dataset"):
        """
        Downloads a zip file from a Hugging Face repository and extracts its contents.

        Args:
            repo_id (str): The repository ID in the form 'namespace/repo_name'.
            file_name (str): The name of the zip file in the repository.
            output_dir (str): Directory to extract the contents of the zip file.
            repo_type (str): The type of the repository ('model', 'dataset', or 'space').
        """
        # Download the zip file
        zip_file_path = hf_hub_download(repo_id=repo_id, filename=file_name, repo_type=repo_type)
        print(f"Downloaded '{file_name}' from '{repo_id}' to '{zip_file_path}'.")

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Extract the zip file
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
        print(f"Extracted contents to '{output_dir}'.")
