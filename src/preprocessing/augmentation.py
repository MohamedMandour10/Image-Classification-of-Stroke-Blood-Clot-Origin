from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import torch

class CustomImageDataset:
    def __init__(self, df_train, df_val, df_test, batch_size=32, num_workers=4):
        """
        Initializes the CustomImageDataset class to manage datasets and dataloaders.

        Args:
            df_train (pd.DataFrame): DataFrame for training data.
            df_val (pd.DataFrame): DataFrame for validation data.
            df_test (pd.DataFrame): DataFrame for testing data.
            batch_size (int): Batch size for DataLoader.
            num_workers (int): Number of workers for DataLoader.
        """
        self.df_train = df_train
        self.df_val = df_val
        self.df_test = df_test
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Define transformations
        self.train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
            transforms.RandomRotation(degrees=45),
            transforms.ColorJitter(brightness=0.2, hue=0.3, saturation=0.3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.val_test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Create datasets and dataloaders
        self.train_dataset = self.ImageDataset(self.df_train, transform=self.train_transform)
        self.val_dataset = self.ImageDataset(self.df_val, transform=self.val_test_transform)
        self.test_dataset = self.ImageDataset(self.df_test, transform=self.val_test_transform)

        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    class ImageDataset(Dataset):
        def __init__(self, df, transform=None):
            """
            Custom Dataset class for loading image data.

            Args:
                df (pd.DataFrame): DataFrame containing image paths and labels.
                transform (callable, optional): Transformations to apply to the images.
            """
            self.df = df
            self.transform = transform or transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        def __len__(self):
            return len(self.df)

        def __getitem__(self, idx):
            img_path = self.df.iloc[idx]["path"]
            label = self.df.iloc[idx]["label"]

            image = Image.open(img_path).convert("RGB")
            image = self.transform(image)
            return image, torch.tensor(label, dtype=torch.long)

    def preview_batch(self, loader_type="train"):
        """
        Previews a batch from the specified DataLoader.

        Args:
            loader_type (str): Type of DataLoader ('train', 'val', 'test').
        """
        loader_map = {
            "train": self.train_loader,
            "val": self.val_loader,
            "test": self.test_loader
        }
        loader = loader_map.get(loader_type, self.train_loader)

        for images, labels in loader:
            print(f"{loader_type.capitalize()} loader:")
            print(f"Image batch shape: {images.shape}")
            print(f"Labels batch shape: {labels.shape}")
            break  # Only preview the first batch
