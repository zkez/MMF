from torch.utils.data import Dataset
import os
import numpy as np
import torch
import random
import torchvision.transforms as transforms
from sklearn.preprocessing import LabelEncoder
from scipy.ndimage import zoom


# Data Augmentation Transforms
class RandomFlip(object):
    """
    Randomly flip the input data horizontally and/or vertically.
    """
    def __call__(self, sample):
        """
        Args:
            sample (numpy array): Input sample to be flipped.

        Returns:
            numpy array: Flipped sample.
        """
        if random.random() > 0.5:
            sample = np.flip(sample, axis=0).copy()  # Horizontal flip
        if random.random() > 0.5:
            sample = np.flip(sample, axis=1).copy()  # Vertical flip
        return sample


class RandomRotation(object):
    """
    Randomly rotate the input data by 0, 90, 180, or 270 degrees.
    """
    def __call__(self, sample):
        """
        Args:
            sample (numpy array): Input sample to be rotated.

        Returns:
            numpy array: Rotated sample.
        """
        angle = random.choice([0, 90, 180, 270])
        sample = np.rot90(sample, k=angle // 90, axes=(0, 1)).copy()
        return sample


class AddGaussianNoise(object):
    """
    Add Gaussian noise to the input data.
    """
    def __call__(self, sample, mean=0, std=0.05):
        """
        Args:
            sample (numpy array): Input sample to add noise to.
            mean (float): Mean of the Gaussian noise.
            std (float): Standard deviation of the Gaussian noise.

        Returns:
            numpy array: Sample with added Gaussian noise.
        """
        noise = np.random.normal(mean, std, sample.shape)
        sample = sample + noise
        sample = np.clip(sample, 0, 1)  # Ensure the values are in the range [0, 1]
        return sample


# Data Augmentation Pipeline
transform_pipe = transforms.Compose([
    RandomFlip(),
    RandomRotation(),
    AddGaussianNoise()
])


class BaseLungNoduleDataset(Dataset):
    """
    Base class for Lung Nodule Dataset, containing common functionality.
    """
    def __init__(self, csv_data, data_dir, text_data=None, seg_dir=None, normalize=True, transform=None, augment_minority_class=True):
        """
        Args:
            csv_data (DataFrame): Metadata containing image paths and labels.
            data_dir (str): Directory containing image data.
            text_data (DataFrame, optional): DataFrame containing additional information about patients.
            seg_dir (str, optional): Directory containing segmentation data.
            normalize (bool): Whether to normalize images.
            transform (callable, optional): A function/transform to apply to the images.
            augment_minority_class (bool): Whether to apply augmentation to minority class samples.
        """
        self.data_dir = data_dir
        self.csv_data = csv_data
        self.text_data = dict(zip(text_data['pid'], text_data[['race', 'cigsmok', 'gender', 'age']].values.tolist())) if text_data is not None else None
        self.normalize = normalize
        self.seg_dir = seg_dir
        self.transform = transform
        self.augment_minority_class = augment_minority_class
        self.csv_data.reset_index(drop=True, inplace=True)
        self.subject_ids = self.csv_data['Subject ID'].unique()

    def __len__(self):
        """
        Returns:
            int: Number of unique subjects in the dataset.
        """
        return len(self.subject_ids)

    def normalize_image(self, image):
        """
        Normalize image to zero mean and unit variance.

        Args:
            image (numpy array): Image to be normalized.

        Returns:
            numpy array: Normalized image.
        """
        mean = np.mean(image)
        std = np.std(image)
        if std > 0:
            image = (image - mean) / std
        else:
            image = image - mean
        return image

    def load_image(self, file_path):
        """
        Load an image from a given file path.

        Args:
            file_path (str): Path to the image file.

        Returns:
            numpy array: Loaded image.
        """
        return np.load(file_path).astype(np.float32)


class LungNoduleDataset(BaseLungNoduleDataset):
    """
    Dataset for loading lung nodule images and corresponding information.
    """
    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: Tuple containing T0_image, T1_image, T0_seg, T1_seg, label, and text_input.
        """
        if idx >= len(self.subject_ids):
            raise IndexError(f"Index {idx} is out of bounds for axis 0 with size {len(self.subject_ids)}")

        subject_id = self.subject_ids[idx]
        subject_data = self.csv_data[self.csv_data['Subject ID'] == subject_id]
        T0_row = subject_data[subject_data['study_yr'] == 'T0']
        T1_row = subject_data[subject_data['study_yr'] == 'T1']
        T2_row = subject_data[subject_data['study_yr'] == 'T2']

        if T0_row.empty or T1_row.empty or T2_row.empty:
            raise ValueError(f"Missing data for subject {subject_id}")

        T0_path = os.path.join(self.data_dir, f"{subject_id}_T0.npy")
        T1_path = os.path.join(self.data_dir, f"{subject_id}_T1.npy")
        T0_seg_path = os.path.join(self.seg_dir, f"{subject_id}_T0_seg.npy")
        T1_seg_path = os.path.join(self.seg_dir, f"{subject_id}_T1_seg.npy")
        
        T2_label = T2_row.iloc[0]['label']
        T2_label = int(T2_label)

        T0_image = self.load_image(T0_path)
        T1_image = self.load_image(T1_path)
        T0_seg = self.load_image(T0_seg_path)
        T1_seg = self.load_image(T1_seg_path)
        label = int(T2_row.iloc[0]['label'])

        if self.normalize:
            T0_image = self.normalize_image(T0_image)
            T1_image = self.normalize_image(T1_image)
            
        if self.augment_minority_class and T2_label == 1 and self.transform:
            T0_image = self.transform(T0_image)
            T1_image = self.transform(T1_image)

        T0_image = torch.tensor(T0_image, dtype=torch.float32, requires_grad=True)
        T1_image = torch.tensor(T1_image, dtype=torch.float32, requires_grad=True)
        T0_seg = torch.tensor(T0_seg, dtype=torch.float32, requires_grad=True)
        T1_seg = torch.tensor(T1_seg, dtype=torch.float32, requires_grad=True)
        label = torch.tensor(label, dtype=torch.float32, requires_grad=True)

        text_input = self.text_data.get(subject_id)
        if text_input is None:
            raise ValueError(f"No text data found for Subject ID: {subject_id}")

        text_input = torch.tensor(text_input, dtype=torch.long)
        return T0_image, T1_image, T0_seg, T1_seg, label, text_input


class LungNodule2DSliceDataset(BaseLungNoduleDataset):
    """
    Dataset for extracting a specific 2D slice from lung nodule 3D images.
    """
    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: Tuple containing T1_slice and label.
        """
        if idx >= len(self.subject_ids):
            raise IndexError(f"Index {idx} is out of bounds for axis 0 with size {len(self.subject_ids)}")

        subject_id = self.subject_ids[idx]
        subject_data = self.csv_data[self.csv_data['Subject ID'] == subject_id]
        T1_row = subject_data[subject_data['study_yr'] == 'T1']

        if T1_row.empty:
            raise ValueError(f"Missing data for subject {subject_id}")

        T1_label = int(T1_row.iloc[0]['label'])
        T1_path = os.path.join(self.data_dir, f"{subject_id}_T1.npy")

        if not os.path.exists(T1_path):
            raise FileNotFoundError(f"File not found: {T1_path}")

        T1_image = self.load_image(T1_path)

        if T1_image.shape[0] < 8:
            raise ValueError(f"3D image for subject {subject_id} does not have 8 slices.")

        T1_slice = T1_image[7]  # Extract the 8th slice (index 7)

        if self.normalize:
            T1_slice = self.normalize_image(T1_slice)
            
        if self.transform:
            T1_slice = self.transform(T1_slice)

        T1_slice = torch.tensor(T1_slice, dtype=torch.float32, requires_grad=True)
        label = torch.tensor(T1_label, dtype=torch.float32, requires_grad=True)

        return T1_slice, label


class LungNoduleResizedDataset(BaseLungNoduleDataset):
    """
    Dataset for resizing lung nodule images to a target size using cubic interpolation.
    """
    def __init__(self, csv_data, data_dir, text_data, target_size=(64, 64, 64), normalize=True, transform=None, augment_minority_class=True):
        """
        Args:
            csv_data (DataFrame): Metadata containing image paths and labels.
            data_dir (str): Directory containing image data.
            text_data (DataFrame): DataFrame containing additional information about patients.
            target_size (tuple): Target size to resize the images to.
            normalize (bool): Whether to normalize images.
            transform (callable, optional): A function/transform to apply to the images.
            augment_minority_class (bool): Whether to apply augmentation to minority class samples.
        """
        super().__init__(csv_data, data_dir, text_data, normalize=normalize, transform=transform, augment_minority_class=augment_minority_class)
        self.target_size = target_size

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: Tuple containing T0_image, T1_image, and label.
        """
        if idx >= len(self.subject_ids):
            raise IndexError(f"Index {idx} is out of bounds for axis 0 with size {len(self.subject_ids)}")

        subject_id = self.subject_ids[idx]
        subject_data = self.csv_data[self.csv_data['Subject ID'] == subject_id]
        T0_row = subject_data[subject_data['study_yr'] == 'T0']
        T1_row = subject_data[subject_data['study_yr'] == 'T1']

        if T0_row.empty or T1_row.empty:
            raise ValueError(f"Missing data for subject {subject_id}")

        T0_path = os.path.join(self.data_dir, f"{subject_id}_T0.npy")
        T1_path = os.path.join(self.data_dir, f"{subject_id}_T1.npy")
        T1_label = int(T1_row.iloc[0]['label'])

        T0_image = self.load_image(T0_path)
        T1_image = self.load_image(T1_path)

        T0_image = self.resize_image(T0_image, target_size=self.target_size)
        T1_image = self.resize_image(T1_image, target_size=self.target_size)

        if self.normalize:
            T0_image = self.normalize_image(T0_image)
            T1_image = self.normalize_image(T1_image)

        if self.augment_minority_class and T1_label == 1 and self.transform:
            T0_image = self.transform(T0_image)
            T1_image = self.transform(T1_image)

        T0_image = torch.tensor(T0_image, dtype=torch.float32, requires_grad=True)
        T1_image = torch.tensor(T1_image, dtype=torch.float32, requires_grad=True)
        label = torch.tensor(T1_label, dtype=torch.float32, requires_grad=True)

        return T0_image, T1_image, label

    def resize_image(self, image, target_size):
        """
        Resize image to the target size using cubic interpolation.

        Args:
            image (numpy array): Image to be resized.
            target_size (tuple): Target size for resizing.

        Returns:
            numpy array: Resized image.
        """
        zoom_factors = [
            target_size[i] / image.shape[i] for i in range(3)
        ]
        resized_image = zoom(image, zoom_factors, order=3)  # order=3 indicates cubic interpolation
        return resized_image


class LungNoduleTextDataset(BaseLungNoduleDataset):
    """
    Dataset for loading lung nodule images along with corresponding tabular data.
    """
    def __init__(self, csv_data, data_dir, text_data, normalize=True, transform=None, augment_minority_class=True):
        """
        Args:
            csv_data (DataFrame): Metadata containing image paths and labels.
            data_dir (str): Directory containing image data.
            text_data (DataFrame): DataFrame containing additional tabular information about patients.
            normalize (bool): Whether to normalize images.
            transform (callable, optional): A function/transform to apply to the images.
            augment_minority_class (bool): Whether to apply augmentation to minority class samples.
        """
        super().__init__(csv_data, data_dir, text_data, normalize=normalize, transform=transform, augment_minority_class=augment_minority_class)
        self.specific_columns = ['race', 'cigsmok', 'gender', 'age', 'scr_res0', 'scr_iso0']
        self.text_data = text_data.set_index('pid')[self.specific_columns].fillna('NA').astype('category')
        self.label_encoders = {col: LabelEncoder().fit(self.text_data[col]) for col in self.specific_columns}
        for col in self.specific_columns:
            self.text_data[col] = self.label_encoders[col].transform(self.text_data[col])

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            dict: Dictionary containing T0_image, T1_image, label, and tabular data.
        """
        if idx >= len(self.subject_ids):
            raise IndexError(f"Index {idx} is out of bounds for axis 0 with size {len(self.subject_ids)}")

        subject_id = self.subject_ids[idx]
        subject_data = self.csv_data[self.csv_data['Subject ID'] == subject_id]
        T0_row = subject_data[subject_data['study_yr'] == 'T0']
        T1_row = subject_data[subject_data['study_yr'] == 'T1']

        if T0_row.empty or T1_row.empty:
            raise ValueError(f"Missing data for subject {subject_id}")

        T0_path = os.path.join(self.data_dir, f"{subject_id}_T0.npy")
        T1_path = os.path.join(self.data_dir, f"{subject_id}_T1.npy")
        T1_label = int(T1_row.iloc[0]['label'])

        T0_image = self.load_image(T0_path)
        T1_image = self.load_image(T1_path)

        if self.normalize:
            T0_image = self.normalize_image(T0_image)
            T1_image = self.normalize_image(T1_image)

        T0_image = self.apply_transform(T0_image, T1_label)
        T1_image = self.apply_transform(T1_image, T1_label)

        T0_image = torch.tensor(T0_image, dtype=torch.float32, requires_grad=True)
        T1_image = torch.tensor(T1_image, dtype=torch.float32, requires_grad=True)
        label = torch.tensor(T1_label, dtype=torch.float32, requires_grad=True)

        table_info = self.text_data.loc[subject_id].values
        table_info = torch.tensor(table_info, dtype=torch.int64)

        return {
            'T0_image': T0_image,
            'T1_image': T1_image,
            'label': label,
            'table_info': table_info
        }
