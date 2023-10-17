import torch
import torchvision
import os
import pickle
from PIL import Image
import tifffile
import numpy as np
import math

class SmartCrop(torchvision.transforms.RandomResizedCrop):
    """
    This is a modified version of random resized crop.

    It adds another layer of selection to the crop:
    Only accept crops where the starting pixel is not empty.

    Additionally, the scale parameter now affects scale of the crop, 
    not the scale of the image.
    """

    def __init__(
        self,
        size,
        scale=(0.08, 1.0),
        ratio=(3.0 / 4.0, 4.0 / 3.0),
        interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
        antialias = True,
    ):
        super().__init__(size = size,
                         scale = scale,
                         ratio = ratio,
                         interpolation = interpolation,
                         antialias = antialias)
        

    def get_params(self, img, scale, ratio):
        """Modified to check if pixel is empty before accepting crop"""

        _, height, width = torchvision.transforms.functional.get_dimensions(img)
        area = self.size[0] * self.size[1]

        log_ratio = torch.log(torch.tensor(ratio))
        for _ in range(10):
            target_area = area * torch.empty(1).uniform_(scale[0], scale[1]).item()
            aspect_ratio = torch.exp(torch.empty(1).uniform_(log_ratio[0], log_ratio[1])).item()

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                i = torch.randint(0, height - h + 1, size=(1,)).item()
                j = torch.randint(0, width - w + 1, size=(1,)).item()
                # modified here
                if img[:, i, j] != 0:
                    return i, j, h, w

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if in_ratio < min(ratio):
            w = width
            h = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = height
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w


class ImageNet2017(torchvision.datasets.ImageFolder):
    """Class to handle the 2017 ImageNet dataset. This is a subclass of the ImageFolder
    dataset class, so it has the same functionality. The only difference is that it
    automatically extracts the dataset if it is not already extracted, and it saves
    the dataset as a pickle file for faster loading in the future."""
    def __init__(self,
                 root,
                 split = "train",
                 transform=None,
                 tar_name = "ILSVRC2017_CLS-LOC.tar.gz",
                 extracted_path = "ILSVRC",
                 dataset_pkl = "imagenet_meta.pkl"):
        
        file_list = list(os.listdir(root))

        if (tar_name in file_list) and (extracted_path not in file_list):
            tar_path = os.path.join(root, tar_name)
            torchvision.datasets.utils.extract_archive(tar_path)
        
        self.root = os.path.join(root, extracted_path)

        if split == "val":
            reprocess_val(root)
            self.split_folder = os.path.join(self.root, "Data/CLS-LOC", f"val_parsed")
        else:
            self.split_folder = os.path.join(self.root, "Data/CLS-LOC", split)

        super().__init__(self.split_folder)

        self.transform = transform
        self.save_pkl(os.path.join(root, f"{split}_{dataset_pkl}"))
 
    
    def save_pkl(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
    
def load_pkl(filepath):
    with open(filepath, 'rb') as f:
        dataset = pickle.load(f)
    return dataset

def get_imagenet(root, split = "train", dataset_pkl = "imagenet_meta.pkl", transform = None):
    """Function to get the 2017 imagenet dataset from an already downloaded file.
    Handles pickling and pickle loading the dataset for faster loading in the future."""
    full_pkl_path = os.path.join(root, f"{split}_{dataset_pkl}")
    if os.path.exists(full_pkl_path):
        dataset = load_pkl(full_pkl_path)
    else:
        dataset = ImageNet2017(root = root, split = split, dataset_pkl = dataset_pkl)

    if transform is not None:
        dataset.transform = transform

    return dataset

def reprocess_val(root):
    """Reprocesses the validation set from the common ImageNet2017 download to be similar
    to the training data (i.e. in folders by class). This also makes it compatible with the
    ImageFolder dataset class."""
    out_path = os.path.join(root, "ILSVRC/Data/CLS-LOC/val_parsed")
    if not os.path.exists(out_path):
        os.makedirs(out_path, exist_ok = True)
        # importing here cause not needed elsewhere
        print("Reprocessing validation data...")
        import xml.etree.ElementTree as ET
        xmls = os.listdir(os.path.join(root, "ILSVRC/Annotations/CLS-LOC/val"))
        for xml in xmls:
            if xml.endswith(".xml"):
                # parse the xml to get classname
                raw_filename = xml.split(".")[0]
                xml_path = os.path.join(root, "ILSVRC/Annotations/CLS-LOC/val", xml)
                tree = ET.parse(xml_path)
                rt = tree.getroot()
                image_class = rt.find("object").find("name").text

                # make a class dir and move the file
                os.makedirs(os.path.join(out_path, image_class), exist_ok = True)
                os.rename(os.path.join(root, "ILSVRC/Data/CLS-LOC/val", raw_filename + ".JPEG"),
                        os.path.join(out_path, image_class, raw_filename + ".JPEG"))


class VesuviusDataset(torch.utils.data.Dataset):
    def __init__(self,
                 path,
                 acceptable_size = 200 * 200,
                 acceptable_missing = 0.1,
                 transform = None,
                 ):
        """
        A dataset class for the Vesuvius dataset. This dataset is a collection of tif files
        with corresponding masks.

        Parameters
        ----------
        path : str
            The path to the directory containing the data.
        acceptable_size : int, optional
            The minimum number of nonmissing pixels required in a tif file to be included.
            The default is 200 * 200.
        transform : torchvision.transforms, optional
            A transform to apply to the data. The default is None.
        """
        super().__init__()
        self.path = path
        self.transform = transform
        self.tifs, self.masks = self.get_tifs()
        self.acceptable_missing = acceptable_missing
        if not os.path.exists(os.path.join(self.path, "tifs.pkl")):
            self.preprocess(required_not_missing = acceptable_size)
        else:
            with open(os.path.join(self.path, "tifs.pkl"), 'rb') as f:
                self.tifs = pickle.load(f)
        self.valid_counter = 0

    def get_tifs(self):
        # get valid tifs in path
        tifs = []
        masks = {}
        for root, dirs, files in os.walk(self.path):
            for file in files:
                if file.endswith(".tif"):
                    if ("mask" not in file):
                        tifs.append(os.path.join(root, file))
                    else:
                        tif_no_mask = file.replace("_mask", "")
                        masks[os.path.join(root, tif_no_mask)] = os.path.join(root, file)

        return tifs, masks

    def preprocess(self, required_not_missing = 100 * 200):
        clean_tifs = []
        running_mean = None
        running_var = None

        for tif in self.tifs:
            tif_img = tifffile.imread(tif)

            # convert uint16 to float32
            tif_img = tif_img.astype(np.float32) / 65535

            # check for missing data
            not_missing = np.sum(np.array(tif_img) != 0)
            if not_missing < required_not_missing:
                print(f"Skipping {tif} due to lack of nonmissing data ({not_missing} good pixels)")
                continue

            # calc running mean and var
            if running_mean is None:
                running_mean = tif_img.mean()
                running_var = tif_img.var()
            else:
                running_mean = (running_mean + tif_img.mean()) / 2
                running_var = (running_var + tif_img.var()) / 2

            clean_tifs.append(tif)
        
        print(f"Found {len(clean_tifs)} valid tifs (removed {len(self.tifs) - len(clean_tifs)})")
        print(f"Running mean: {running_mean}, running var: {running_var}")
        self.tifs = clean_tifs

        # save the tifs
        with open(os.path.join(self.path, "tifs.pkl"), 'wb') as f:
            pickle.dump(self.tifs, f)

    def __getitem__(self, index):
        path = self.tifs[index]
        img_pil = tifffile.imread(path)

        # convert uint16 to float32
        img_pil = img_pil.astype(np.float32) / 65535
            
        if self.transform is not None:

            img = self.transform(img_pil)
            # keep transforming until we get a valid image
            missing = torch.sum(img == 0) / torch.numel(img)
            n_tries = 0
            while missing > self.acceptable_missing:
                img = self.transform(img_pil)
                missing = torch.sum(img == 0) / torch.numel(img)
                n_tries += 1
                if n_tries > 10:
                    break
            self.valid_counter += n_tries
        else:
            img = img_pil
        return img

    def __len__(self):
        return len(self.tifs)
    
VESUVIUS_TRANSFORM = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                            SmartCrop((128, 
                                                       128),
                                                       scale = (0.9, 1.1),
                                                       ratio = (1, 1),
                                                       interpolation = torchvision.transforms.InterpolationMode.NEAREST),
                                            torchvision.transforms.RandomHorizontalFlip(0.5),
                                            torchvision.transforms.RandomVerticalFlip(0.5),
                                            torchvision.transforms.Normalize(mean = [0.395],
                                                                             std = [0.386])])

#TODO : add updates for validation data (which does not have the same structure as train data)
if __name__ == "__main__":
    from tqdm import tqdm
    test_imagenet = False

    if test_imagenet:
        root = "Z:/"
        split = "val"
        dataset = get_imagenet(root, split = split)
    else:
        root = "D:/Projects/scrolls"
        dataset = VesuviusDataset(root, transform = VESUVIUS_TRANSFORM)

        for i in tqdm(range(1000)):
            j = np.random.randint(len(dataset))
            dataset[j]

        print(f"Average number of tries to get valid image: {dataset.valid_counter / 1000}")

        