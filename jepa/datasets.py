import torch
import torchvision
import os
import pickle
from PIL import Image
import tifffile
import numpy as np
import math
from tqdm import tqdm
import yaml
import gc

config = yaml.safe_load(open("../config/training.yml", "r"))

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

        height, width = torchvision.transforms.functional.get_dimensions(img)[-2:]
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
                if img[:, i, j].sum() != 0:
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

preproccess_transform = torchvision.transforms.Compose([
                                            # first crop makes subsequent ops faster
                                            SmartCrop((700, 700),
                                                        scale = (0.9, 1.1),
                                                        ratio = (1, 1),
                                                        interpolation = torchvision.transforms.InterpolationMode.NEAREST),
                                            torchvision.transforms.RandomAffine(degrees = 180,
                                                                                shear=2),
                                            torchvision.transforms.RandomHorizontalFlip(0.5),
                                            torchvision.transforms.RandomVerticalFlip(0.5),
                                            SmartCrop((512, 512),
                                                       scale = (0.9, 1.1),
                                                       ratio = (1, 1),
                                                       interpolation = torchvision.transforms.InterpolationMode.NEAREST),
                                            ])


MAIN_TRANSFORM = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                torchvision.transforms.RandomCrop((config["crop_h"], config["crop_w"])),
                                                torchvision.transforms.RandomApply([torchvision.transforms.GaussianBlur(7, sigma = (0.1, 2))], 
                                                                                p = 0.1),
                                                torchvision.transforms.RandomApply([torchvision.transforms.ElasticTransform(alpha = 10.0, 
                                                                                                                        sigma = 1.5)],
                                                                                p = 0.1),
                                                ])

#TODO : add updates for validation data (which does not have the same structure as train data)
if __name__ == "__main__":
    test_imagenet = False

    if test_imagenet:
        root = "Z:/"
        split = "val"
        dataset = get_imagenet(root, split = split)

        