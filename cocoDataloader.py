import os
import torch
import torch.utils.data
from PIL import Image
from pycocotools.coco import COCO
import json
import random


"""
====================================================================================================

    Main function to run
      - create_train_validation_test_loader

====================================================================================================
"""

"""
Create the train, valididation, test data loader individually

Waring:
    for train, validation, test split, we are randomizing then splitting the available video filenames
        (if there are two annotations for the same video, both annotations will belong to the same set)

Parameter:
    image_dir_path - string, path to the directory that contain all the image/frames used by the overall COCO annotation
    merged_coco_ann_path - string, (include json filename), path to overall COCO annotation json
    batch_size - int, batch size for the dataloader
    transform_fn - function, image transformations
                    at least should have:  torchvision.transforms.Compose([torchvision.transforms.ToTensor()] 
    video_file_id_map_path -  string, (include json filename), path to json that keeps track to all the video filenames and
                                        the mapping between video filenames and file ids (used by the ID Generator)
    train_validation_test_split - tuple, (train_percentage, validation_percentage)
        the test percentage is implicitly represented as: 1 - train_percentage - validation_percentage
"""
def create_train_validation_test_loader(image_dir_path, merged_coco_ann_path, batch_size, transform_fn, 
                                        video_file_id_map_path, train_validation_test_split):
    with open(video_file_id_map_path, "r") as f:
        video_filename_list = json.load(f)["filenames"]       # list of video names (without .mp4)

    train_pct, valid_pct = train_validation_test_split
    train_idx = int(len(video_filename_list) * train_pct)
    valid_idx = int(len(video_filename_list) * valid_pct)

    # shuffle the videos then split them
    random.shuffle(video_filename_list)

    train_video_filenames = video_filename_list[:train_idx]
    valid_video_filenames = video_filename_list[train_idx : train_idx + valid_idx]
    test_video_filenames = video_filename_list[train_idx + valid_idx:]

    coco = COCO(merged_coco_ann_path)

    # identify the keys in coco.imgs that belong to the individual dataset
    train_dataset_key, valid_dataset_key, test_dataset_key = filter_keys(train_video_filenames, valid_video_filenames, test_video_filenames, coco)

    train_dataset = CustomCocoDataset(image_dir_path, merged_coco_ann_path, sorted(train_dataset_key), transform_fn)
    valid_dataset = CustomCocoDataset(image_dir_path, merged_coco_ann_path, sorted(valid_dataset_key), transform_fn)
    test_dataset = CustomCocoDataset(image_dir_path, merged_coco_ann_path, sorted(test_dataset_key), transform_fn)

    return  create_dataloader(train_dataset, batch_size), create_dataloader(valid_dataset, batch_size), create_dataloader(test_dataset, batch_size)


"""
====================================================================================================

    Helper functions

====================================================================================================
"""

"""
Given a Pytorch dataset and the batch size, create the corresponding dataloader
"""
def create_dataloader(dataset, batch_size):
    # collate_fn needs for batch
    def collate_fn(batch):
        return tuple(zip(*batch))

    # own DataLoader
    # solve the issue of 
    return torch.utils.data.DataLoader(dataset,
                                        batch_size=batch_size,
                                        shuffle=True,
                                        num_workers=0,
                                        collate_fn=collate_fn)


"""
Filter and identify the image keys that belng to a particular dataaset

Parameters:
    train_video_filename_list - list of video filenames that belong to the training dataset
    valid_video_filename_list - list of video filenames that belong to the validation dataset
    test_video_filename_list - list of video filenames that belong to the testing dataset
    coco - coco annotation object, the overall coco that contains all the data

Return:
    train_dataset_key, valid_dataset_key, test_dataset_key
        3 lists of keys, each one specifying the key in coco.imags that belong to a dataset
            (the key essentially specifies that images belong to the dataset)
"""
def filter_keys(train_video_filename_list, valid_video_filename_list, test_video_filename_list, coco):
    train_dataset_key = []
    valid_dataset_key = []
    test_dataset_key = []

    for key, value in coco.imgs.items():
        img_id= value['id']

        if coco.getAnnIds(imgIds=[img_id]) != []:
            # get the video filename from the frame path
            #   get rid of the _2 ending
            video_filename_split = value['file_name'].split("_")
                
            if "_2" in value['file_name']:
                video_filename_split = video_filename_split[:-2]
            else:
                video_filename_split = video_filename_split[:-1]
            
            # recombnie the splitted video filenames into one string
            video_filename = video_filename_split[0]
            for video_filename_chuck in video_filename_split[1:]:
                video_filename += "_" + video_filename_chuck
            
            if video_filename in train_video_filename_list:
                train_dataset_key.append(key)
            elif video_filename in valid_video_filename_list:
                valid_dataset_key.append(key)
            elif video_filename in test_video_filename_list:
                test_dataset_key.append(key)
            else:
                print(f"ERROR: filename = {value['file_name']} does not belong to any dataset")
        else:
          print(f"WARNING: filename = {value['file_name']} does not have annotation")           

    return train_dataset_key, valid_dataset_key, test_dataset_key


"""
Note:
    Most of the code is taken from 
        https://medium.com/fullstackai/how-to-train-an-object-detector-with-your-own-coco-dataset-in-pytorch-319e7090da5

Parameters:
    root - string, the path to the directory that contain all the images/frames for the COCO annotation
    annotation - string, (including the json filename), path to merged COCO annotation
    img_ids - list, list of keys that correspond to images that will get used in the dataset
        If it is not specified, we assume to be using all the images for this dataset
    transforms - function, image transformations
                    at least should have:  torchvision.transforms.Compose([torchvision.transforms.ToTensor()] 
"""
class CustomCocoDataset(torch.utils.data.Dataset):
    def __init__(self, root, annotation, img_ids=None, transforms=None):
        self.root = root
        self.transforms = transforms
        self.coco = COCO(annotation)
        if img_ids == None:
            self.ids = list(sorted(self.coco.imgs.keys()))
        else:
            self.ids = img_ids

    """
    Required member function

    Given an index, return the image and the annotation at that index in the dataset
    """
    def __getitem__(self, index):
        # Own coco file
        coco = self.coco
        # Image ID
        img_id = self.ids[index]
        # List: get annotation id from coco
        ann_ids = coco.getAnnIds(imgIds=img_id)
        # Dictionary: target coco_annotation file for an image
        coco_annotation = coco.loadAnns(ann_ids)
        # path for input image
        path = coco.loadImgs(img_id)[0]['file_name']
        # open the input image
        img = Image.open(os.path.join(self.root, path))

        # number of objects in the image
        num_objs = len(coco_annotation)

        # Bounding boxes for objects
        # In coco format, bbox = [xmin, ymin, width, height]
        # In pytorch, the input should be [xmin, ymin, xmax, ymax]
        boxes = []
        for i in range(num_objs):
            xmin = coco_annotation[i]['bbox'][0]
            ymin = coco_annotation[i]['bbox'][1]
            xmax = xmin + coco_annotation[i]['bbox'][2]
            ymax = ymin + coco_annotation[i]['bbox'][3]
            boxes.append([xmin, ymin, xmax, ymax])
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # Labels
        labels = torch.ones((num_objs,), dtype=torch.int64)
        # Tensorise img_id
        img_id = torch.tensor([img_id])
        # Size of bbox (Rectangular)
        areas = []
        for i in range(num_objs):
            areas.append(coco_annotation[i]['area'])
        areas = torch.as_tensor(areas, dtype=torch.float32)
        # Iscrowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        # Annotation is in dictionary format
        my_annotation = {}
        my_annotation["boxes"] = boxes
        my_annotation["labels"] = labels
        my_annotation["image_id"] = img_id
        my_annotation["area"] = areas
        my_annotation["iscrowd"] = iscrowd

        if self.transforms is not None:
            img = self.transforms(img)

        return img, my_annotation

    def __len__(self):
        return len(self.ids)
