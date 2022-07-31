from lxml import etree
import os
from parse_params import parse_params
import constants
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt

def preprocess_data(hyp_dir: str = "hyp/hyp-mask-face.yaml"):
    params = parse_params(hyp_dir)
    annotation_dir = params['train'] + constants.ANNOTATION_PATH
    image_dir = params['train'] + constants.IMAGES_PATH

    return preprocess_annotation_data(annotation_dir) 



def preprocess_annotation_data(annotation_dir: str) -> list[dict]:
    """
    Parameter:
        - annotation_dir: the location of annotation folder that contain the annotate information of each image
    ------
    Return the list of annotate information about each picture
    Each picture information will be stored as a diction with the information:
        - folder: folder name that include the picture -> str
        - image: image name correspoding to the annotate information -> str
        - spatial_size: the spatial size of the picture (width, height, depth) -> int
        - objects: include information of each object in the picture
            + Class name of the object (str)
            + xmin, ymin, xmax, ymax: the coordinate of bounding box that specify the object in picture (int)
    """

    annotation_loc = []
    for file_name in os.listdir(annotation_dir):
        annotation_loc.append(annotation_dir + "/" + file_name)

    annotation_info = []
    for file_loc in annotation_loc:
        image_anno = {}
        xml_data_tree = etree.parse(file_loc)
        root = xml_data_tree.getroot()

        # Get folder name of image
        folder_ele = root.find("folder")
        image_anno["folder"] = folder_ele.text

        # Get name of image
        image_name = root.find("filename").text
        image_anno["image"] = image_name

        # Get spatial size infomation
        size_tag = root.find("size")
        image_anno["spatial_size"] = {}

        for spatial_size in size_tag:
            image_anno["spatial_size"][spatial_size.tag] = int(spatial_size.text)

        # Get object class information and bouding box information
        image_anno["objects"] = []

        for object_iter in root.iter("object"):
            object_info = {}

            class_name = object_iter.find("name").text
            object_info["class_name"] = class_name

            for feature_info in object_iter.find("bndbox"):
                object_info[feature_info.tag] = int(feature_info.text)

            image_anno["objects"].append(object_info)

        annotation_info.append(image_anno)
    return annotation_info

def preprocess_image(image_dir, spatial_dim: tuple[int, int]):
    images_loc = []

    for images_name in os.listdir(image_dir):
        images_loc.append(image_dir + '/' + images_name)
        
    for image_file in images_loc:
        orig_image = Image.open(image_file)
        print(orig_image.shape)

    




