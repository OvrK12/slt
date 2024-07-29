import os
import gzip
import pickle
import torch
import torchvision
from collections import defaultdict
from tqdm import tqdm
import xml.etree.ElementTree as ET
from PIL import Image
import random
import albumentations as A
import numpy as np
from openai import OpenAI

def create_pil_images(data_path):
    """Create a dictionary of PIL images for each video in the given data path.

    Args:
        data_path (str): Path to the directory containing the video files

    Returns:
        dict: A dictionary where keys are video names and values are lists of the corresponding PIL images.
    """    
    pil_image_dict = defaultdict(list)
    video_dirs = os.listdir(data_path)

    for dir in tqdm(video_dirs, desc=f"Creating pil images for {data_path}"):
        dir_path = os.path.join(data_path, dir)
        video_imgs = []
        for img in os.listdir(dir_path):
            if img != ".history.forster":
                path = os.path.join(dir_path, img)
                image = Image.open(path).convert("RGB")
                video_imgs.append(image)
        pil_image_dict[dir] = video_imgs

    return pil_image_dict


class DataProcessor:
    def __init__(self, config):
        self.config = config
        self.data_dicts = []
        self.openai_client = None
        # you need to have an openAI API key set in your env variables for this to work
        # see: https://platform.openai.com/docs/quickstart
        if config['gpt_subs'] or config['gpt_full']:
            self.openai_client = OpenAI()        

    def dump_data(self, filename):
        """Dump the processed data_dicts to a pickle file.

        Args:
            filename (str): Name of the file to save the data.
        """        
        with gzip.open(filename, "wb") as data_file:
            pickle.dump(self.data_dicts, data_file)

    def load_data(self, filename):
        """Load data_dicts from a pickle file.

        Args:
            filename (str): Name of the file to load the data from.

        Returns:
            list: data_dicts from the pickle file.
        """             
        with gzip.open(filename, "rb") as f:
            return pickle.load(f)

    def split_data(self, filename, rate=(0.7, 0.1)):
        """Split the data into train, dev, and test sets and save them to separate files.

        Args:
            filename (str): Base filename for the split data files.
            rate (tuple, optional): Tuple containing the split ratios for train and dev sets. Remaining ratio is assigned to test.
        """       
        train_file = filename[:-7] + "_train.pickle"
        dev_file = filename[:-7] + "_dev.pickle"
        test_file = filename[:-7] + "_test.pickle"

        train_index = int(len(self.data_dicts) * rate[0])
        dev_index = train_index + int(len(self.data_dicts[train_index:]) * rate[1])

        with gzip.open(train_file, "wb") as train:
            pickle.dump(self.data_dicts[:train_index], train)
        with gzip.open(dev_file, "wb") as dev:
            pickle.dump(self.data_dicts[train_index:dev_index], dev)
        with gzip.open(test_file, "wb") as test:
            pickle.dump(self.data_dicts[dev_index:], test)

        print(f"Train data: {len(self.data_dicts[:train_index])}")
        print(f"Dev data: {len(self.data_dicts[train_index:dev_index])}")
        print(f"Test data: {len(self.data_dicts[dev_index:])}")
    
    def gloss_set(self, gloss):
        """Create a set of unique words from a gloss string.
        The set is used to compare the original glosses to the preprocessed glosses.

        Args:
            gloss (str): The gloss string to process.

        Returns:
            set: The wordset extracted from the gloss string.
        """      
        gloss_set = set()
        for word in gloss.split(" "):
            if word.startswith("loc-"):
                gloss_set.add("LOC")
                gloss_set.add(word[4:])
            elif "-" in word:
                gloss_set.update(word.split("-"))
            else:
                gloss_set.add(word)
        return gloss_set

    def match_gloss(self, orig_gloss):
        """Find the index of the best matching gloss from the ground truth glosses.

        Args:
            orig_gloss (str): The original gloss to match.

        Returns:
            int: Index of the best matching gloss in the ground truth set.
        """     
        # create word set of original gloss
        word_set = self.gloss_set(orig_gloss)
        with open(f"{self.config['raw_data_location']}/translation_full_set/glosses.train", "r", encoding="utf-8") as gloss_train:
            # create groundtruth glosses
            gt_glosses = [gt_gloss.strip() for gt_gloss in gloss_train.readlines()]
            # create groundtruth word sets
            gt_gloss_sets = [set(gt_gloss.split(" ")) for gt_gloss in gt_glosses]
            # find index with highest intersection
            intersections = [gt_set.intersection(word_set) for gt_set in gt_gloss_sets]
            return max(range(len(intersections)), key=lambda i: len(intersections[i]))

    def add_text(self, index, gloss):
        """Add natural language text to the data_dict according to the matched gloss.
        If the config has enabled gpt_subs, we call the openAI API to fill in placeholders of the original text.

        Args:
            index (int): Index of the text to add
            gloss (str): The gloss string used for text modification by the openAI API

        Returns:
            str: The retrieved and potentially modified text.
        """        
        with open(f"{self.config['raw_data_location']}/translation_full_set/german.train", "r", encoding="utf-8") as german_train:
            german_texts = [line.strip() for line in german_train.readlines()]
        retrieved_text = german_texts[index]

        # Check if the text contains placeholders, otherwise return string as it is
        if "$" in retrieved_text and self.config['gpt_subs']:
            # Use the API client to complete the text
            completion = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "user",
                        "content": (
                            f"You will receive two strings. The first string contains glosses from sign language. "
                            f"The second string is a natural language sentence with some placeholders. "
                            f"Fill in the placeholders according to the gloss of the first string. Just give me the final result and nothing else. "
                            f"First string: {gloss} "
                            f"Second string: {retrieved_text}"
                        )
                    }
                ],
                max_tokens=50,
            )
            retrieved_text = completion.choices[0].message.content.strip()
        return retrieved_text

class BaselineDataProcessor(DataProcessor):
    def __init__(self, config, pil_image_dict):
        super().__init__(config)
        self.pil_image_dict = pil_image_dict

    def create_data(self):
        """Create baseline data.
        """     
        root = ET.parse(f"{self.config['raw_data_location']}/20110111-annotated-groundtruth.xml").getroot()
        root_all = ET.parse(f"{self.config['raw_data_location']}/rwth-phoenix-full-20120323.corpus").getroot()
        for video in tqdm(root.findall("video"), desc="Creating baseline data"):
            video_dict = {}
            video_name = video.get("name")
            video_dict["name"] = video_name
            video_dict["sign"] = self.create_video_tensor(self.pil_image_dict[video_name])

            for recording in root_all.findall("recording"):
                if recording.get("name") == video_name:
                    segment = recording[0]
                    video_dict["signer"] = segment[0].get("name")
                    video_dict["gloss"] = segment.find("orth").text.strip()

            if self.config['gpt_full']:
                video_dict["text"] = self.generate_GPT_text(video_dict["gloss"])
            else:
                index = self.match_gloss(video_dict["gloss"])
                video_dict["text"] = self.add_text(index, video_dict["gloss"])

            self.data_dicts.append(video_dict)
    
    def create_video_tensor(self, pil_images_list):
        """Create a tensor representation of a video from a list of PIL images.

        Args:
            pil_images_list (list): List of PIL Image objects representing video frames.

        Returns:
            torch.Tensor: A tensor representation of the video.
        """        
        image_transform_tensor = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor()
        ])
        
        # concat RGB
        images_tensor = torch.cat([image_transform_tensor(image) for image in pil_images_list], dim=2)
        # num of rows in final tensor
        num_images = len(pil_images_list)
        reshaped_tensor = images_tensor.view(3, 210, num_images, 260).permute(2, 0, 1, 3)
        flattened_tensor = reshaped_tensor.reshape(num_images, -1)
        return torch.nn.functional.interpolate(flattened_tensor.unsqueeze(1), size=(1024,)).squeeze(1)

    def generate_GPT_text(self, gloss):
        """Generate a German natural sentence translation directly from sign language glosses.
        With this method we do not have to match glosses, but create completley new strings.
        Can be used by setting gpt_full in the config to True.

        Args:
            gloss (str): The sign language gloss to translate.

        Returns:
            str: The generated German natural sentence translation.
        """       
        completion = self.openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"You will receive glosses from sign language from weather forecasts."
                        f"Your job is to translate the glosses to natural language German sentences."
                        f"Only reply with the German sentence translation: "
                        f"{gloss}"
                    )
                }
            ],
            max_tokens=60,
        )
        retrieved_text = completion.choices[0].message.content.strip()
        return retrieved_text

class BodypartDataProcessor(DataProcessor):
    def __init__(self, config, pil_image_dict, whole_data_file, combination_data_file = None):
        super().__init__(config)
        self.pil_image_dict = pil_image_dict
        self.whole_data_dict = self.load_whole_data(whole_data_file)
        if combination_data_file:
            self.combination_data_dict = self.load_combination_data(combination_data_file, whole_data_file)

    def load_whole_data(self, whole_data_file):
        """Load and process the baseline data from a pickle file into a dictionary.

        Args:
            whole_data_file (str): Path to the file containing the baseline pickle file.

        Returns:
            dict: The data_dict from the baseline pickle file.
        """        
        whole_data = self.load_data(whole_data_file)
        return {video["name"]: video for video in whole_data}

    def load_combination_data(self, combination_data_file, whole_data_file):
        """Load combination data from a file or use whole data if files are the same.

        Args:
            combination_data_file (str): Path to the combination data file.
            whole_data_file (str): Path to the whole data file.

        Returns:
            dict: A dictionary where keys are video names and values are video data dictionaries.
        """         
        if combination_data_file == whole_data_file:
            return self.whole_data_dict
        else:
            combination_data = self.load_data(combination_data_file)
            return {video["name"]: video for video in combination_data}

    def create_data(self):
        """Create data with added bodyparts focus
        """        
        root = ET.parse(f"{self.config['raw_data_location']}/20110111-annotated-groundtruth.xml").getroot()
        root_all = ET.parse(f"{self.config['raw_data_location']}/rwth-phoenix-full-20120323.corpus").getroot()

        for video in tqdm(root.findall("video"), desc="Creating bodypart data"):
            video_dict = {}
            video_name = video.get("name")
            video_dict["name"] = video_name
            video_dict["sign"] = self.create_bodypart_video_tensor(video_name)

            for recording in root_all.findall("recording"):
                if recording.get("name") == video_name:
                    segment = recording[0]
                    video_dict["signer"] = segment[0].get("name")
                    video_dict["gloss"] = segment.find("orth").text.strip()

            video_dict["text"] = self.whole_data_dict[video_name]["text"]

            self.data_dicts.append(video_dict)

    def create_bodypart_video_tensor(self, video_name):
        """Takes the tensor representation of the baseline video and adds tensors focusing on specific
        bodyparts (mouth, hands)

        Args:
            video_name (str): Name of the video to process.

        Returns:
            torch.Tensor: A tensor representation of the video with added body part focus.
        """        
        # find images for given video
        pil_images_list = self.pil_image_dict[video_name]
        image_transform_tensor = torchvision.transforms.Compose([
            torchvision.transforms.Resize(size=(210, 260)),
            torchvision.transforms.ToTensor()
        ])

        images_tensor = torch.cat([image_transform_tensor(image) for image in pil_images_list], dim=2)
        num_images = len(pil_images_list)
        reshaped_tensor = images_tensor.view(3, 210, num_images, 260).permute(2, 0, 1, 3)
        flattened_tensor = reshaped_tensor.reshape(num_images, -1)
        final_tensor = torch.nn.functional.interpolate(flattened_tensor.unsqueeze(1), size=(1024,)).squeeze(1)
        if self.combination_data_dict:
            combination_tensor = self.combination_data_dict[video_name]["sign"]
            
            # combine the tensors with elementwise multiplication
            # if the shapes are not aligned, trim the larger one
            # if the combination data is data.pickle, these tensors are always 1 value larger than the bodypart vector
            final_shape = final_tensor.shape[0]
            combination_shape = combination_tensor.shape[0]
            diff = abs(final_shape - combination_shape)

            if diff != 0:
                if final_shape > combination_shape:
                    final_tensor = torch.narrow(final_tensor, 0, 0, combination_shape)
                else:
                    combination_tensor = torch.narrow(combination_tensor, 0, 0, final_shape)
        
        return torch.mul(final_tensor, combination_tensor)

class AugmentedDataProcessor(DataProcessor):
    def __init__(self, config, data_file):
        super().__init__(config)
        self.data_dicts = []
        self.orig_data_dicts = self.load_data(data_file)

    def augment_data(self):
        """Create new images by flipping and changing the contrast in the images.
        """        
        if self.data_dicts == []:
            # Transformation methods
            transform_flip = A.Compose([
                A.HorizontalFlip(),
            ])
            transform_contrast = A.Compose([
                A.RandomBrightnessContrast(),
            ])

            for video in self.orig_data_dicts:
                image = np.array(video["sign"])
                flip_video = video.copy()
                flip_video["name"] += "_flip"
                flip_video["sign"] = torch.from_numpy(transform_flip(image=image)["image"])

                contrast_video = video.copy()
                contrast_video["name"] += "_contrast"
                contrast_video["sign"] = torch.from_numpy(transform_contrast(image=image)["image"])

                self.data_dicts.append(video)
                self.data_dicts.append(flip_video)
                self.data_dicts.append(contrast_video)

            random.shuffle(self.data_dicts)

            if len(self.data_dicts) != len(self.orig_data_dicts)*3:
                print(f"ERROR: Augmentation did not work. The new data_dict contains {len(self.data_dicts)} items but it should contain {len(self.orig_data_dicts)*3} items.")
            else:
                print("Length of the new dataset:", len(self.data_dicts))
        else:
            print("No augmentation needed. Data has already been augmented.")