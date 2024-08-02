import os
from tqdm import tqdm
import xml.etree.ElementTree as ET
from PIL import Image

RAW_DATA_LOCATION = './raw'

def get_bodyparts_coordinates(annotations_file):
    """Get coordinates of bodyparts for all videos from the provided xml annotations file

    Args:
        annotations_file (str): path to the xml annotations file

    Returns:
        dict: A nested dictionary structure where:
            - The top level keys are video names.
            - The second level keys are frame numbers.
            - The third level keys are body part names ("left_hand", "right_hand", "nose").
            - The values are tuples of (x, y) coordinates for each body part.

            Example:
            {
                'video1': {
                    '0': {
                        'left_hand': (100, 200),
                        'right_hand': (300, 400),
                        'nose': (250, 150)
                    },
                ...
            }
    """ 
    root = ET.parse(annotations_file).getroot()
    coordinates = {}

    for video in root.findall("video"):
        video_name = video.get("name")
        frames = {}

        for frame in video:
            frame_num = frame.get("number")
            frame_coords = {}

            for point in frame:
                point_type = {
                    "0": "left_hand",
                    "1": "right_hand",
                    "2": "nose"
                }.get(point.get("n"))
                
                if point_type:
                    frame_coords[point_type] = (int(point.get("x")), int(point.get("y")))

            frames[frame_num] = frame_coords

        coordinates[video_name] = frames
    
    return coordinates

def img_crop(img, bodypart, *coords):
    """
    Crop image based on bodypart and coordinates.
    
    Args:
        img (str): Path to the PIL image file.
        bodypart (str): Name of the body part (left hand, right hand, mouth). This will affect the cropped area of the image
        *coords: Bodypart coordinate tuples. Can contain 2 tuples for both hands input.
    
    Returns:
        PIL.Image: Cropped image.
    """
    if type(img) == str:
        img = Image.open(img)

    setoffs = {
        'left_hand': {'x1': -40, 'y1': -40, 'x2': +30, 'y2': +30},
        'right_hand': {'x1': -40, 'y1': -40, 'x2': +30, 'y2': +30},
        'mouth': {'x1': -30, 'y1': 0, 'x2': +30, 'y2': +30}
    }

    # determine coordinates of crop window
    if len(coords) == 1:
        center1_x, center1_y = coords[0]
        setoff = setoffs[bodypart]

        xmin=min(center1_x + setoff['x1'], center1_x + setoff['x2'])
        xmax=max(center1_x + setoff['x1'], center1_x + setoff['x2'])
        ymin=min(center1_y + setoff['y1'], center1_y + setoff['y2'])
        ymax=max(center1_y + setoff['y1'], center1_y + setoff['y2'])

    elif len(coords) == 2: # for both_hands input
        center1_x, center1_y = coords[0]
        center2_x, center2_y = coords[1]
        setoff = {'x1': setoffs['right_hand']['x1'], 'y1': setoffs['right_hand']['y1'], 'x2': setoffs['left_hand']['x2'], 'y2': setoffs['left_hand']['y2']}

        xmin=min(center1_x + setoff['x1'], center1_x + setoff['x2'], center2_x + setoff['x1'], center2_x + setoff['x2'])
        xmax=max(center1_x + setoff['x1'], center1_x + setoff['x2'], center2_x + setoff['x1'], center2_x + setoff['x2'])
        ymin=min(center1_y + setoff['y1'], center1_y + setoff['y2'], center2_y + setoff['y1'], center2_y + setoff['y2'])
        ymax=max(center1_y + setoff['y1'], center1_y + setoff['y2'], center2_y + setoff['y1'], center2_y + setoff['y2'])
    else:
        print(f"Expected 1 or 2 coordinate tuples, got {len(coords)}")

    box = (xmin, ymin, xmax, ymax)

    return img.crop(box)

def process_videos(data_path, target_path, coordinates):
    """Extract bodypart PIL images for all videos and frames in the dataset.

    Args:
        data_path (str): Path to the original videos.
        target_path (str): Path where the images should be extracted to.
        coordinates (dict): Dictionary which contains the coordinates of the bodyparts.
    """    
    bodyparts = ["right_hand", "left_hand", "both_hands", "mouth"]

    for video_name in tqdm(os.listdir(data_path), desc="Extracting bodyparts from videos..."):
        video_path = os.path.join(data_path, video_name)
        video_dict = coordinates[video_name]

        for bodypart in bodyparts:
            os.makedirs(os.path.join(target_path, bodypart, video_name), exist_ok=True)

        # Process frames
        frame_count = 0
        for img_name in sorted(os.listdir(video_path)):
            if img_name == ".history.forster":
                frame_count = 0
                continue

            frame_dict = video_dict[str(frame_count)]
            img_path = os.path.join(video_path, img_name)

            # Crop and save images for each bodypart
            crop_configs = [
                ("right_hand", frame_dict["right_hand"]),
                ("left_hand", frame_dict["left_hand"]),
                ("both_hands", frame_dict["right_hand"], frame_dict["left_hand"]),
                ("mouth", frame_dict["nose"])
            ]

            for bodypart, *coords in crop_configs:
                cropped_img = img_crop(img_path, bodypart, *coords)
                target_img_name = f"{os.path.splitext(img_name)[0]}_{bodypart.replace('_', '-')}{os.path.splitext(img_name)[1]}"
                target_img_path = os.path.join(target_path, bodypart, video_name, target_img_name)
                cropped_img.save(target_img_path)

            frame_count += 1

if __name__ == "__main__":
    annotations_file = f"{RAW_DATA_LOCATION}/20110111-annotated-groundtruth.xml"
    data_path = f"{RAW_DATA_LOCATION}/tracking-groundtruth-sequences"
    target_path = f"{RAW_DATA_LOCATION}/tracking-groundtruth-sequences-bodyparts"

    coordinates = get_bodyparts_coordinates(annotations_file)
    os.makedirs(target_path, exist_ok=True)
    process_videos(data_path, target_path, coordinates)
