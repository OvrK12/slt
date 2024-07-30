import yaml
import os
from data_processor import create_pil_images, BaselineDataProcessor, BodypartDataProcessor, AugmentedDataProcessor

CONFIG_LOCATION = './preprocess_config.yaml'

def generate_file_affix(config):
    """ Generate the file affix.

    Args: Configuration

    Returns: Affix from the configuration file.
    """
    affix = ""
    if config['gpt_subs']:
        affix += "_gptsub"
    if config['gpt_full']:
        affix += "_gptfull"
    if config['bleu']:
        affix += "_bleu"
    if config['augmented']:
        affix += "_aug"

    return affix

def process_baseline_data(config):   
    """Create baseline data

    Args: Configuration

    Returns: Filename of the created dataset.
    """
    pil_image_dict = create_pil_images(f"{config['raw_data_location']}/tracking-groundtruth-sequences")
    processor = BaselineDataProcessor(config, pil_image_dict)
    processor.create_data()
    filename = f"baseline{generate_file_affix(config)}.pickle"
    processor.dump_data(f"{config['preprocessed_data_location']}/{filename}")
    processor.split_data(f"{config['preprocessed_data_location']}/{filename}")
    return filename

def process_bodypart_data(config, baseline_data):   
    """Create datasets for the different bodyparts (both hands, mouth, individual hands) combined with the base images (baseline)
    
    Args: 
        Configuration
        Filename of the baseline dataset

    Returns: Filenames of the created datasets
    """
    filenames = []
    bodyparts = os.listdir(f"{config['raw_data_location']}/tracking-groundtruth-sequences-bodyparts")
    for curr_bodypart in bodyparts:
        bodypart_pil_images = create_pil_images(f"{config['raw_data_location']}/tracking-groundtruth-sequences-bodyparts/{curr_bodypart}")
        processor = BodypartDataProcessor(config, bodypart_pil_images, 
                                        f"{config['preprocessed_data_location']}/{baseline_data}", 
                                        f"{config['preprocessed_data_location']}/{baseline_data}")
        processor.create_data()
        filename = f"{curr_bodypart}{generate_file_affix(config)}.pickle"
        filenames.append(filename)
        processor.dump_data(f"{config['preprocessed_data_location']}/{filename}")
        if not config['augmented']:
            processor.split_data(f"{config['preprocessed_data_location']}/{filename}")
    return filenames

def process_augmented_data(filenames):
    """ Augment existing datasets

    Args: Filenames of the files that should be augmented
    """
    for filename in filenames:
        processor = AugmentedDataProcessor(config,f"{config['preprocessed_data_location']}/{filename}")
        processor.augment_data()
        processor.dump_data(f"{config['preprocessed_data_location']}/{filename}")
        processor.split_data(f"{config['preprocessed_data_location']}/{filename}")

    

if __name__ == "__main__":
    config = None
    
    with open(CONFIG_LOCATION, "r", encoding="utf-8") as ymlfile:
        config = yaml.safe_load(ymlfile)
    baseline_data = process_baseline_data(config)
    filenames = process_bodypart_data(config, baseline_data)
    if config['augmented']:
        process_augmented_data(filenames)