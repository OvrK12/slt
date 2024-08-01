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

def process_bodypart_data(config, bodypart, combination_data, whole_data, customFilename = None):   
    """Create datasets for the different bodyparts (both hands, mouth, individual hands) combined with the base images (baseline)
    
    Args: 
        Configuration
        bodypart: folder name which contains the extracted body part images
        combination_data: filename of the dataset you want to create combined data for. Can be either baseline dataset or other bodypart (e.g. to create mouth + handas + whole)
        whole_data: filename of the baseline dataset for which you want to create bodypart data
        customFilename: give a custom name to the resulting pickle file. Otherwise name will be generated from the bodypart folder name

    Returns: Filenames of the created datasets
    """
    bodypart_pil_images = create_pil_images(f"{config['raw_data_location']}/tracking-groundtruth-sequences-bodyparts/{bodypart}")
    processor = BodypartDataProcessor(config, bodypart_pil_images, 
                                    f"{config['preprocessed_data_location']}/{whole_data}", 
                                    f"{config['preprocessed_data_location']}/{combination_data}")
    processor.create_data()

    filename = f"{customFilename}{generate_file_affix(config)}.pickle" if customFilename else f"{bodypart}{generate_file_affix(config)}.pickle"
    processor.dump_data(f"{config['preprocessed_data_location']}/{filename}")
    if not config['augmented']:
        processor.split_data(f"{config['preprocessed_data_location']}/{filename}")
    return filename

def process_augmented_data(config, filename):
    """ Augment existing datasets

    Args: Filename of the file that should be augmented
    """
    processor = AugmentedDataProcessor(config,f"{config['preprocessed_data_location']}/{filename}")
    processor.augment_data()
    processor.dump_data(f"{config['preprocessed_data_location']}/{filename}")
    processor.split_data(f"{config['preprocessed_data_location']}/{filename}")


if __name__ == "__main__":
    config = None
    
    with open(CONFIG_LOCATION, "r", encoding="utf-8") as ymlfile:
        config = yaml.safe_load(ymlfile)
    # generate baseline data without bodyparts
    baseline_data = process_baseline_data(config)
    # generate data with extracted bodyparts
    bodypart_filenames = []
    bodyparts = os.listdir(f"{config['raw_data_location']}/tracking-groundtruth-sequences-bodyparts")
    for curr_bodypart in bodyparts:
        bodypart_filenames.append(process_bodypart_data(config, curr_bodypart, baseline_data, baseline_data))
    
    # generate mouth + hands + whole image dataset
    mouth_hands_whole = process_bodypart_data(config, "both_hands", baseline_data, f"mouth{generate_file_affix(config)}.pickle", customFilename="mouth_hands_whole")
    bodypart_filenames.append(mouth_hands_whole)

    # augment previously created dataset
    if config['augmented']:
        process_augmented_data(config, baseline_data)
        for curr_bodypart in bodypart_filenames:
            process_augmented_data(config, curr_bodypart)