import re
import requests
from bs4 import BeautifulSoup
import csv
from collections import defaultdict
import pickle
import gzip
import gdown

def download_file(url, output):
    gdown.download(url, output, quiet=False)

def load_dataset_file(filename):
    with gzip.open(filename, "rb") as f:
        loaded_object = pickle.load(f)
        return loaded_object

def get_dgs_homonyms(url):
    """
    Retrieve homonyms from meine DGS.
    """
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36'}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        rows = soup.find_all('tr')
        words_with_ids = defaultdict(list)
        for row in rows:
            word_td = row.find('td')
            if word_td:
                word = word_td.get_text(strip=True).lower()
                if ')' in word:
                    word = word.split(') ')[1] if ') ' in word else word.split(')')[1]
                links = row.find_all('a')
                for link in links:
                    identifier = link.get_text(strip=True)
                    if identifier:
                        base_id = identifier.split('#')[0]
                        words_with_ids[word].append(base_id)
        return words_with_ids
    else:
        print(f"Failed to retrieve data from {url}. Status code: {response.status_code}")
        return {}

def filter_dgs_homonyms(words_with_ids):
    """
    Filter words that have multiple unique identifiers.
    """
    homonyms = {}
    for word, ids in words_with_ids.items():
        if len(ids) > 1:
            unique_ids = set(ids)
            if len(unique_ids) > 1:
                homonyms[word] = unique_ids
    return homonyms

def save_to_csv(homonyms, filename):
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Word', 'Identifiers'])
        for word, identifiers in homonyms.items():
            writer.writerow([word, ', '.join(identifiers)])
    print(f'Saved to {filename}.')

def clean_gloss(gloss):
    cleaned_gloss = re.sub(r'loc-|IX|poss-|bh-|lh-|cl-|-PLUSPLUS|neg-|negalp-', '', gloss)
    cleaned_gloss = re.sub(r'\s+', ' ', cleaned_gloss).strip()
    return cleaned_gloss

def process_gloss(gloss):
    gloss = gloss.lower()
    umlaut_replacements = {'ue': 'ü', 'ae': 'ä', 'oe': 'ö'}
    for pattern, replacement in umlaut_replacements.items():
        gloss = gloss.replace(pattern, replacement)
    gloss = clean_gloss(gloss)
    tokens = gloss.split()
    return tokens

def extract_glosses_from_dataset(dataset):
    """
    Extract glosses from the dataset and process them.
    """
    glosses_with_homonyms = []
    for section in dataset:
        gloss = section['gloss']
        processed_gloss = process_gloss(gloss)
        glosses_with_homonyms.append((gloss, processed_gloss))
    return glosses_with_homonyms

def compare_words_with_homonyms(glosses_with_homonyms, homonyms):
    """
    Compare glosses with known homonyms and find matches.
    """
    results = []
    for original_gloss, processed_gloss in glosses_with_homonyms:
        matching_homonyms = [word for word in processed_gloss if word in homonyms]
        results.append((original_gloss, matching_homonyms))
    return results

def save_matching_homonyms_to_csv(matching_homonyms, filename):
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Original Gloss', 'Homonyms'])
        for gloss, homonyms in matching_homonyms:
            writer.writerow([gloss, ', '.join(homonyms)])
    print(f'Saved to {filename}.')

if __name__ == "__main__":
    data_url = "https://drive.google.com/uc?id=1hoeU3EE1kGB35qqQM3jQdtNaGRlfsVkZ"
    data_file = "data.pickle"

    download_file(data_url, data_file)
    loaded_data = load_dataset_file(data_file)

    url = "https://www.sign-lang.uni-hamburg.de/korpusdict/overview/index-topics.html"
    homonyms_file = 'homonyms.csv'
    matching_homonyms_file = 'matching_homonyms.csv'

    words_with_ids = get_dgs_homonyms(url)
    if words_with_ids:
        homonyms = filter_dgs_homonyms(words_with_ids)
        save_to_csv(homonyms, homonyms_file)

        glosses_with_homonyms = extract_glosses_from_dataset(loaded_data)
        matching_homonyms = compare_words_with_homonyms(glosses_with_homonyms, homonyms)

        total_glosses_with_homonyms = 0

        for gloss, homonyms in matching_homonyms:
            if homonyms:
                homonyms_str = ', '.join(homonyms)
                print(f'Gloss: {gloss}, Homonyms: {homonyms_str}')
                total_glosses_with_homonyms += 1

        print(f'Total glosses with homonyms: {total_glosses_with_homonyms}')

        save_matching_homonyms_to_csv(matching_homonyms, matching_homonyms_file)
    else:
        print("No homonyms found.")