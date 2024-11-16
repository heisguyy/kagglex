"""Script to prepare data for training."""

import logging
import os
import random
import shutil

import kagglehub
import pandas as pd
from datasets import load_dataset

# pylint: disable=broad-exception-caught, invalid-name

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

DESTINATION_FOLDER = "data"
os.makedirs(DESTINATION_FOLDER)

logger.info("Starting data preparation process")

# Load captions data
logger.info("Loading caption data from sources")
try:
    afrimmd = pd.read_parquet(
        "hf://datasets/AfriMM/AfriMMD/data/train-00000-of-00001.parquet"
    )
    aviladata = pd.read_csv(
        "hf://datasets/sartifyllc/AViLaData/"
        "all-final-v1-african-languages-captions.csv"
    )
except Exception as e:
    logger.error("Error loading data: %s", str(e))
    raise

# Resturcture the afrimmd data
logger.info("Restructuring AfriMMD data")
lang_cols = [col for col in afrimmd.columns if col not in ["id", "image_id"]]
afrimmd = pd.melt(
    afrimmd,
    id_vars=["id", "image_id"],
    value_vars=lang_cols,
    var_name="language",
    value_name="caption",
)
afrimmd.image_id = afrimmd.image_id.apply(lambda x: f"{x.split('_')[0]}.jpg")

# Drop unnecessary columns and ensure common column names with afrimmd
logger.info("Processing AViLa data")
aviladata.drop(columns=["Unnamed: 0", "caption_number"], inplace=True)
aviladata.rename(columns={"image": "image_id"}, inplace=True)
aviladata = aviladata.reindex(columns=["id", "image_id", "language", "caption"])

# Concatenate the two dataframes
logger.info("Concatenating datasets")
aviladata.id = aviladata.id + afrimmd.id.max() + 1
data = pd.concat([afrimmd, aviladata]).reset_index(drop=True)
data.rename(columns={"image_id": "file_name"}, inplace=True)

# Drop duplicate captions
logger.info("Removing duplicate captions")
initial_size = len(data)
data.drop_duplicates(subset="caption", inplace=True)
logger.info("Removed %s duplicate captions", initial_size - len(data))

language_map = {
    "eng": "en",
    "yor": "yo",
    "afr": "af",
    "amh": "am",
    "ibo": "ig",
    "hau": "ha",
    "swh": "sw",
    "sna": "sn",
    "arb": "ar",
    "fra": "fr",
    "por": "pt",
    "zul": "zu",
}
data = data[data["language"].isin(language_map.keys())]
data.reset_index(drop=True, inplace=True)
data.language = data.language.map(language_map)

question_map = {
    "en": ["caption this image", "what is a good caption for this image?"],
    "yo": ["àkọlé àwòrán yìí", "kí ni àkọlé tó dára fún àwòrán yìí?"],
    "af": [
        "onderskrif hierdie beeld",
        "Wat is 'n goeie onderskrif vir hierdie prentjie?",
    ],
    "am": ["የዚህ ምስል መግለጫ", "ለዚህ ምስል ጥሩ መግለጫ ምንድነው?"],
    "ig": ["Nkọwa foto a", "kedu ihe ga-adị mma ị ga-ede n'okpuru foto a?"],
    "ha": ["rubutun wannan hoton", "menene kyakkyawan taken wannan hoton?"],
    "sw": ["maelezo ya picha hii", "Ni maelezo gani mazuri kwa picha hii?"],
    "sn": [
        "mashoko omuzasi womufananidzo uyu",
        "Ndechipi chinyorwa chakanaka chomufananidzo uyu?",
    ],
    "ar": ["وصف هذه الصورة", "ما هو التسمية الجيدة لهذه الصورة؟"],
    "fr": [
        "la légende de cette image",
        "Quelle est la bonne légende pour cette image?",
    ],
    "pt": ["legenda esta imagem", "Qual é uma boa legenda para esta imagem?"],
    "zu": [
        "umbhalo ongaphansi kwalesi sithombe",
        "yisiphi isihlokwana esihle salesi sithombe?",
    ],
}

data["question"] = data.language.apply(lambda x: random.choice(question_map[x]))
data.drop_duplicates(subset=['id', 'language'], inplace=True)
logger.info("Final data length: %s", len(data))

logger.info("Cut data length")
data["weight"] = 1/(data["language"].nunique()-2)
data.loc[data["language"] == "ha", "weight"] = 0
data.loc[data["language"] == "af", "weight"] = 0
data_sample = data.sample(
    n=150_000,
    weights=data['weight']
)
data = pd.concat([data_sample, data[data["language"].isin(["ha", "af"])]])
logger.info("Reduced data length: %s", len(data))

# Save data to parquet
logger.info("Saving processed data to CSV")
try:
    data.to_csv("data/metadata.csv", index=False)
except Exception as e:
    logger.error("Error saving CSV file: %s", str(e))
    raise

# Download the flickr8k dataset from kaggle
logger.info("Downloading Flickr30k dataset")
try:
    source_folder = kagglehub.dataset_download(
        "adityajn105/flickr30k", force_download=True
    )
    source_folder = os.path.join(source_folder, "Images")
except Exception as e:
    logger.error("Error downloading dataset: %s", str(e))
    raise

# Move the images to the destination folder
logger.info("Moving images to destination folder")
moved_count = 0
for filename in os.listdir(source_folder):
    source_path = os.path.join(source_folder, filename)
    destination_path = os.path.join(DESTINATION_FOLDER, filename)
    if os.path.isfile(source_path):
        try:
            shutil.move(source_path, destination_path)
            moved_count += 1
        except Exception as e:
            logger.error("Error moving file %s: %s", filename, str(e))

logger.info("Successfully moved %s images", moved_count)
logger.info("Data preparation completed")

# Load your ImageFolder dataset
dataset = load_dataset('csv', data_files='data/metadata.csv', split='train')
dataset.push_to_hub("heisguyy/pt-paligemma-multilingual-imagecaptions")
logger.info("Data pushed to huggingface")
