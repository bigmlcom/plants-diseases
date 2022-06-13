""" Convert PlantDoc dataset to a format accepted by BigML

The script assumes it is executed from the same folder that
contains the orignal dataset repository:
https://github.com/pratikkayal/PlantDoc-Object-Detection-Dataset

"""

import shutil
import json
from  pathlib import Path
from collections import Counter
import csv
import random
import string


def analyze_classes(labels: Path):
    with labels.open("r") as f:
        reader = csv.DictReader(f)
        classes = [row["class"] for row in reader]
    print(Counter(classes))

def _gen_dataset_folder(name: str):
    path = Path(name)
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(exist_ok=True)
    return path

def _hash():
    available_chars= string.hexdigits[:16]
    return ''.join(
        random.choice(available_chars)
        for dummy in range(16))

def _ensure_img(folder: Path, copied_files: dict, original_name: str, output_folder: Path):
    if original_name in copied_files:
        return copied_files[original_name]
    src = folder / original_name
    if not src.exists():
        return False
    new_name = f"{_hash()}{Path(original_name).suffix}"
    copied_files[original_name] = new_name
    shutil.copyfile(src, output_folder / new_name)
    return new_name

def _add_label(output_labels: dict, new_file_name: str, row: dict):
    if new_file_name not in output_labels:
        output_labels[new_file_name] = []
    output_labels[new_file_name].append(
        {
            "label": row["class"],
            "xmin": row["xmin"],
            "xmax": row["xmax"],
            "ymin": row["ymin"],
            "ymax": row["ymax"],

        }
    )

def _store_new_labels(output_labels: dict, output_folder: Path):
    output = []
    for fname, boxes in output_labels.items():
        output.append({"file": fname, "boxes": boxes})        
    with open(output_folder / "labels.json", 'w') as f:
         json.dump(output, f)

def gen_dataset(folder: Path, labels: Path, dataset_name: str, classes: list[str]):
    output_folder = _gen_dataset_folder(dataset_name)
    copied_files: dict = {}
    output_labels: dict = {}
    with labels.open("r") as f:
        reader = csv.DictReader(f)
        for label in reader:
            if label["class"] in classes or len(classes)==0:
                dst = _ensure_img(folder, copied_files, label["filename"], output_folder)
                if dst:
                    _add_label(output_labels, dst, label)
        _store_new_labels(output_labels, output_folder)
    
                

plantdoc_healthy = [
    "Blueberry leaf", "Peach leaf", "Raspberry leaf", "Strawberry leaf",
    "Tomato leaf", "Bell_pepper leaf", "Soyabean leaf", "Apple leaf",
    "Cherry leaf", "grape leaf", "Potato leaf"
]

plantdoc_healthy_5 = [
    "Blueberry leaf", "Peach leaf", "Raspberry leaf", "Strawberry leaf", "Tomato leaf",
]

plantdoc_10 =  [
    "Blueberry leaf", "Tomato leaf yellow virus", "Peach leaf", "Raspberry leaf",
    "Strawberry leaf", "Tomato Septoria leaf spot", "Tomato leaf", "Corn leaf blight",
    "Potato leaf early blight", "Bell_pepper leaf"
]

plantdoc = []

plantdoc_tomato = [
    "Tomato leaf yellow virus", "Tomato Septoria leaf spot", "Tomato leaf", "Tomato mold leaf",
    "Tomato leaf bacterial spot", "Tomato leaf mosaic virus", "Tomato leaf late blight",
    "Tomato Early blight leaf", "Tomato two spotted spider mites leaf"
]


if __name__ == "__main__":
    analyze_classes(Path("train_labels.csv"))                     
    gen_dataset(Path("TRAIN"), Path("train_labels.csv"), "plantdoc-healthy-10-train", plantdoc_healthy_10)
