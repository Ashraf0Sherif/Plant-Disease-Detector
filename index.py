import os
import shutil
import random
from pathlib import Path

source_dir = Path("C:/Users/AshrafSherifMahmoudB/plantvillage/PlantVillage")
output_base = Path("dataset")

split_ratios = {'train': 0.7, 'val': 0.15, 'test': 0.15}

for split in split_ratios:
    for category in os.listdir(source_dir):
        Path(output_base / split / category).mkdir(parents=True, exist_ok=True)

for category in os.listdir(source_dir):
    files = list((source_dir / category).glob("*.jpg"))
    random.shuffle(files)

    train_end = int(len(files) * split_ratios['train'])
    val_end = train_end + int(len(files) * split_ratios['val'])

    splits = {
        'train': files[:train_end],
        'val': files[train_end:val_end],
        'test': files[val_end:]
    }

    for split_name, split_files in splits.items():
        for f in split_files:
            shutil.copy(f, output_base / split_name / category / f.name)

print("Classification Done") 
