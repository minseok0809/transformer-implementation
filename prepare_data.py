import glob
from datasets import load_dataset

def load_data(data_dir):
    json_paths = glob.glob(data_dir + "/*.json")
    for json_path in json_paths:
        if 'train' in json_path: train_data = {"train": json_path.split("/")[-1]}
        elif 'validation' in json_path: validation_data = {"validation": json_path.split("/")[-1]}
        elif 'test' in json_path: test_data = {"test": json_path.split("/")[-1]}

    train_dataset = load_dataset(data_dir, data_files=train_data)
    validation_dataset = load_dataset(data_dir, data_files=validation_data)
    test_dataset = load_dataset(data_dir, data_files=test_data)

    return train_dataset, validation_dataset, test_dataset 
    
