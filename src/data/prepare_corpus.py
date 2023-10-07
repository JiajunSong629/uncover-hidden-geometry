import os
import numpy as np
from src.data.utils import download_dataset, tokenize

dataset = "github"
streaming_samples = 1000
overwrite = False  # overwrite the dataset if exists
logging = True
exec(open("configurator.py").read())  # overrides from command line or config file
# ---------------------------------------------------------------------------------

if __name__ == "__main__":
    data_dir = os.path.join(os.path.dirname(__file__), dataset)
    os.makedirs(data_dir, exist_ok=True)
    input_file_path = os.path.join(data_dir, dataset)

    # -------------------------------------------------------------
    if not os.path.exists(input_file_path) or overwrite:
        download_dataset(dataset, streaming_samples, input_file_path)

    with open(input_file_path, "r", encoding="utf-8") as f:
        data = f.read()

    tokenized_data = tokenize(data, data_dir)
    for model_name in tokenized_data:
        ids = tokenized_data[model_name]
        n = len(ids)
        train_ids = ids[: int(n * 0.9)]
        val_ids = ids[int(n * 0.9) :]

        # export to bin files
        train_ids = np.array(train_ids, dtype=np.uint16)
        val_ids = np.array(val_ids, dtype=np.uint16)
        train_ids.tofile(os.path.join(data_dir, f"train_{model_name}.bin"))
        val_ids.tofile(os.path.join(data_dir, f"val_{model_name}.bin"))
