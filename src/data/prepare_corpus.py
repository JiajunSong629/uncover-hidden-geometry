import os
import numpy as np
from src.data.utils import download_dataset, tokenize

dataset = "github"
streaming_samples = 1000
overwrite = False  # overwrite the dataset if exists
logging = True
tokenizers = "gpt2+bloom"
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

    tokenized_data = tokenize(
        tokenizers=tokenizers.split("+"),
        data=data,
        data_dir=data_dir,
    )

    log = ""
    for model_name in tokenized_data:
        ids = tokenized_data[model_name]
        n = len(ids)

        # export to bin files
        ids = np.array(ids, dtype=np.uint16)
        ids.tofile(os.path.join(data_dir, f"{model_name}.bin"))

        log += f"{model_name.upper()}: length of dataset in tokens {n}\n"

    with open(os.path.join(data_dir, "readme"), "w") as f:
        f.write(log)
