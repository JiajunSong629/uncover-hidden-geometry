import os
from tqdm import tqdm
import numpy as np
from src.data.prepare_corpus import tokenize

dataset = "github_topics"
overwrite = False
exec(open("configurator.py").read())  # overrides from command line or config file
# ---------------------------------------------------------------------------------
data_dir = os.path.join(os.path.dirname(__file__), dataset)
print(data_dir)

n_topics = len(
    [f for f in os.listdir(data_dir) if f.startswith("topic") and not f.endswith("bin")]
)


for i in tqdm(range(n_topics), desc="Topic progress"):
    input_file_path = os.path.join(data_dir, f"topic_{i}")
    try:
        with open(input_file_path, "r") as f:
            data = f.read()
    except Exception as e:
        raise e

    tokenized_data = tokenize(data, data_dir)
    for model_name in tokenized_data:
        ids = tokenized_data[model_name]
        ids = np.array(ids, dtype=np.uint16)
        ids.tofile(os.path.join(data_dir, f"topic_{i}_{model_name}.bin"))
