import json
import os.path
import pandas as pd

DATA_DIR = os.path.join("datasets", "word")

df: pd.DataFrame = (
    pd.read_csv(os.path.join(DATA_DIR, "overall.csv"), usecols=("tokens", "ac"))
    .applymap(eval)
    .applymap(tuple)
)

print("Before")
print(df.info())
initial_count = len(df)

# Remove rows with same tokens and same annotations
df.drop_duplicates(inplace=True)

print("\n\nAfter")
print(df.info())

print("\n\nDuplicates", initial_count - len(df))

# Get all the tokens that are in the dataset
total_labels = set()
for ac in df["ac"].apply(set):
    total_labels |= ac
total_labels.discard("O")

label_names = list(total_labels)
# Sort by the later part first, then by first part
label_names.sort(key=lambda cat: cat.split("-", 1)[::-1])

# label_names when printed
# ['B-Others',
#  'I-Others',
#  'B-character_assasination',
#  'I-character_assasination',
#  'B-ethnic_violence',
#  'I-ethnic_violence',
#  'B-general_threat',
#  'I-general_threat',
#  'B-physical_threat',
#  'I-physical_threat',
#  'B-profanity',
#  'I-profanity',
#  'B-rape_threat',
#  'I-rape_threat',
#  'B-religion_violence',
#  'I-religion_violence',
#  'B-sexism',
#  'I-sexism']

# Insert no entity token at the top
label_names.insert(0, "O")

print(label_names)

with open(os.path.join(DATA_DIR, "label_names.json"), "w") as f:
    json.dump(label_names, f)

label2id = {l: i for i, l in enumerate(label_names)}

df["ac"] = df["ac"].apply(lambda labels: tuple(label2id[lab] for lab in labels))

df.to_csv(os.path.join(DATA_DIR, "combined.csv"), index=False)
