import glob
import os
import json
import csv
import sys
csv.field_size_limit(sys.maxsize)
import random

# Aria
aria_metadata_filepath = "/data/scratch/acw769/improvnet/aria-midi-v1-pruned-ext/metadata.json"
with open(aria_metadata_filepath, "r") as f:
    aria_metadata = json.load(f)

# Get all .mid files within multiple directories
aria_midi_files = glob.glob("/data/scratch/acw769/improvnet/aria-midi-v1-pruned-ext/data/**/*.mid", recursive=True)
print(f"Found {len(aria_midi_files)} ARIA MIDI files.")

aria_data = []
for midi_file in aria_midi_files:
    filename = os.path.basename(midi_file).split("_")[0]
    if filename in aria_metadata:
        entry = {
            "midi_filepath": midi_file,
            "genre": aria_metadata[filename].get('metadata', {}).get('genre', None).lower() if aria_metadata[filename].get('metadata', {}).get('genre', None) else None,
            "composer": aria_metadata[filename].get('metadata', {}).get('composer', None).lower() if aria_metadata[filename].get('metadata', {}).get('composer', None) else None,
            "form": aria_metadata[filename].get('metadata', {}).get('form', None).lower() if aria_metadata[filename].get('metadata', {}).get('form', None) else None,
            "musical_period": aria_metadata[filename].get('metadata', {}).get('musical_period', None).lower() if aria_metadata[filename].get('metadata', {}).get('musical_period', None) else None,
        }
        aria_data.append(entry)

# Maestro
maestro_metadata_filepath = "/data/scratch/acw769/improvnet/maestro-v3.0.0-midi/maestro-v3.0.0/maestro-v3.0.0.csv"

with open(maestro_metadata_filepath, "r", newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    maestro_metadata = {row['midi_filename']: row for row in reader}    

maestro_data = []
for key, value in maestro_metadata.items():
    midi_filepath = os.path.join("/data/improvnet/maestro-v3.0.0/", value['midi_filename'])
    forms = ['sonata', 'etude', 'waltz', 'nocturne', 'prelude', 'fugue', 'suite', 'ballade', 'mazurka', 'polonaise', 'scherzo', 'fantasy', 'fughetta', 'impromptu', 'variation']
    form = None
    for f in forms:
        if f.lower() in value.get('canonical_title', '').lower():
            form = f.lower()
            break
    entry = {
        "midi_filepath": midi_filepath,
        "genre": "classical",
        "composer": value.get('canonical_composer', None).lower(),
        "form": form,
        "musical_period": None,
    }
    maestro_data.append(entry)

# PiJAMA
# Get all .mid files within multiple directories
pijama_midi_files = glob.glob("/data/scratch/acw769/improvnet/pijama-retranscribed/data/**/*.mid", recursive=True)
print(f"Found {len(pijama_midi_files)} PiJAMA MIDI files.")

pijama_data = []
for midi_file in pijama_midi_files:
    filename = os.path.basename(midi_file)
    entry = {
        "midi_filepath": midi_file,
        "genre": "jazz",
        "composer": None,
        "form": None,
        "musical_period": None,
    }
    pijama_data.append(entry)

# Get all .mid files within multiple directories
doug_midi_files = glob.glob("/data/scratch/acw769/improvnet/doug_mcenzie_jazz/**/*.mid", recursive=True)
print(f"Found {len(doug_midi_files)} Doug McKenzie MIDI files.")

doug_data = []
for midi_file in doug_midi_files:
    filename = os.path.basename(midi_file)
    entry = {
        "midi_filepath": midi_file,
        "genre": "jazz",
        "composer": None,
        "form": None,
        "musical_period": None,
    }
    doug_data.append(entry)

# SymphonyNet
# Get all .mid files within multiple directories
symphony_files = glob.glob("/data/scratch/acw769/improvnet/SymphonyNet_Dataset/**/*.mid", recursive=True)
print(f"Found {len(symphony_files)} SymphonyNet MIDI files.")

symphony_data = []
for midi_file in symphony_files:
    filename = os.path.basename(midi_file)
    if "classical" in midi_file:
        entry = {
            "midi_filepath": midi_file,
            "genre": "classical",
            "composer": None,
            "form": None,
            "musical_period": None,
        }
    else:
        entry = {
            "midi_filepath": midi_file,
            "genre": None,
            "composer": None,
            "form": None,
            "musical_period": None,
        }
    symphony_data.append(entry)

# Classical Archives
# Get all .mid files within multiple directories
classicalarchive_files = glob.glob("/data/scratch/acw769/improvnet/ClassicalArchives-MIDI-Collection/**/*.mid", recursive=True)
print(f"Found {len(classicalarchive_files)} Classical Archives MIDI files.")

classicalarchive_data = []
for midi_file in classicalarchive_files:
    filename = os.path.basename(midi_file)
    entry = {
        "midi_filepath": midi_file,
        "genre": "classical",
        "composer": None,
        "form": None,
        "musical_period": None,
    }
    classicalarchive_data.append(entry)

# Combine all data
all_data = aria_data + maestro_data + pijama_data + doug_data + symphony_data + classicalarchive_data
print(f"Total number of MIDI files collected: {len(all_data)} from {len(aria_data)} (Aria) + {len(maestro_data)} (Maestro) + {len(pijama_data)} (Pijama) + {len(doug_data)} (Doug McKenzie) + {len(symphony_data)} (SymphonyNet) + {len(classicalarchive_data)} (ClassicalArchive)")

# Get count of all unique genre, composers and form and print them

unique_genres = set()
unique_composers = set()
unique_forms = set()
for entry in all_data:
    if entry['genre']:
        unique_genres.add(entry['genre'])
    if entry['composer']:
        unique_composers.add(entry['composer'])
    if entry['form']:
        unique_forms.add(entry['form'])

print(f"Unique genres: {unique_genres}")
print(f"Unique composers: {unique_composers}")
print(f"Unique forms: {unique_forms}")

# Create misc_data.jsonl
if not os.path.exists("improvnet/data"):
    os.makedirs("improvnet/data")

# Write all_data to a JSONL file
with open("improvnet/data/misc_data.jsonl", "w") as f:
    for entry in all_data:
        # Merge forms
        if entry['form'] == "fantasia":
            entry['form'] = "fantasy"
        
        # Train (95%), validation (2%), test split (3%)
        rand_val = random.random()
        if rand_val < 0.97:
            entry['split'] = 'train'
        elif rand_val < 0.99:
            entry['split'] = 'validation'
        else:
            entry['split'] = 'test'

        f.write(json.dumps(entry) + "\n")

### GigaMIDI ###

csv_filepath = "/data/scratch/acw769/improvnet/GigaMIDI/Final_GigaMIDI_V1.1_Final/Final-Metadata-Extended-GigaMIDI-Dataset-updated.csv"

# Read CSV file and print file_path and music_styles_curated keys
with open(csv_filepath, "r") as csvfile:
    reader = csv.DictReader(csvfile)
    # Create data entries for each row
    gigamidi_data = []
    for row in reader:
        midi_filepath = os.path.join("/data/scratch/acw769/improvnet/GigaMIDI/Final_GigaMIDI_V1.1_Final/", row['file_path'].lstrip("./Final_GigaMIDI_V1.1_Final/"))
        genres = row['music_styles_curated'].lower().split(";") if row['music_styles_curated'] else []
        genre = genres[0] if genres else None
        entry = {
            "midi_filepath": midi_filepath,
            "genre": genre,
            "composer": row['artist'].lower() if row['artist'] else None,
            "form": None,
            "musical_period": None,
        }
        gigamidi_data.append(entry)

print(f"Total number of GigaMIDI MIDI files collected: {len(gigamidi_data)}")

# Count unique genres in gigamidi_data
genre_count = {}
for entry in gigamidi_data:
    genre = entry['genre']
    if genre:
        if genre in genre_count:
            genre_count[genre] += 1
        else:
            genre_count[genre] = 1

# Print genre counts
for genre, count in genre_count.items():
    print(f"{genre}: {count}")

genres = {'rock', 'classical', 'ragtime', 'pop', 'blues', 'soundtrack', 'folk', 'atonal', 'ambient', 'jazz', 'metal', 'game'}

# Write gigamidi_data to a JSONL file and for genres other than the above, set genre to None
with open("improvnet/data/gigamidi_data.jsonl", "w") as f:
    for entry in gigamidi_data:
        if entry['genre'] not in genres:
            entry['genre'] = None
        
        # Train (95%), validation (2%), test split (3%)
        rand_val = random.random()
        if rand_val < 0.97:
            entry['split'] = 'train'
        elif rand_val < 0.99:
            entry['split'] = 'validation'
        else:
            entry['split'] = 'test'

        f.write(json.dumps(entry) + "\n")