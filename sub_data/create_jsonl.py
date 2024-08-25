import csv
import json

# Path to the CSV file
csv_file_path = 'sub_data/path_caption_score_new.csv'

# The output JSON file
jsonl_file_path = 'sub_data/train/metadata.jsonl'

# Read the CSV and add data to a dictionary
data_list = []
with open(csv_file_path, mode='r', encoding='utf-8') as csv_file:
    csv_reader = csv.DictReader(csv_file)
    for row in csv_reader:
        # Assuming the image file name can be extracted from the 'Img_path' column
        # file_name = row['Img_path'].split('/')[-1].replace('.jpg', '.png')
        file_name = row['Img_path'].split('/')[-1]
        caption = row[' Caption']
        score = row[' Score']
        # Create a new dictionary for each row
        data_dict = {
            "file_name": file_name,
            "text": caption,
            'score': score,
        }
        data_list.append(data_dict)

# Write the dictionary to a JSON file, one dictionary per line
with open(jsonl_file_path, mode='w', encoding='utf-8') as json_file:
    for json_object in data_list:
        json_file.write(json.dumps(json_object) + "\n")
