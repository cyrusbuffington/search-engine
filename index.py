import os
import json

def process_json_files(folder_path):
    'Yields data of each JSON file in folder_path'
    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            if filename.endswith('.json'):
                with open(os.path.join(root, filename), 'r') as f:
                    data = json.load(f)
                    yield data

if __name__ == '__main__':
    page_count = 0
    for page in process_json_files('ANALYST/'):
        page_count += 1
        print(page['url'])
    print(page_count)
