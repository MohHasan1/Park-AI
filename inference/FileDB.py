import os
import json


class FileDB:
    def __init__(self, filepath="./inference/filedb.json"):
        self.filepath = filepath      # Path to the JSON file
        self.data = None              # Dictionary holding key-value pairs
        self._load()                  # Load data from file into self.data

    def _load(self):
        if os.path.exists(self.filepath):
            with open(self.filepath, "r") as file:
                try:
                    self.data = json.load(file)
                except json.JSONDecodeError:
                    self.data = {}
        else:
            self.data = {}

    def _save(self):
        with open(self.filepath, "w") as file:
            json.dump(self.data, file, indent=4)

    def create(self, key, value):
        if key in self.data:
            return
        self.data[key] = value
        self._save()

    def read(self, key):
        return self.data.get(key, None)

    def update(self, key, value):
        if key not in self.data:
            self.create(key, value)
            return
        self.data[key] = value
        self._save()

    def delete(self, key):
        if key in self.data:
            del self.data[key]
            self._save()
        else:
            raise KeyError(f"Key '{key}' not found.")

    def get_all(self):
        return self.data
