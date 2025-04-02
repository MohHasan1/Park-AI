import sys
import os

# Add the parent directory (project root) to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from inference.FileDB import FileDB

db = FileDB("./test/mydata.json")

# Create
db.create("username", "nazia")
db.create("age", 23)

# Read
print(db.read("username")) 

# Update
db.update("age", 24)

# Delete
# db.delete("username")

# Get All
print(db.get_all())  
