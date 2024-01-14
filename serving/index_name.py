import json

class_names = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

index_to_name = {index: name for index, name in enumerate(class_names)}

with open("index_to_name.json", "w") as f:
    json.dump(index_to_name, f)