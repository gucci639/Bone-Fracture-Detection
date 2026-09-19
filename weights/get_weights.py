import os
import requests
path = "https://github.com/gucci639/Bone-Fracture-Detection/releases/download/v1.0.0/"
files = ["model_fast.pt", 'model_precise.pt']
def get_weights():
    os.makedirs("weights", exist_ok=True)
    for file in files:
        response = requests.get(path+file, timeout=30)
        response.raise_for_status()
        with open('weights/'+file, 'wb') as f:
            f.write(response.content)

get = get_weights