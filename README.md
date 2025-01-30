# OphthaBERT-glaucoma
This repository deploys a local API for inference of OphthaBERT for glaucoma diagnosis and subtype identification. The only pre-requisite for installation is a working updated installation of [Docker](https://docs.docker.com/engine/install/).

## Docker Installation (recommended)
Folow to build Docker image of the inference API and run the API image.
```bash
# Open a terminal (Command Prompt or PowerShell for Windows, Terminal for macOS or Linux)

# Ensure Git is installed
# Visit https://git-scm.com to download and install console Git if not already installed

# Clone the repository
git clone https://github.com/ShahRishi/OphthaBERT-glaucoma.git

# Navigate to the project directory
cd OphthaBERT-glaucoma

# Check if Docker is installed
docker --version  # Check the installed version of Docker
# Visit the official Docker website to install or update it if necessary

# Check if Docker daemon is running
docker info
sudo systemctl start docker   # Start daemon if not running
sudo systemctl enable docker  # Alternatively, enable starting daemon on boot
 
# Build and run docker image
docker compose up --build     # Run Docker image in the foreground
docker compose up --build -d  # Alternatively, run Docker image in the background
```

## Source Installation (not recommended)
```bash
# Open a terminal (Command Prompt or PowerShell for Windows, Terminal for macOS or Linux)

# Ensure Git is installed
# Visit https://git-scm.com to download and install console Git if not already installed

# Clone the repository
git clone https://github.com/ShahRishi/OphthaBERT-glaucoma.git

# Navigate to the project directory
cd OphthaBERT-glaucoma

# Install dependencies in virtual environment
uv venv --python=3.11               # create a python 3.11 virtual environment
uv pip install -r requirements.txt  # install locked dependencies

# Run API
python3 api.py
```

## Utilizing API
First test, the installation is working through the following command in your terminal
```bash
# Test binary endpoint
curl -X POST "http://127.0.0.1:8080/predict/binary" \
     -H "Content-Type: application/json" \
     -d '{
           "text": "Patient presents with elevated intraocular pressure and optic nerve changes."
         }'

# Test subtypes endpoint
curl -X POST "http://127.0.0.1:8080/predict/subtypes" \
     -H "Content-Type: application/json" \
     -d '{
           "text": "Patient has primary open angle glaucoma with elevated eye pressure."
         }'
```

We can also utilize this API in scripts while the image is running
```python
import requests

BASE_URL = "http://127.0.0.1:8080"

# Function to test the binary classification endpoint
def binary_classification(text):
    endpoint = f"{BASE_URL}/predict/binary"
    payload = {"text": text}
    headers = {"Content-Type": "application/json"}
    
    try:
        response = requests.post(endpoint, json=payload, headers=headers)
        if response.status_code == 200:
            print("Binary Classification Result:", response.json())
        else:
            print(f"Error {response.status_code}: {response.text}")
    except requests.RequestException as e:
        print("Error while making the request:", e)

# Function to test the subtype classification endpoint
def subtype_classification(text):
    endpoint = f"{BASE_URL}/predict/subtypes"
    payload = {"text": text}
    headers = {"Content-Type": "application/json"}
    
    try:
        response = requests.post(endpoint, json=payload, headers=headers)
        if response.status_code == 200:
            print("Subtype Classification Result:", response.json())
        else:
            print(f"Error {response.status_code}: {response.text}")
    except requests.RequestException as e:
        print("Error while making the request:", e)
```
