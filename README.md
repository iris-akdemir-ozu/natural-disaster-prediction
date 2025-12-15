**AI-Powered Natural Disaster Damage Assessment System**

**Project Overview**
This project implements a web-based automated damage assessment system utilizing deep learning techniques. The system analyzes high-resolution post-disaster satellite imagery to detect buildings and classify damage levels (No Damage, Minor Damage, Major Damage, Destroyed). By integrating a U-Net Convolutional Neural Network (CNN) with a Flask backend and Leaflet.js frontend, the application generates real-time semantic segmentation maps to aid in disaster response and risk analysis.

**System Architecture**
The system operates through a structured pipeline designed to handle geospatial data and deep learning inference:

1. Data Input: Users upload satellite imagery through the web interface.
2. Preprocessing:
   - The backend (Flask) receives the image.
   - To handle high-resolution satellite data, a "Sliding Window (Tiling)" algorithm is applied, dividing the image into 256x256 pixel patches to prevent resolution loss during resizing.
   - Image normalization is performed to match the model's input requirements.
3. Deep Learning Inference:
   - A custom U-Net architecture processes each tile.
   - The model performs pixel-wise classification into 5 distinct categories: Background, No Damage, Minor Damage, Major Damage, and Destroyed.
4. Post-Processing & Aggregation:
   - Predicted masks from individual tiles are stitched back together to reconstruct the full-size segmentation map.
   - A damage ratio is calculated to determine the overall "Risk Level" of the area.
5. Visualization:
   - The segmentation mask is overlaid on the original map using Leaflet.js, providing an interactive visual assessment for the user.

**Implementation Details**

**Dataset and Data Processing**
The model was trained on the xView2 (xBD) dataset, a large-scale dataset for building damage assessment.
- Data Parsing: The original dataset provided annotations in JSON/WKT polygon format. A custom Python script (scripts/prepare_dataset.py) was developed to parse these geospatial polygons and convert them into pixel-perfect semantic segmentation masks suitable for CNN training.

**Model Configuration**
- Architecture: U-Net (Encoder-Decoder network with skip connections).
- Loss Function: A combined loss function consisting of Categorical Cross-Entropy and Dice Loss was implemented to address class imbalance issues inherent in satellite imagery (e.g., vast background areas vs. small building footprints).
- Optimization: The model was optimized using the Adam optimizer with a dynamic learning rate.

**Tech Stack**
- Deep Learning: TensorFlow, Keras
- Computer Vision: OpenCV, NumPy
- Backend: Python, Flask
- Frontend: HTML5, CSS3, JavaScript, Leaflet.js

**Installation and Setup**

**Prerequisites**
- Python 3.8 or higher
- pip package manager

**Dataset Setup:**
The project uses the xView2 (xBD) dataset. Due to upload size limitations, the dataset images are not included in this repository.

To run the training script:
1. Download the "Challenge Training Set" from the official xView2 website: https://xview2.org/dataset
2. Extract the files into the `data/` directory in this project folder.
3. Run `python scripts/json_to_mask.py` to process the raw labels into masks.

After cloning the repository: 

**1. Install Dependencies**
pip install -r requirements.txt

**2. Dataset Preparation**
Execute the following command to process the raw labels into masks (change the directories in the code before running):
python scripts/json_to_mask.py

**3. Model Generation**
Execute the following command to train the model and save the weights:
python scripts/train_unet_model.py

(This script will utilize the data processing pipeline to train the U-Net model and save the output to the models/ directory.)

**Usage**

1. Start the Backend Server:
   python backend/app.py
   (The server will initialize at http://localhost:5000)

2. Access the Interface:
   Open the index.html file in a web browser.

3. Perform Analysis:
   Upload a satellite image to the system to view the predicted damage mask and risk statistics.

**Project Structure**
- backend/: Contains the Flask API and inference logic (app.py).
- scripts/: Contains utility scripts for data preparation (prepare_dataset.py) and model training (train_unet_model.py).
- data/: Directory structure for dataset storage (excluded from version control).
- models/: Directory for saving trained model weights.
- index.html: The main user interface file.
- requirements.txt: List of Python dependencies.
