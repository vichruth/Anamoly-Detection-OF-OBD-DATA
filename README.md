# Anomaly Detection in OBD Data  

This project detects anomalies in vehicle onboard diagnostics (OBD) data using an autoencoder neural network.  

## 📂 Project Structure  
- `csvimporting.py` → Python script for data preprocessing and anomaly detection  
- `requirements.txt` → Dependencies needed for the project  
- `odbdata/` → Folder containing CSV files with OBD data  

## 🚀 Setup & Installation  
1. Clone the repository:  
   ```sh
   git clone https://github.com/vichruth/Anamoly-Detection-OF-OBD-DATA.git
   cd Anamoly-Detection-OF-OBD-DATA
2.Create & activate a virtual environment:
```sh
   python -m venv .venv
   source .venv/bin/activate
```
3.Install Dependencies:
```sh
  pip install -r requirements.txt
```
3.Run the script:
```sh
  python csvimporting.py
```
📊 Anomaly Detection
Uses an autoencoder neural network to reconstruct normal OBD data.
Anomalies are detected when the reconstruction error exceeds a threshold.
Generates a histogram to visualize the anomaly detection threshold.
🔧 Technologies Used
Python (NumPy, Pandas)
TensorFlow & Keras
Matplotlib
Scikit-learn
🛠 Future Improvements
Tune hyperparameters for better anomaly detection.
Deploy the model for real-time OBD anomaly detection.
Build a UI to visualize results interactively.
💡 Contributions & feedback are welcome!
