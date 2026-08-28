# 🫀 Beat2Bit: Ultra-Low-Power Edge AI for Remote Medical Devices

> **Core Research Question:** How much can an ECG arrhythmia detection neural network be compressed and optimized for edge deployment while maintaining acceptable classification performance?

Beat2Bit is a research-oriented Deep Learning and TinyML project. It investigates the trade-offs between classification performance, model size, inference latency, and energy consumption when deploying 1D Convolutional Neural Networks (CNNs) for ECG Arrhythmia detection directly onto resource-constrained microcontrollers.

---

## 🚀 Key Features
* **100% Offline Inference:** No cloud processing required, ensuring functionality in remote areas.
* **Ultra-Low Power (TinyML):** Utilizes **INT8 Quantization** and **Magnitude Pruning** to reduce model footprint by over 6x, enabling weeks of battery life.
* **Privacy-Preserving:** Biometric data never leaves the device. Compliant by design.
* **Enterprise Dashboard:** A modern, responsive Next.js frontend built with Tailwind CSS and shadcn/ui to visualize the research pipeline, datasets, and live edge telemetry.

---

## 📂 Repository Structure

```text
beat2bit/
│
├── frontend/                  # Next.js Interactive Dashboard & UI
│   ├── app/                   # App router & pages
│   ├── components/            # React components (ECG Chart, UI)
│   └── public/                # Static assets
│
├── notebooks/                 # Jupyter Notebooks (Ready for Google Colab)
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_baseline_model.ipynb
│   └── 04_quantization_pruning.ipynb
│
├── src/                       # Python source code for ML Pipeline
│   ├── 01_data_exploration.py
│   ├── 02_preprocessing.py
│   ├── 03_baseline_model.py
│   └── 04_quantization_pruning.py
│
├── data/                      # (Ignored) MIT-BIH dataset & processed arrays
├── models/                    # (Ignored) Saved .keras and .tflite models
└── requirements.txt           # Python dependencies
```

---

## 💻 How to Run the Web Dashboard Locally

The frontend is a modern Next.js 14 application. To run it on your local machine, follow these steps:

### Prerequisites
* **Node.js** (v18.17 or higher) installed on your machine.
* **npm** (comes with Node.js).

### Steps
1. **Clone the repository and switch to the branch (if not already there):**
   ```bash
   git clone https://github.com/snehapadgaonkar/beat2bit.git
   cd beat2bit
   ```

2. **Navigate to the frontend directory:**
   ```bash
   cd frontend
   ```

3. **Install the dependencies:**
   ```bash
   npm install
   ```

4. **Start the development server:**
   ```bash
   npm run dev
   ```

5. **View the App:**
   Open your browser and navigate to [http://localhost:3000](http://localhost:3000). You should see the interactive Beat2Bit enterprise dashboard!

---

## 🧠 How to Run the Machine Learning Pipeline

You can run the ML pipeline locally or upload the `notebooks/` folder directly to Google Colab.

### Running Locally (Python)
1. Navigate to the root directory of the project:
   ```bash
   cd beat2bit
   ```
2. Install the required Deep Learning packages:
   ```bash
   pip install -r requirements.txt
   pip install tensorflow scikit-learn tensorflow-model-optimization
   ```
3. Execute the pipeline in order:
   ```bash
   python src/01_data_exploration.py
   python src/02_preprocessing.py
   python src/03_baseline_model.py
   python src/04_quantization_pruning.py
   ```

### Running on Google Colab
1. Go to [Google Colab](https://colab.research.google.com/).
2. Click **File > Upload notebook**.
3. Select the `.ipynb` files from the `notebooks/` directory.
4. Run the cells sequentially. *Note: For step 2, the `wfdb` library will automatically download the MIT-BIH dataset to the Colab environment.*

---

## 🔬 Scientific Pipeline Overview
1. **Data Preprocessing:** Extracts fixed-size (180 sample) heartbeat windows from the **MIT-BIH Arrhythmia Database**, utilizing an AAMI patient-aware split to prevent data leakage.
2. **Baseline Model:** A lightweight FP32 1D CNN trained to classify beats as Normal (N) or Abnormal (V, A, etc.).
3. **Pruning:** Employs `tensorflow_model_optimization` to apply Magnitude Pruning, forcing up to 70% of the network's weights to zero to reduce multiply-accumulate (MAC) operations.
4. **Quantization:** Uses `TFLiteConverter` for INT8 Post-Training Quantization (PTQ), shrinking the memory footprint from ~75 KB down to ~11.5 KB, making it ready for Microcontroller (C++) deployment.
