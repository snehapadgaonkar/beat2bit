# 🫀 Beat2Bit

> Ultra-low-power edge AI for ECG arrhythmia detection — clinical-grade accuracy, microcontroller footprint.

[Getting Started](#getting-started) •
[Built With](#built-with) •
[Roadmap](#roadmap) •
[Contributing](#contributing) •
[License](#license)

---

Beat2Bit investigates a core TinyML research question:

**How much can an ECG arrhythmia detection neural network be compressed and optimised for edge deployment while maintaining clinically acceptable classification performance?**

The result: a 1D convolutional network trained on the full MIT-BIH Arrhythmia Database, compressed to a 34.6 KB INT8 model that runs at **0.073 ms per beat** on a microcontroller — with no cloud dependency and full AAMI EC57 compliance.

---

## Getting Started

Instructions for setting up the project locally — both the ML pipeline and the interactive web dashboard.

### Prerequisites

- [Python 3.9+](https://www.python.org/downloads/)
- [Node.js v18.17+](https://nodejs.org/) (for the frontend dashboard)
- [npm](https://www.npmjs.com/) (comes with Node.js)
- [Git](https://git-scm.com/)

### Installation

```bash
git clone https://github.com/snehapadgaonkar/beat2bit.git
cd beat2bit
```

Install Python dependencies:

```bash
pip install -r requirements.txt
pip install tensorflow scikit-learn tensorflow-model-optimization wfdb
```

Install frontend dependencies:

```bash
cd frontend
npm install
```

### Run

**Web dashboard** (interactive research visualisation):

```bash
cd frontend
npm run dev
# Open http://localhost:3000
```

**Full optimised training pipeline** (downloads MIT-BIH automatically, trains all 5 models, writes JSON reports):

```bash
python scripts/train_optimised.py
```

**Run individual pipeline stages:**

```bash
python src/02_preprocessing.py        # Download + window MIT-BIH
python src/03_baseline_model.py       # Train FP32 baseline
python src/04_quantization_pruning.py # Prune + quantize
```

**Run on Google Colab** (recommended — free GPU):

```bash
# In Colab:
!pip install tensorflow wfdb scikit-learn tensorflow-model-optimization
# Upload scripts/train_optimised.py, then:
!python scripts/train_optimised.py
```

### Test

```bash
pytest tests/ -v
```

---

## Results

Evaluated on the MIT-BIH DS2 test set (50,262 beats, patient-independent AAMI EC57 split):

| Model | Accuracy | Sensitivity | +Predictivity | Latency | Size | AAMI EC57 |
|---|---|---|---|---|---|---|
| Baseline 1D-CNN (FP32) | 93.8% | 75.0% | 71.3% | 18.0 ms | 87.5 KB | ✅ PASS |
| **INT8 Quantized** ← deployed | **93.8%** | **75.0%** | **71.2%** | **0.073 ms** | **34.6 KB** | **✅ PASS** |
| Pruned 50% + INT8 | 91.9% | 75.1% | 61.2% | 0.058 ms | 34.6 KB | ⚠️ Se only |
| Pruned 60% + INT8 | 90.3% | 75.1% | 54.9% | 0.055 ms | 34.6 KB | ⚠️ Se only |

The INT8 hybrid quantization model is the recommended deployed model: **2.59× smaller**, **247× faster** than FP32, with zero AAMI EC57 degradation.

---

## Built With

- [TensorFlow / Keras](https://www.tensorflow.org/) — 1D CNN training and INT8 quantization
- [TensorFlow Model Optimization](https://www.tensorflow.org/model_optimization) — magnitude pruning
- [TensorFlow Lite](https://www.tensorflow.org/lite) — MCU-ready `.tflite` conversion
- [wfdb](https://wfdb.readthedocs.io/) — MIT-BIH Arrhythmia Database loading
- [scikit-learn](https://scikit-learn.org/) — AAMI EC57 metric computation
- [Next.js 14](https://nextjs.org/) — interactive research dashboard
- [Tailwind CSS](https://tailwindcss.com/) — responsive UI
- [Recharts](https://recharts.org/) — benchmark visualisations
- [MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/) — training and evaluation corpus

---

## Repository Structure

```text
beat2bit/
│
├── frontend/                  # Next.js interactive dashboard
│   ├── app/                   # App router, layout, globals
│   ├── components/            # React components (ECG chart, sections, chat)
│   ├── lib/                   # Utility functions, formatters
│   └── public/reports/        # JSON benchmark reports (read by the UI)
│
├── scripts/
│   └── train_optimised.py     # Full reproducible training pipeline v8
│
├── src/                       # Step-by-step ML source scripts
│   ├── 02_preprocessing.py    # MIT-BIH windowing + AAMI split
│   ├── 03_baseline_model.py   # FP32 1D-CNN training
│   └── 04_quantization_pruning.py # Pruning + INT8 quantization
│
├── notebooks/                 # Colab-ready Jupyter notebooks
├── configs/                   # YAML configs for each experiment
├── docs/                      # Methodology and reproducibility guides
├── tests/                     # pytest test suite
│
├── data/                      # (git-ignored) MIT-BIH + processed arrays
├── models/                    # (git-ignored) saved .keras and .tflite files
│
├── CONTRIBUTING.md
├── LICENSE
└── requirements.txt
```

---

## Roadmap

### v1.0.0 — Research baseline

- [x] Patient-independent AAMI EC57 data split
- [x] Baseline 1D-CNN (FP32) with class-imbalance handling
- [x] INT8 hybrid quantization — AAMI EC57 compliant
- [x] Magnitude pruning at 50% and 60% sparsity
- [x] Interactive Next.js benchmark dashboard
- [x] Full responsive UI (mobile → tablet → desktop)
- [x] Reproducible training pipeline (`scripts/train_optimised.py`)

### v1.1.0 — Planned

- [ ] Quantization-aware training (QAT) to recover pruned-model precision
- [ ] Multi-class output (N / V / A / AF instead of binary)
- [ ] On-device C++ inference demo (Arduino / STM32)
- [ ] PTB-XL and European ST-T dataset integration

> **Note**
> Contributions toward the v1.1.0 items are very welcome — see [CONTRIBUTING.md](CONTRIBUTING.md).

---

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request.

---

## License

This project is licensed under the [MIT License](LICENSE).
