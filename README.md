# Quantum Rocket Simulator 🚀

Quantum Rocket Simulator is a privacy-first, desktop AI assistant that predicts rocket launch risks and anomalies using:

* Local LLM (Llama.cpp)
* Quantum-inspired probabilistic reasoning (Pennylane)
* Real-time contextual inputs (weather, GPS, mission data, etc.)
* Semantic memory (Weaviate vector DB, SQLite)
* End-to-end encryption (AES-GCM, Argon2id)
* GUI (Tkinter + CustomTkinter)
* Automated AI image generation

> This app fuses classical AI, quantum logic, and aerospace analytics into a single, secure, extensible desktop tool.

---

## ✨ Features

* **Mission Risk Assessment:** Enter mission context, get LLM risk predictions with explanations.
* **Quantum Probability Bands:** Quantum circuit computes a “banded” risk range for each mission.
* **Semantic Memory:** Finds and uses similar historical launches/anomalies via Weaviate vector search.
* **Encrypted Storage:** Everything is AES-GCM encrypted; key management & rotation are built in.
* **AI Image Generation:** Every mission can have its own AI-generated image.
* **Runs Fully Local:** No cloud APIs or external data leakage—everything is on your machine.

---

## 🖼️ Screenshot

> *(Add a screenshot image here for best results!)*

---

## 🚀 Quick Start

### 1. Clone the Repository

```
git clone https://github.com/dosh41126/quantum-rocket-simulator.git
cd quantum-rocket-simulator
```

### 2. Build & Run with Docker

*All configuration (`config.json`, etc.) is handled automatically inside the Dockerfile.*

```
docker build -t quantum-rocket-simulator .
docker run -it --rm -p 8000:8000 quantum-rocket-simulator
```

* The GUI, API, and database all run automatically.

---

### 3. Run Locally (Advanced/Dev)

* Python 3.10+ required

```
pip install -r requirements.txt
python quantum_rocket_simulator.py
```

* NLTK data is auto-downloaded on first run.
* Place your Llama.cpp model (`llama-2-7b-chat.ggmlv3.q8_0.bin`) in `/data/`.
* (Optional) Add a `logo.png` to the project root.

---

## 🧠 How It Works

1. **Enter mission context:** Site, payload, weather, GPS, recent tests/anomalies, etc.
2. **Quantum logic:** Computes a context-driven probability band for each risk assessment.
3. **LLM prompting:** All context (and relevant historical memory) forms a Sherlock-Holmes-style prompt.
4. **LLM output:** Predicts risks/anomalies, confidence, reasoning, and a quantum band for each.
5. **Display:** Results and AI-generated images appear in the GUI.
6. **Storage:** All user/bot interactions are AES-GCM encrypted and stored in SQLite/Weaviate.

---

## 🛡️ Security & Privacy

* AES-GCM encryption for all stored data
* Argon2id key derivation and rotation
* No cloud: 100% private, runs locally

---

## 🛠️ Architecture

* Tkinter / CustomTkinter: GUI
* FastAPI: Local REST API
* Weaviate (embedded): Vector DB
* Llama.cpp: Local LLM
* Pennylane: Quantum logic
* NLTK / TextBlob: NLP & semantics
* cryptography, argon2: Encryption

---

## 📦 Project Structure

```
quantum-rocket-simulator/
│
├── quantum_rocket_simulator.py      # Main application
├── requirements.txt                 # Python dependencies
├── Dockerfile                       # Build instructions (config.json handled here)
├── /data/                           # LLM model(s)
├── /secure/                         # Key vault & encrypted data
├── /saved_images/                   # Generated images
├── logo.png                         # (optional) logo for GUI
└── README.md
```

---

## 🤝 Contributing

PRs and issues are welcome!

1. Fork the repo
2. Make your changes
3. Open a pull request

---

## 📄 License
GPL3

---

## 🙏 Acknowledgments

* [Llama.cpp](https://github.com/ggerganov/llama.cpp)
* [Pennylane](https://pennylane.ai/)
* [Weaviate](https://weaviate.io/)
* The open-source Python & ML community

---

**Built for a new era of secure, local, AI-powered engineering.**
