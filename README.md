# DroneLogNLP
NLP-powered Drone &amp; Jet Mission Log Analyzer with Real-Time Querying and Embeddings.
# DroneLogNLP

A Python-based project for **real-time drone log analysis** using Natural Language Processing (NLP) with a user-friendly GUI. This project processes synthetic drone logs, generates embeddings using **sentence-transformers**, and allows querying to extract actionable insights from the logs.

---

## 🔹 Motivation

In the rapidly growing field of autonomous drones, log analysis is critical for monitoring system behavior, identifying issues, and improving performance. Most solutions require manual log inspection or expensive cloud-based NLP services.

The motivation behind **DroneLogNLP** is to:

* Automate log analysis using NLP techniques.
* Provide real-time querying capability through a simple GUI.
* Offer a fully local solution without relying on third-party cloud services.
* Help researchers, drone developers, and hobbyists analyze logs efficiently.

---

## 🔹 Features

* **Log Analysis:** Load and process drone logs in CSV format.
* **Embedding Generation:** Use `sentence-transformers` to create semantic embeddings for log content.
* **Real-Time Querying:** Ask questions about drone logs and get instant responses.
* **GUI Interface:** Lightweight Tkinter GUI with an interactive design and an embedded image.
* **Fully Local:** No need for internet connection after initial setup.

---

## 🔹 Directory Structure

```
nlp-project/
├── analyze_logs.py          # Script to load and process drone logs
├── embedding_model.py       # Generates embeddings for the logs
├── gui_tkinter.py           # Tkinter GUI for real-time queries
├── processed_logs.csv       # Processed log dataset
├── synthetic_drone_logs.csv # Original synthetic log dataset
├── summary_embeddings.npy   # Saved embeddings for querying
├── Untitled.jpeg            # Image used in GUI
├── requirements.txt         # Python dependencies
└── venv/                    # Virtual environment (optional)
```

---

## 🔹 Installation

1. **Clone the repository**

```bash
git clone https://github.com/Zenvila/DroneLogNLP.git
cd DroneLogNLP
```

2. **Create a virtual environment and activate it**

```bash
python -m venv venv
source venv/bin/activate      # Linux / macOS
venv\Scripts\activate       # Windows
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Download or ensure the `sentence-transformers` model is available**
   The project uses **`all-MiniLM-L6-v2`** model locally for embeddings.

---

## 🔹 Usage

### Step 1: Generate embeddings

```bash
python embedding_model.py
```

* Processes the log dataset.
* Generates semantic embeddings.
* Saves embeddings as `summary_embeddings.npy`.

### Step 2: Run GUI for queries

```bash
python gui_tkinter.py
```

* Opens a Tkinter GUI.
* Enter queries about drone logs in real-time.
* Get immediate results displayed in the GUI.
* GUI includes a visually appealing layout with your embedded image.

---

## 🔹 Example Queries

Here are some example questions you can ask in the GUI:

* "Strike on Sector B2"
* "Drone altitude dropped below safe threshold"
* "Battery warning issued during flight"
* "Communication lost with drone"
* "Log anomalies in electronics system"

---

## 🔹 Use Cases

* Real-time drone operations monitoring.
* Post-flight analysis for autonomous drones.
* Research in AI-based log analysis.
* Educational tool for NLP and embedding concepts.
* Automation of log review for drone fleets.

---

## 🔹 Technologies Used

* **Python 3.12**
* **Pandas** – for CSV data processing
* **Numpy** – for embeddings handling
* **Sentence-Transformers** – semantic embeddings
* **Tkinter** – GUI interface
* **Pillow** – image handling in GUI

---

## 🔹 Future Improvements

* Add support for **real drone logs** from actual flight controllers.
* Integrate **graphical visualizations** for analytics.
* Extend NLP models to **summarize logs automatically**.
* Allow **multi-query session tracking** in GUI.

---

## 🔹 License

This project is **open-source** and available under the MIT License.

---
### 🔹 Short Description for GitHub Summary

**DroneLogNLP:** Local Python project for real-time drone log analysis with NLP embeddings and a user-friendly GUI.
