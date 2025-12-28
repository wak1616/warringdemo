# Warring Demo - LRI Arcuate Incision Predictor

A data analysis and machine learning project for predicting Limbal Relaxing Incision (LRI) parameters during cataract surgery to correct astigmatism.

## Overview

This project analyzes ophthalmic measurement data to build predictive models for arcuate incision planning. It processes various astigmatism measurements from devices like Pentacam and IOL 700, and uses Barrett Integrated-K calculations to determine optimal LRI parameters.

## Features

- **Data Processing**: Handles astigmatism measurements in both positive and negative cylinder notation
- **Axis Conversion**: Converts astigmatism magnitudes to specific target axes using double-angle formulas
- **Feature Engineering**: Creates derived features like magnitude at Barrett Integrated-K axis
- **ML Pipeline**: Prepares data for predicting single vs. paired arcuate incisions

## Data

The project uses de-identified LRI AI Spreadsheet data containing:
- Patient demographics (Age, Laterality)
- Manifest refraction data (Sphere, Cylinder, Axis)
- Pentacam measurements (∆k magnitude and axis)
- IOL 700 measurements (∆k, posterior astigmatism, ∆TK)
- Barrett Integrated-K calculations
- LRI surgical parameters (length and axis)

## Getting Started

### Prerequisites

- Conda or Mamba package manager
- Python 3.12+

### Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd warringdemo
   ```

2. Create the conda environment:
   ```bash
   conda env create -f environment.yml
   ```

3. Activate the environment:
   ```bash
   conda activate warring_data
   ```

4. Launch Jupyter:
   ```bash
   jupyter notebook
   ```

## Usage

Open `data_notebook.ipynb` to explore the data processing and analysis pipeline:

1. Load and preprocess the LRI spreadsheet data
2. Convert astigmatism measurements to consistent formats
3. Calculate magnitudes at Barrett Integrated-K axis
4. Prepare features for machine learning models

## Project Structure

```
warringdemo/
├── data/                          # Data files (de-identified)
│   └── LRI_AI_Spreadsheet_*.csv
├── Notes 9.2025/                  # Clinical notes
├── data_notebook.ipynb            # Main analysis notebook
├── environment.yml                # Conda environment specification
└── README.md
```

## Key Concepts

- **Barrett Integrated-K**: A calculation method for predicting corneal astigmatism
- **Double-angle formula**: Used to convert astigmatism measurements between different axes
- **WTR/ATR/Oblique**: With-the-rule (0°/180°), Against-the-rule (90°), and Oblique (45°/135°) astigmatism classifications

## Contributing

Feel free to contribute to this project by submitting issues or pull requests.

## License

TBD
