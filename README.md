
# blackjax-ns for Gravitational Wave Inference on GPUs

This repository contains the source code and manuscript for the paper, **"Gravitational wave inference at GPU speed: A bilby-like nested sampling kernel within blackjax-ns"**.

## Abstract



## Repository Contents

This repository is structured into two main directories:

*   `/paper`: Contains the LaTeX source files for the manuscript, including figures and the bibliography. The main file is `paper/main.tex`.
*   `/src`: Contains the Python source code used to run the gravitational wave inference analyses, generate the comparison plots, and produce the results presented in the paper.

## Quick Start: Install and Run Guide

### Prerequisites

- Python 3.10+
- NVIDIA GPU with CUDA support (for GPU acceleration)
- Git

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd blackjax_ns_gw
   ```

2. **Install BlackJAX from the nested sampling branch:**
   ```bash
   pip install git+https://github.com/handley-lab/blackjax.git@nested_sampling
   ```

3. **Install other dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Running the Main Analysis

The main gravitational wave inference script is `src/bilbykernel_unitcube_4s.py`. This script performs GPU-accelerated nested sampling for gravitational wave parameter estimation using BlackJAX with original likelihood for an injected signal.

**To run the analysis:**

```bash
cd src
python bilbykernel_unitcube_4s.py
```

**Expected outputs:**
- `bilbykernel_unitcube_4s_results.csv`: Nested sampling posterior samples
- `adaptive_chain_diagnostics.pdf`: MCMC adaptation diagnostics
- `performance_diagnostics.pdf`: Detailed performance metrics

### Key Features

- **GPU Acceleration**: Uses JAX for GPU-accelerated likelihood evaluations
- **Custom Nested Sampling Kernel**: Bilby-inspired adaptive differential evolution sampler, based on 'acceptance-walk' method
- **11-parameter space**: Chirp mass, mass ratio, spins, sky location, and more
- **Periodic parameter handling**: Proper treatment of angular parameters (RA, phase, polarization)

### Configuration

The script uses the following key settings:
- 1000 live points for nested sampling
- Target acceptance rate of 60 accepted steps per point
- Power-law prior on luminosity distance
- Frequency range: 20-1024 Hz
- 4-second analysis duration

### Troubleshooting

- **Import Errors**: Ensure all dependencies are installed and the `custom_ns_kernels` module is in the Python path
- **File Not Found**: The script expects data files in the `src/` directory - they should be copied automatically during setup

## Reproducing the Results

The scripts to run the inference and generate the figures from the paper are located in the `src/` directory.

