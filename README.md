
# Gravitational-Wave Inference on GPUs with blackjax-ns

This repository contains the source code and manuscript for the paper:

**"Gravitational-wave inference at GPU speed: A bilby-like nested sampling kernel within blackjax-ns"**

**Paper**: [arXiv:2509.04336](https://arxiv.org/abs/2509.04336)  
**Data & Plots**: [Zenodo (DOI: 10.5281/zenodo.17012010)](https://zenodo.org/records/17012011)

## Citation

If you use this code in your work, please cite both:

1. **Nested Slice Sampling Framework:**  
   Yallup, D., Kroupa, N., & Handley, W. (2025). "Nested Slice Sampling." *FPI-ICLR2025*. [OpenReview](https://openreview.net/forum?id=ekbkMSuPo4&referrer=%5Bthe%20profile%20of%20David%20Yallup%5D)

2. **Gravitational-wave Inference Application:**  
   Prathaban, M., Yallup, D., Alvey, J., Yang, M., Templeton, W., & Handley, W. (2025). "Gravitational-wave inference at GPU speed: A bilby-like nested sampling kernel within blackjax-ns." [arXiv:2509.04336](https://arxiv.org/abs/2509.04336)

## Abstract

We present a GPU-accelerated implementation of the gravitational-wave Bayesian inference pipeline for parameter estimation and model comparison. Specifically, we implement the `acceptance-walk' sampling method, a cornerstone algorithm for gravitational-wave inference within the bilby and dynesty framework. By integrating this trusted kernel with the vectorized blackjax-ns framework, we achieve typical speedups of 20-40x for aligned spin binary black hole analyses, while recovering posteriors and evidences that are statistically identical to the original CPU implementation. This faithful re-implementation of a community-standard algorithm establishes a foundational benchmark for gravitational-wave inference. It quantifies the performance gains attributable solely to the architectural shift to GPUs, creating a vital reference against which future parallel sampling algorithms can be rigorously assessed. This allows for a clear distinction between algorithmic innovation and the inherent speedup from hardware. Our work provides a validated community tool for performing GPU-accelerated nested sampling in gravitational-wave data analyses.


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

1. **Create a virtual environment:**
   ```bash
   python -m venv blackjax_gw_env
   source blackjax_gw_env/bin/activate  # On Windows: blackjax_gw_env\Scripts\activate
   ```

2. **Clone the repository:**
   ```bash
   git clone https://github.com/mrosep/blackjax_ns_gw.git
   cd blackjax_ns_gw
   ```

3. **Install all dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   This will install key packages including: `blackjax` (nested sampling branch), `bilby`, `anesthetic`, `jimgw`, `ripple`, and other essential libraries for gravitational-wave analysis.

### Running the Main Analysis

The main analysis script is `src/example.py`. This comprehensive tutorial script performs GPU-accelerated nested sampling for gravitational-wave parameter estimation using BlackJAX, with detailed comments explaining each step.

**To run the analysis:**

```bash
cd src
python example.py
```

**Expected outputs:**
- `results.csv`: Nested sampling posterior samples (anesthetic format)
- `final_state.pkl`: Complete nested sampling final state
- `timings.pkl`: Performance timing information

### Analyzing Results

The output `results.csv` file can be analyzed using the [anesthetic](https://anesthetic.readthedocs.io/en/latest/index.html) package for nested sampling analysis:

```python
import anesthetic
import matplotlib.pyplot as plt

# Load the nested sampling results
samples = anesthetic.read_csv('results.csv')

# Plot posterior distributions
samples.plot_2d(['M_c', 'q'])  # Corner plot for chirp mass vs mass ratio
plt.show()

# Calculate Bayesian evidence
evidence_samples = samples.logZ(100)  # Evidence estimates, accounting for uncertainty in weights
print(f"Log evidence: {evidence_samples.mean():.2f} ± {evidence_samples.std():.2f}")

```

### Key Features

- **GPU Acceleration**: Uses JAX for GPU-accelerated likelihood evaluations
- **Custom Nested Sampling Kernel**: Bilby-inspired adaptive differential evolution sampler, based on 'acceptance-walk' method
- **Ripple Waveform Models**: Incorporates GPU-accelerated waveform models from the [Ripple library](https://github.com/tedwards2412/ripple/tree/main). Visit the Ripple repository to see available waveform models and learn how to import them.
- **Periodic parameter handling**: Proper treatment of periodic parameters (e.g., phase) - easily extensible to any parameter with periodic boundary conditions

### Configuration

**Nested Sampling Settings:**

For GPU implementations, the nominal `n_live` differs from CPU implementations. We recommend:

- **Live points**: For effectively N live points, use `n_live = int(2 * log(2) * N)` actual live points
- **Deletion rate**: Set `num_delete = 0.5 * n_live` (as shown in `example.py`)
- **Target acceptance**: `target = 60` accepted steps per point
- **MCMC steps**: `max_mcmc = 5000` maximum steps per MCMC chain
- **Proposal limit**: `max_proposals = 1000` (GPU-specific parameter, robust default that rarely needs adjustment)

For example, to run with effectively 1000 live points:
```python
n_live = int(2 * np.log(2) * 1000)  # ≈ 1386 actual live points
num_delete = int(0.5 * n_live)      # ≈ 693 deletion rate
```

See the [arXiv paper](https://arxiv.org/abs/2509.04336) for detailed explanations of these GPU-specific settings.

### Troubleshooting

- **Import Errors**: Ensure all dependencies are installed and the `custom_kernels` module is in the Python path
- **CUDA Errors**: Verify your GPU drivers and CUDA installation are compatible with the JAX CUDA packages
- **Memory Issues**: Reduce `n_live` or adjust `XLA_PYTHON_CLIENT_MEM_FRACTION` environment variable
- **File Not Found**: The script expects data files (ASD files, etc.) in the `src/` directory

### Support

For questions, issues, or contributions:
- **Email**: myp23@cam.ac.uk
- **Issues**: Submit issues on the [GitHub repository](https://github.com/mrosep/blackjax_ns_gw/issues)
- **Pull Requests**: Contributions are welcome via pull requests

## Reproducing the Results

The scripts to run the inference and generate the figures from the paper are located in the `src/` directory.

