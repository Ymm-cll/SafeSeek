# SafeSeek: Universal Attribution of Safety Circuits in Language Models

<p align="center">
  <img src="safeseek.png" alt="SafeSeek Framework" width="100%"/>
</p>

## Overview

SafeSeek is a universal safety attribution framework for locating and analyzing safety circuits in Large Language Models (LLMs).

## Project Structure

```
safeseek/
├── alignment/       # Safety alignment circuit
│   ├── head.py
│   ├── neuron.py
│   └── head_neuron.py
└── backdoor/        # Backdoor circuit
    ├── head.py
    ├── neuron.py
    └── head_neuron.py
```

## Usage

1. **Configure paths** in the training scripts:
   - `MODEL_PATH`: path to your model
   - `OUTPUT_DIR`: output directory
   - `Dataset path`

2. **Run analysis**:
   ```bash
   # For attention head analysis
   python alignment/head.py
   python backdoor/head.py
   
   # For neuron analysis
   python alignment/neuron.py
   python backdoor/neuron.py
   
   # For combined analysis
   python alignment/head_neuron.py
   python backdoor/head_neuron.py
   ```

3. Results will be saved to the configured output directory

