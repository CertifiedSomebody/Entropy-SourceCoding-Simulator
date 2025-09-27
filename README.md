# Entropy Calculator & Information Rate Estimator

This project implements a tool in **Python** to compute entropy and estimate information rate for text and audio signals, as well as simulate **Huffman** and **Shannon–Fano** source coding.

## Features

* **Entropy estimation**: Shannon entropy from symbol probabilities.
* **Block entropy & rate extrapolation**: Hₙ for block sizes and extrapolated entropy rate.
* **Universal estimator**: LZ78-based entropy rate estimator.
* **Coding algorithms**: Huffman and Shannon–Fano code construction and average code length.
* **Data handling**:

  * Text → characters as symbols.
  * Audio (WAV) → quantized samples as symbols.
* **CLI options** for experiments and optional plotting.

## Requirements

* Python 3.8+
* Required libraries: `numpy`, `scipy`
* Optional (for plotting): `matplotlib`

Install requirements:

```bash
pip install numpy scipy matplotlib
```

## Usage

### 1. Demo mode (toy example)

```bash
python Entropy_InfoRate_Simulator.py --mode demo
```

### 2. Text mode

```bash
python Entropy_InfoRate_Simulator.py --mode text --file sample.txt --max-block 8 --plot
```

* `--file` : input text file
* `--max-block` : maximum block size for entropy estimation
* `--plot` : plot Hₙ and LZ78 estimates

### 3. Audio mode

```bash
python Entropy_InfoRate_Simulator.py --mode audio --file sample.wav --plot
```

* `--file` : input WAV file (mono/stereo)
* Downsampling and quantization are handled inside the script.

## Example Output

```
Results:
H1_bits : 4.192
Hn : { 'Hn': [...], 'extrapolated_H_rate': 3.95, ... }
lz78_bits_per_sym : 4.01
huffman_avg_len : 4.21
shannon_fano_avg_len : 4.32
```

## Project Structure

```
Entropy_InfoRate_Simulator.py   # Main script
README.md                       # Instructions
sample.txt                      # Example text input (add your own)
sample.wav                      # Example audio input (add your own)
```

## Notes

* Large block sizes (`--max-block > 8`) can be slow for long files.
* Huffman/Shannon–Fano codes are built directly from symbol probabilities.
* Audio must be `.wav`; other formats can be converted beforehand.

---

**Author**: Team Prabhat / DCOM Project
