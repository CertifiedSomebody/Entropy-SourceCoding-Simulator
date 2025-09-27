"""
Entropy Calculator & Source Coding Simulator
File: Entropy_InfoRate_Simulator.py

What this file contains:
- Entropy calculations for discrete sources (from symbol frequencies / probabilities)
- Block entropy estimator (H_n) and extrapolation to entropy rate
- LZ78-based information rate estimator (universal estimator)
- Huffman coding implementation: build tree, encode, decode, compute avg code length
- Shannon-Fano coding implementation: build codebook, compute avg length
- Utilities to load text files and WAV audio (mono) and convert to symbol sequences
- Simple CLI and examples at bottom for running experiments and plotting results

Usage examples (from the command line):
> python Entropy_InfoRate_Simulator.py --mode text --file sample.txt --max-block 8
> python Entropy_InfoRate_Simulator.py --mode audio --file sample.wav --frame-size 1024

Dependencies:
- Python 3.8+
- numpy, scipy, matplotlib (optional for plotting)

Notes:
- The file is written to be readable and easy to adapt for project submission.
- For large files and higher block sizes, block-entropy estimation may be slow and memory-intensive.

"""

import sys
import argparse
from collections import Counter, defaultdict
import math
import heapq
import numpy as np

try:
    from scipy.io import wavfile
except Exception:
    wavfile = None

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

# ---------------------------
# Basic entropy utilities
# ---------------------------

def probs_from_sequence(seq):
    c = Counter(seq)
    total = sum(c.values())
    return {s: v / total for s, v in c.items()}


def entropy_from_probs(pdict, base=2):
    """Compute Shannon entropy H(X) given a dict of probabilities."""
    H = 0.0
    for p in pdict.values():
        if p > 0.0:
            H -= p * math.log(p, base)
    return H


def entropy_of_sequence(seq, base=2):
    return entropy_from_probs(probs_from_sequence(seq), base=base)

# ---------------------------
# Block entropy estimator
# ---------------------------

def block_entropy(seq, n, base=2):
    """Estimate H_n (block entropy) for block size n: H_n = - \sum p(block) log p(block) / n"""
    if n <= 0:
        raise ValueError("n must be >= 1")
    blocks = Counter()
    L = len(seq)
    for i in range(L - n + 1):
        blocks[tuple(seq[i:i+n])] += 1
    total = sum(blocks.values())
    Hblock = 0.0
    for count in blocks.values():
        p = count / total
        Hblock -= p * math.log(p, 2)
    return Hblock / n


def estimate_entropy_rate_by_extrapolation(seq, max_block=6):
    """Compute H_n for n=1..max_block and return estimates + a simple extrapolation.
    Extrapolation: fit H_n versus 1/n line and take intercept as estimate (heuristic).
    Returns: dict with Hn list and extrapolated value
    """
    Hn = []
    for n in range(1, max_block+1):
        try:
            Hn_val = block_entropy(seq, n)
        except MemoryError:
            Hn_val = float('nan')
        Hn.append(Hn_val)
    # linear fit Hn = a*(1/n) + H_rate
    xs = np.array([1.0 / n for n in range(1, max_block+1)])
    ys = np.array(Hn)
    mask = np.isfinite(ys)
    if mask.sum() >= 2:
        a, H_rate = np.polyfit(xs[mask], ys[mask], 1)
    else:
        H_rate = float('nan')
    return {
        'Hn': Hn,
        'extrapolated_H_rate': H_rate,
        'block_sizes': list(range(1, max_block+1))
    }

# ---------------------------
# LZ78-based estimator
# ---------------------------

def lz78_entropy_estimator(seq):
    """Use LZ78 parsing length-based estimator. For sequence length n and c phrases,
    estimate entropy rate H ≈ (c * log n) / n (natural logs -> convert to bits).
    See universal coding bounds; this is a simple practical estimator.
    """
    n = len(seq)
    # dictionary of phrases -> next symbol
    phrases = {}
    w = []
    c = 0
    i = 0
    while i < n:
        j = i
        # find the longest phrase starting at i that is in dict
        phrase = ()
        last_found = None
        while j < n:
            candidate = tuple(seq[i:j+1])
            if candidate in phrases:
                last_found = candidate
                j += 1
            else:
                break
        # add new phrase
        new_phrase = tuple(seq[i:j+1])
        phrases[new_phrase] = True
        c += 1
        if j == i:
            i += 1
        else:
            i = j+0
    if n == 0:
        return 0.0
    # estimator (bits per symbol)
    H_est = (c * math.log(n, 2)) / n
    return H_est

# ---------------------------
# Huffman coding
# ---------------------------

class HuffmanNode:
    def __init__(self, symbol=None, freq=0, left=None, right=None):
        self.symbol = symbol
        self.freq = freq
        self.left = left
        self.right = right

    def __lt__(self, other):
        return self.freq < other.freq


def build_huffman_code(prob_dict):
    heap = []
    for sym, p in prob_dict.items():
        heapq.heappush(heap, (p, HuffmanNode(symbol=sym, freq=p)))
    if len(heap) == 0:
        return {}
    while len(heap) > 1:
        p1, n1 = heapq.heappop(heap)
        p2, n2 = heapq.heappop(heap)
        merged = HuffmanNode(symbol=None, freq=p1+p2, left=n1, right=n2)
        heapq.heappush(heap, (merged.freq, merged))
    root = heapq.heappop(heap)[1]

    codebook = {}
    def walk(node, prefix=''):
        if node is None:
            return
        if node.symbol is not None:
            codebook[node.symbol] = prefix or '0'
            return
        walk(node.left, prefix + '0')
        walk(node.right, prefix + '1')
    walk(root, '')
    return codebook

# ---------------------------
# Shannon-Fano coding (simple recursive split)
# ---------------------------

def build_shannon_fano_code(prob_dict):
    items = sorted(prob_dict.items(), key=lambda x: -x[1])
    codebook = {sym: '' for sym, _ in items}

    def split_assign(items_list):
        if len(items_list) <= 1:
            return
        # find split point where cumulative probs are as balanced as possible
        total = sum(p for _, p in items_list)
        cum = 0.0
        best_idx = 0
        best_diff = float('inf')
        for i in range(len(items_list)):
            cum += items_list[i][1]
            diff = abs((total - cum) - cum)
            if diff < best_diff:
                best_diff = diff
                best_idx = i
        left = items_list[:best_idx+1]
        right = items_list[best_idx+1:]
        for s, _ in left:
            codebook[s] += '0'
        for s, _ in right:
            codebook[s] += '1'
        split_assign(left)
        split_assign(right)

    split_assign(items)
    # some symbols may still have empty code if only one symbol exists
    for s in codebook:
        if codebook[s] == '':
            codebook[s] = '0'
    return codebook

# ---------------------------
# Utilities: average length, encode/decode length estimates
# ---------------------------

def average_code_length(codebook, prob_dict):
    avg = 0.0
    for s, p in prob_dict.items():
        l = len(codebook.get(s, ''))
        avg += p * l
    return avg

# ---------------------------
# Helpers for audio/text -> symbol sequences
# ---------------------------

def text_to_symbols(text, alphabet=None):
    # simple: characters as symbols; optionally map to limited alphabet
    seq = list(text)
    if alphabet is not None:
        # map unknowns to a token
        mapping = {c: c for c in alphabet}
        seq = [c if c in mapping else '<UNK>' for c in seq]
    return seq


def wav_to_symbols(filename, downsample=1, quantize_bits=None):
    if wavfile is None:
        raise RuntimeError('scipy is required to load wav files')
    sr, data = wavfile.read(filename)
    if data.ndim > 1:
        data = data.mean(axis=1).astype(data.dtype)
    if downsample > 1:
        data = data[::downsample]
    if quantize_bits is not None:
        # simple uniform quantization
        max_val = np.max(np.abs(data))
        levels = 2 ** quantize_bits
        if max_val == 0:
            quant = np.zeros_like(data)
        else:
            norm = (data / max_val + 1.0) / 2.0  # 0..1
            quant = (norm * (levels - 1)).astype(int)
        return list(map(int, quant))
    else:
        return list(map(int, data))

# ---------------------------
# CLI / Experiment runner
# ---------------------------

def run_text_experiment(filename, max_block=6, do_huffman=True, do_sf=True, plot=False):
    with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read()
    seq = text_to_symbols(text)
    prob_dict = probs_from_sequence(seq)
    H1 = entropy_from_probs(prob_dict)
    results = {
        'H1_bits': H1,
    }
    # block entropy
    extrap = estimate_entropy_rate_by_extrapolation(seq, max_block=max_block)
    results['Hn'] = extrap

    # LZ78 estimator
    lz_est = lz78_entropy_estimator(seq)
    results['lz78_bits_per_sym'] = lz_est

    if do_huffman:
        code_h = build_huffman_code(prob_dict)
        avg_h = average_code_length(code_h, prob_dict)
        results['huffman_avg_len'] = avg_h
    if do_sf:
        code_sf = build_shannon_fano_code(prob_dict)
        avg_sf = average_code_length(code_sf, prob_dict)
        results['shannon_fano_avg_len'] = avg_sf

    if plot and plt is not None:
        plt.figure()
        xs = extrap['block_sizes']
        ys = extrap['Hn']
        plt.plot(xs, ys, marker='o', label='H_n')
        plt.axhline(y=results['lz78_bits_per_sym'], linestyle='--', label='LZ78 est')
        plt.xlabel('Block size n')
        plt.ylabel('bits / symbol')
        plt.legend()
        plt.title('Block entropy and estimators')
        plt.show()

    return results


def run_audio_experiment(filename, downsample=4, quantize_bits=8, do_huffman=True, do_sf=True, plot=False):
    seq = wav_to_symbols(filename, downsample=downsample, quantize_bits=quantize_bits)
    prob_dict = probs_from_sequence(seq)
    H1 = entropy_from_probs(prob_dict)
    results = {'H1_bits': H1}
    extrap = estimate_entropy_rate_by_extrapolation(seq, max_block=6)
    results['Hn'] = extrap
    results['lz78_bits_per_sym'] = lz78_entropy_estimator(seq)
    if do_huffman:
        code_h = build_huffman_code(prob_dict)
        results['huffman_avg_len'] = average_code_length(code_h, prob_dict)
    if do_sf:
        code_sf = build_shannon_fano_code(prob_dict)
        results['shannon_fano_avg_len'] = average_code_length(code_sf, prob_dict)
    return results

# ---------------------------
# Simple self-test / demo
# ---------------------------

def demo_on_sample_text():
    sample = 'TOBEORNOTTOBEORTOBEORNOT'  # classic demo sequence
    print('\nSample:', sample)
    seq = list(sample)
    print('H(X) =', entropy_of_sequence(seq), 'bits')
    print('Block H2 =', block_entropy(seq, 2))
    print('LZ78 est =', lz78_entropy_estimator(seq))
    p = probs_from_sequence(seq)
    print('Huffman codes:', build_huffman_code(p))
    print('Shannon-Fano codes:', build_shannon_fano_code(p))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Entropy & Source Coding Simulator')
    parser.add_argument('--mode', choices=['text', 'audio', 'demo'], default='demo')
    parser.add_argument('--file', help='input filename for text or audio')
    parser.add_argument('--max-block', type=int, default=6)
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args()

    if args.mode == 'demo':
        demo_on_sample_text()
        sys.exit(0)
    if args.mode == 'text':
        if not args.file:
            print('Provide --file sample.txt')
            sys.exit(1)
        res = run_text_experiment(args.file, max_block=args.max_block, plot=args.plot)
        print('\nResults:')
        for k, v in res.items():
            print(k, ':', v)
    elif args.mode == 'audio':
        if not args.file:
            print('Provide --file sample.wav')
            sys.exit(1)
        res = run_audio_experiment(args.file, plot=args.plot)
        print('\nResults:')
        for k, v in res.items():
            print(k, ':', v)

# End of file
