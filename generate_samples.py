"""
generate_samples.py

Automatically creates sample.txt and sample.wav for the Entropy & Source Coding project.
"""

import numpy as np
from scipy.io.wavfile import write

# --- Create sample.txt ---
text_content = """
This is a simple example text for testing entropy and source coding simulations.
It contains repeated characters and words to produce meaningful symbol probabilities.
"""
with open('sample.txt', 'w', encoding='utf-8') as f:
    f.write(text_content.strip())

print('sample.txt created successfully.')

# --- Create sample.wav ---
fs = 8000  # sampling rate in Hz
duration = 1.0  # 1 second
t = np.linspace(0, duration, int(fs*duration), endpoint=False)
frequency = 440  # A4 note in Hz
signal = 0.5 * np.sin(2 * np.pi * frequency * t)  # sine wave
signal_int16 = np.int16(signal * 32767)  # convert to 16-bit PCM

write('sample.wav', fs, signal_int16)
print('sample.wav created successfully.')
