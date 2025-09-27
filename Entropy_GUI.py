# Entropy_GUI.py
# GUI for Entropy Calculator & Source Coding Simulator

import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk

import matplotlib.pyplot as plt

from Entropy_InfoRate_Simulator import (
    run_text_experiment,
    run_audio_experiment
)

class EntropyApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Entropy & Source Coding Simulator")
        self.root.geometry("600x500")

        # Mode selection
        self.mode_var = tk.StringVar(value="text")
        tk.Label(root, text="Mode:").pack(anchor='w', padx=10, pady=5)
        ttk.Radiobutton(root, text="Text", variable=self.mode_var, value="text").pack(anchor='w', padx=20)
        ttk.Radiobutton(root, text="Audio", variable=self.mode_var, value="audio").pack(anchor='w', padx=20)

        # File selection
        tk.Label(root, text="Select File:").pack(anchor='w', padx=10, pady=5)
        file_frame = tk.Frame(root)
        file_frame.pack(fill='x', padx=20)
        self.file_entry = tk.Entry(file_frame)
        self.file_entry.pack(side='left', fill='x', expand=True)
        tk.Button(file_frame, text="Browse", command=self.browse_file).pack(side='left', padx=5)

        # Max block size
        tk.Label(root, text="Max Block Size:").pack(anchor='w', padx=10, pady=5)
        self.max_block_var = tk.IntVar(value=6)
        tk.Entry(root, textvariable=self.max_block_var).pack(anchor='w', padx=20)

        # Huffman / Shannon-Fano toggles
        self.huffman_var = tk.BooleanVar(value=True)
        self.sf_var = tk.BooleanVar(value=True)
        tk.Checkbutton(root, text="Use Huffman Coding", variable=self.huffman_var).pack(anchor='w', padx=20, pady=2)
        tk.Checkbutton(root, text="Use Shannon-Fano Coding", variable=self.sf_var).pack(anchor='w', padx=20, pady=2)

        # Run button
        tk.Button(root, text="Run Experiment", command=self.run_experiment, bg='lightblue').pack(pady=10)

        # Results display
        tk.Label(root, text="Results:").pack(anchor='w', padx=10, pady=5)
        self.results_text = tk.Text(root, height=15)
        self.results_text.pack(fill='both', padx=20, pady=5, expand=True)

    def browse_file(self):
        if self.mode_var.get() == "text":
            filename = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
        else:
            filename = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])
        if filename:
            self.file_entry.delete(0, tk.END)
            self.file_entry.insert(0, filename)

    def run_experiment(self):
        filename = self.file_entry.get()
        if not filename:
            messagebox.showerror("Error", "Please select a file!")
            return

        max_block = self.max_block_var.get()
        do_huffman = self.huffman_var.get()
        do_sf = self.sf_var.get()
        self.results_text.delete(1.0, tk.END)

        mode = self.mode_var.get()
        try:
            if mode == "text":
                results = run_text_experiment(filename, max_block=max_block,
                                              do_huffman=do_huffman, do_sf=do_sf, plot=False)
            else:
                results = run_audio_experiment(filename, do_huffman=do_huffman, do_sf=do_sf, plot=False)

            # Display results
            for k, v in results.items():
                self.results_text.insert(tk.END, f"{k}: {v}\n")

            # Optional: plot block entropy if available
            if "Hn" in results and results["Hn"] is not None:
                Hn = results["Hn"]
                plt.figure()
                plt.plot(Hn['block_sizes'], Hn['Hn'], marker='o', label='H_n')
                if 'lz78_bits_per_sym' in results:
                    plt.axhline(y=results['lz78_bits_per_sym'], linestyle='--', label='LZ78 est')
                plt.xlabel("Block size n")
                plt.ylabel("Bits per symbol")
                plt.title("Block Entropy and Estimators")
                plt.legend()
                plt.show()

        except Exception as e:
            messagebox.showerror("Error", str(e))


if __name__ == "__main__":
    root = tk.Tk()
    app = EntropyApp(root)
    root.mainloop()
