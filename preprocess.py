import csv
import glob
import os
import string
import sys
from matplotlib.patches import Rectangle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.colors import ListedColormap
from tqdm import tqdm
from multiprocessing import Pool, Manager, Lock

CSV_FILE = './train.csv'
OUTPUT_DIR = './train'

def add_outline(ax, color='black', linewidth=2):
    rect = Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                     color=color, fill=False, linewidth=linewidth)
    ax.add_patch(rect)

def preprocess_data(npz_file):
    try:
        out_img = npz_file.replace(".npz", ".png")
        out_img = os.path.join(OUTPUT_DIR, out_img)
        os.makedirs(os.path.dirname(out_img), exist_ok=True)

        data = np.load(npz_file)
        gray_with_transparency = plt.cm.gray(np.linspace(0, 1, 256))
        gray_with_transparency[-1, -1] = 0
        transparent_cmap = ListedColormap(gray_with_transparency)

        fig = plt.figure(figsize=(5, 5), dpi=100)
        images = data['image']
        target = data['target']

        for i in range(8):
            ax = plt.subplot2grid((5, 4), ((i // 3)*1, i % 3), rowspan=1, colspan=1)
            ax.imshow(images[i], cmap=transparent_cmap)
            ax.axis("off")
            add_outline(ax, color='grey', linewidth=3)

        ax = plt.subplot2grid((5, 4), (2, 2), rowspan=1, colspan=1)
        ax.text(0.5, 0.4, "?", fontsize=60, ha="center", va="center")
        ax.axis("off")

        line = mlines.Line2D([0.05, 0.95], [0.41, 0.41], color='black', linewidth=2, transform=fig.transFigure, figure=fig)
        fig.add_artist(line)

        for i in range(8):
            ax = plt.subplot2grid((5, 4), (3 + (i // 4)*1, i % 4), rowspan=1, colspan=1)
            ax.imshow(images[8 + i], cmap=transparent_cmap)
            # choice_label = f"{i+1}"
            # ax.set_title(choice_label, fontsize=12, pad=0, loc='center')
            ax.text(0.5, -0.1, string.ascii_uppercase[i], fontsize=12, ha='center', va='center', transform=ax.transAxes)
            ax.axis("off")
            add_outline(ax, color='grey', linewidth=3)

        # plt.tight_layout()
        plt.subplots_adjust(top=0.95, bottom=0.05, left=0.05, right=0.95)
        plt.savefig(out_img, format='png')
        plt.close()

        return out_img, target
    except Exception as e:
        print(f"Error processing {npz_file}: {e}")
        return None, None

def process_and_write(args):
    npz_file, lock = args
    png_img, target = preprocess_data(npz_file)
    if png_img and target is not None:
        with lock:
            with open(CSV_FILE, mode='a', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow([png_img, string.ascii_uppercase[target]])

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python process_npz.py <directory>")
        sys.exit(1)

    directory = sys.argv[1]
    if os.path.exists(CSV_FILE):
        os.remove(CSV_FILE)

    if not os.path.isdir(directory):
        print(f"Error: {directory} is not a valid directory.")
        sys.exit(1)

    with open(CSV_FILE, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["filepath", "caption"])

    npz_files = glob.glob(f"{directory}/**/*.npz", recursive=True)

    # Use a Manager Lock to ensure thread-safe CSV writing
    with Manager() as manager:
        lock = manager.Lock()
        args = [(npz_file, lock) for npz_file in npz_files]

        # Use multiprocessing.Pool with max processes set to 32
        with Pool(processes=32) as pool:
            list(tqdm(pool.imap_unordered(process_and_write, args), total=len(npz_files), desc='Processing files'))
