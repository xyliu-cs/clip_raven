import csv
import glob
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from tqdm import tqdm

CSV_FILE='./train.csv'
OUTPUT_DIR='./train'

def preprocess_data(npz_file):
    out_img = npz_file.replace(".npz", ".png")
    out_img = os.path.join(OUTPUT_DIR, out_img)
    if not os.path.exists(out_img):
        os.makedirs(os.path.dirname(out_img), exist_ok=True)
    # Load the NPZ file
    data = np.load(npz_file)

    # Create a custom colormap with transparency for white
    gray_with_transparency = plt.cm.gray(np.linspace(0, 1, 256))
    gray_with_transparency[-1, -1] = 0  # Set alpha value of white color to 0 (fully transparent)
    transparent_cmap = ListedColormap(gray_with_transparency)

    # Create a new figure for visualization
    fig = plt.figure(figsize=(12, 12))
    images = data['image']  # Shape: (16, 160, 160) or fewer images
    target = data['target']  # Index of the correct choice

    # Plot the problem matrix (3x3 grid, last cell is a question mark)
    for i in range(8):
        ax = plt.subplot2grid((8, 4), ((i // 3)*2, i % 3), rowspan=2, colspan=1)  # Larger space for the problem matrix
        ax.imshow(images[i], cmap=transparent_cmap)  # Use the transparent colormap
        ax.axis("off")

    # Add the question mark in the last cell
    ax = plt.subplot2grid((8, 4), (4, 2), rowspan=2, colspan=1)
    ax.text(0.5, 0.4, "?", fontsize=80, ha="center", va="center")
    ax.axis("off")

    # Plot the answer choices (4x2 grid with numbers, label the correct answer)
    for i in range(8):
        ax = plt.subplot2grid((8, 4), (6 + i // 4, i % 4), rowspan=1, colspan=1)  # Answer choices grid
        ax.imshow(images[8 + i], cmap=transparent_cmap)  # Use the transparent colormap
        choice_label = f"{i+1}"
        ax.set_title(choice_label, fontsize=12, pad=0, loc='center')
        ax.axis("off")

    # Add a super-title
    #plt.suptitle("Problem Matrix (3x3) and Answer Choices (4x2)", fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.95, bottom=0.05, left=0.05, right=0.95) 

    # plt.show()
    plt.savefig(out_img, format='png')
    plt.close()
    return out_img, target

def add_entry(out_img, target):
    with open(CSV_FILE, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
    
        # Check if the file is empty to write a header
        if os.stat(CSV_FILE).st_size == 0:
            writer.writerow(["filepath", "caption"])  # Add a header row
    
        # Append the new row
        writer.writerow([out_img, target])

if __name__ == "__main__":
    # Ensure the script is called with a directory argument
    if len(sys.argv) != 2:
        print("Usage: python process_npz.py <directory>")
        sys.exit(1)

    # Get the directory from command-line arguments
    directory = sys.argv[1]

    if os.path.exists(CSV_FILE):
        # Remove the file
        os.remove(CSV_FILE)

    # Check if the provided path is a valid directory
    if not os.path.isdir(directory):
        print(f"Error: {directory} is not a valid directory.")
        sys.exit(1)

    # Iterate over all .npz files and process them
    for npz_file in tqdm(glob.glob(f"{directory}/**/*.npz", recursive=True), desc='Processing files'):
        png_img, target = preprocess_data(npz_file)
        add_entry(png_img, target)