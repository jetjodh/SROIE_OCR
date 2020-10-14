""""
Creates the dataset for the recogninition task.
"""
# Importing necessary libs
import glob
import os
from tqdm import tqdm
import cv2

# Declaring output folder
output_fold = "train/"


def filer(dir, ext) -> list:
    """
    Params:
    dir: Directory where files are kept
    ext: File extensions which are to be searched
    Returns:
    List of filenames
    """
    return glob.glob(dir + "/" + "*." + ext)


def seperate_bbox(txt_file, jpg_file, cnt) -> int:
    """
    Params:
    txt_file: Name of text file containing bbox coords and text contained
    jpg_file: Name of corresponding image file
    Returns:
    counter for updating it for use by other function calls
    """
    img = cv2.imread(jpg_file)

    with open(txt_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            cnt += 1
            elements = line.split(",")
            x1 = int(elements[0])
            y1 = int(elements[1])
            x2 = int(elements[4])
            y2 = int(elements[5])
            text = elements[8]

            crop = img[y1:y2, x1:x2]

            try:
                cv2.imwrite(output_fold + str(cnt) + ".jpg", crop)

                with open("gt.txt", "a") as fp:
                    fp.writelines(output_fold + str(cnt) + ".jpg\t" + text)
            except:
                print(jpg_file)
    return cnt


def main():
    """
    Runs the main processing pipeline
    """
    txt_files = sorted(filer(r"D:\qubitrics\0325updated.task1train(626p)", "txt"))
    jpg_files = sorted(filer(r"D:\qubitrics\0325updated.task1train(626p)", "jpg"))
    print(len(txt_files))
    print(len(jpg_files))
    counter = 0
    for i in tqdm(range(len(txt_files))):
        counter = seperate_bbox(txt_files[i], jpg_files[i], counter)


if __name__ == "__main__":
    main()
