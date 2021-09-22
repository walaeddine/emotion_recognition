# import the needed packages
from config import emotion_config as config
import numpy as np
import shutil
import cv2
import os


# remove the data file if exists
if (os.path.exists(config.DATA_PATH)):
    shutil.rmtree(config.DATA_PATH)
    print("[INFO] removing old data folder")

# create data dir
os.mkdir(config.DATA_PATH)
print(f"[INFO] new data folder created successfully")

# open the input file for reading (skipping the header), then
# initialize the list of data and labels for the training,
# validation, and testing sets
print("[INFO] loading input data...")
f = open(config.INPUT_PATH)
next(f)
count = 0

# loop over the rows in the input file
for row in f:
    count += 1
    # extract the label, image, and usage from the row
    (label, usage, image) = row.strip().split(",")
    label = int(label)

    # reshape the flattened pixel list into a 48x48 (grayscale)
    # image
    image = np.array(image.split(" "), dtype="uint8")
    image = image.reshape((48, 48))

    # check if we are examining a training image
    if usage == "Training":
        usagePath = config.TRAIN_PATH

    # check if this is a validation image
    elif usage == "PrivateTest":
        usagePath = config.VAL_PATH

    # otherwise, this must be a testing image
    else:
        usagePath = config.TEST_PATH

    # create usage path
    if not (os.path.exists(usagePath)):
        os.mkdir(usagePath)
        print(f"[INFO] usage folder {usagePath} created successfully")

    # create the label path
    labelPath = os.sep.join([usagePath, config.CLASS_NAMES[label]])

    # create usage path
    if not (os.path.exists(labelPath)):
        os.mkdir(labelPath)
        print(f"[INFO] label folder {labelPath} created successfully")

    imagePath = os.sep.join(
        [labelPath, str(config.CLASS_NAMES[label]) + str(count).zfill(6) + ".jpg"])
    
    cv2.imwrite(imagePath, image)

# close the input file
f.close()

print(f"\n[INFO] {count} saved to disk")
