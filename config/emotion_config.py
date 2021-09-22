from os import path
from os import getcwd
INPUT_PATH = "icml_face_data.csv"

BASE_PATH = getcwd()

DATA_PATH = path.sep.join([BASE_PATH, "dataset"])
TRAIN_PATH = path.sep.join([DATA_PATH, "train"])
VAL_PATH = path.sep.join([DATA_PATH, "validation"])
TEST_PATH = path.sep.join([DATA_PATH, "test"])


OUTPUT_PATH = path.sep.join([BASE_PATH, "output"])
CHECKPOINTS_PATH = path.sep.join([BASE_PATH, "checkpoints"])

CLASS_NAMES = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

IMG_HEIGHT = 48
IMG_WIDTH = 48
NUM_CLASSES = 7

# define input image spatial dimensions
IMAGE_SIZE = (IMG_HEIGHT, IMG_WIDTH)

# initialize our number of epochs, early stopping patience, initial
# learning rate, and batch size
NUM_EPOCHS = 80
EARLY_STOPPING_PATIENCE = 5
INIT_LR = 1e-3
BS = 64