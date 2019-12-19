import os

CODE_ROOT = os.path.dirname(os.path.realpath(__file__))
ANNOTATIONS_PATH = os.path.join(os.path.dirname(CODE_ROOT), "preprocessing")

HOME = os.environ["HOME"]
DATA_PATH = os.path.join(HOME, "data", "TMLGA")
ANET_FEATURES_PATH = DATA_PATH
CHARADES_FEATURES_PATH = os.path.join(DATA_PATH, "i3d_charades_sta")
EMBEDDINGS_PATH = os.path.join(DATA_PATH, "word_embeddings")
