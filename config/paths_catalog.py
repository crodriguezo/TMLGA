import os

class DatasetCatalog(object):
    DATA_DIR = "datasets"
    DATASETS = {
        "anet_cap_train" : {
            "feature_path": "/home/crodriguezo/ANet/training/",
            "ann_file_path" : "/home/crodriguezo/TMLGA/preprocessing/anet-cap/ANetCap_training_tokens.json",
            "embeddings_path" : "/home/crodriguezo//glove.840B.300d/glove.840B.300d.txt",
            },
        "anet_cap_test" : {
            "feature_path": "/home/crodriguezo/ANet/validation/",
            "ann_file_path" : "/home/crodriguezo/TMLGA/preprocessing/anet-cap/ANetCap_validation_tokens.json",
            "embeddings_path" : "/home/crodriguezo//glove.840B.300d/glove.840B.300d.txt",
            },

        "charades_sta_train" : {
            "feature_path": "/home/crodriguezo/charades-sta/rgb/",
            "ann_file_path" : "/home/crodriguezo/TMLGA/preprocessing/charades-sta/charades_sta_train_tokens.json",
            "embeddings_path" : "/home/crodriguezo/glove.840B.300d/glove.840B.300d.txt",
            },

        "charades_sta_test" : {
            "feature_path": "/home/crodriguezo/charades-sta/rgb/",
            "ann_file_path" : "/home/crodriguezo/TMLGA/preprocessing/charades-sta/charades_sta_test_tokens.json",
            "embeddings_path" : "/home/crodriguezo/glove.840B.300d/glove.840B.300d.txt",
            },
    }

    @staticmethod
    def get(name):
        if "charades_sta" in name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                features_path=os.path.join(data_dir, attrs["feature_path"]),
                ann_file_path=os.path.join(data_dir, attrs["ann_file_path"]),
                embeddings_path=os.path.join(data_dir, attrs["embeddings_path"]),
            )
            return dict(
                factory="CHARADES_STA",
                args=args,
            )
        if "anet_cap" in name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                features_path=os.path.join(data_dir, attrs["feature_path"]),
                ann_file_path=os.path.join(data_dir, attrs["ann_file_path"]),
                embeddings_path=os.path.join(data_dir, attrs["embeddings_path"]),
            )
            return dict(
                factory="ANET_CAP",
                args=args,
            )
        raise RuntimeError("Dataset not available: {}".format(name))
