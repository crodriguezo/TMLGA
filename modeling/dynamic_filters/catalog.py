import os

class DynamicFilterCatalog(object):

    @staticmethod
    def get(name):
        if "LSTM" in name:
            return dict(
                factory="LSTM",
                args=args,
            )
        raise RuntimeError("DynamicFilterGen not available: {}".format(name))
