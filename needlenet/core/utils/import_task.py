import importlib

def load_model_builder(model_name):
    model_builder = importlib.import_module(f'needlenet.models.{model_name}.model')
    return model_builder

def load_dataset_builder(dataset_name):
    dataset_builder = importlib.import_module(f'needlenet.data.{dataset_name}.dataset')
    return dataset_builder
