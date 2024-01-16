import mlflow
import os

class MLFlowTracker:
    def __init__(self, ename):
        self.ename = ename

    def set_experiment(self):
        mlflow.set_experiment(self.ename)


    def log_parameters(self, train_cfg, model_cfg, mask_cfg, dataset_cfg):
        mlflow.log_param('Train Cfg', train_cfg)
        mlflow.log_param('Model Cfg', model_cfg)
        mlflow.log_param('Mask Cfg', mask_cfg)
        mlflow.log_param('Dataset Cfg', dataset_cfg)


    def log_metrics(self, name, metric):
        mlflow.log_metric(name, metric)


    def log_artifact(self, path):
        mlflow.log_artifact(path)

    
    def log_model(self, model, name):
        mlflow.pytorch.log_model(model, name)

