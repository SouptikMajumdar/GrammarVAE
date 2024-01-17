from box import Box
import yaml

class HyperparameterOptimization():
    def __init__(self,path):
        self.path = path
        self.params = None        
    
    def get_params(self):
        with open(self.path) as f:
            self.params = yaml.safe_load(f)
            self.params = Box(self.params)
        return self.params

# if __name__ == '__main__':
#     hopt = Hyperparam()

