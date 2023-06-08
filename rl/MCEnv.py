from simso.core.Model import Model

class MCEnv:
    def __init__(self):
        pass

    def step(self, action):
        pass

    def reset(self, configuration):
        self.model = Model(configuration)