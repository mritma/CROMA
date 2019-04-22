class config():
    def __init__(self):
        self.classifier_lr = 0.0001
        self.rl_lr = 0.0002
        self.rl_epoch = 10
        self.classifier_training_percent = 1
        self.is_stop = False
        self.entropy_weight = 0
        self.anneal_weight = 1
        self.decay_period = 10


class fast_config():
    def __init__(self):
        self.classifier_lr = 0.001
        self.rl_lr = 0.001
        self.rl_epoch = 15
        self.classifier_training_percent = 1
        self.is_stop = True
        self.entropy_weight = 0
        self.anneal_weight = 1
        self.decay_period = 10
