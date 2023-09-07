class MeterStorage(object):
    def add(self, name):
        setattr(self, name, AverageMeter())

    def reset(self, name):
        getattr(self, name).reset()

    def update(self, name, val, n=1):
        getattr(self, name).update(val, n)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
