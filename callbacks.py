class Callback:
    '''
    Base class for all callbacks. Callbacks are used to perform actions at specific points during training.
    '''
    def on_train_begin(self, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        pass

    def on_batch_end(self, batch, logs=None):
        pass

    def on_train_end(self, logs=None):
        pass

    def on_validation_batch_end(self, batch, logs=None):
        pass

    def on_validation_begin(self, logs=None):
        pass

    def on_validation_end(self, logs=None):
        pass

    def on_test_begin(self, logs=None):
        pass

    def on_test_end(self, logs=None):
        pass


from torch.utils.tensorboard import SummaryWriter

class TensorBoardCallback(Callback):
    def __init__(self, log_dir='./logs'):
        self.writer = SummaryWriter(log_dir)

    def on_epoch_end(self, epoch, logs=None):
        for key, value in logs.items():
            self.writer.add_scalar(key, value, epoch)
    
    def on_train_end(self, logs=None):
        self.writer.close()

import wandb

from callbacks import Callback

class WandBCallback(Callback):
    def __init__(self, project_name : str, run_name : str =None, config : dict = None):
        wandb.init(project=project_name, name=run_name, config=config)
    
    def on_train_end(self, logs=None):
        wandb.finish()

    def on_batch_end(self, batch, logs=None):
         wandb.log({**logs})

    def on_validation_end(self, logs=None,data=None):
        wandb.log(logs)
        wandb.log({"per class mIoU": wandb.Table(data=data)})

        
    

