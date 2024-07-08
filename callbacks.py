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
    def __init__(self, project_name : str, run_name : str =None, config : dict = None,note : str =''):
        self._wandb_ = wandb.init(project=project_name, name=run_name, config=config, notes=note)
    
    # def _loging_with_api_key_(self,api_key):
    #     self._wandb_ = wandb.login(key=api_key)
    
    def on_train_end(self, logs=None):
        print('The train finished completely and terminate the wandb logger.')
        self._wandb_.finish()

    def on_batch_end(self, batch, logs=None):
        self._wandb_.log({**logs})
            
    def on_epoch_end(self, epoch,logs=None):
        self._wandb_.log({**logs})

    def on_validation_end(self, logs=None,data=None):
        self._wandb_.log(logs)
        self._wandb_.log({"per class mIoU": wandb.Table(data=data)})

        
    

