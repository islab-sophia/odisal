import torch
import os

def save_checkpoint(state, is_best, save_dir, save_step=1):
    if ((state['epoch']+1)%save_step==0)or((state['epoch']+1)<save_step):
        print("SAVE CHECKPOINT")
        filename = 'checkpoint_{}_epoch{}.pth.tar'.format(state['start_time_stamp'], state['epoch'])
        torch.save(state, os.path.join(save_dir, filename))
    filename_isbest = 'checkpoint_{}_model_best.pth.tar'.format(state['start_time_stamp'])
    if is_best:
        torch.save(state, os.path.join(save_dir, filename_isbest))
