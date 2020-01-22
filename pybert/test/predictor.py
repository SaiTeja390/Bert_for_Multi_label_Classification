#encoding:utf-8
import torch
import numpy as np
from ..common.tools import model_device
from ..callback.progressbar import ProgressBar

class Predictor(object):
    def __init__(self,
                 model,
                 logger,
                 n_gpu,
                 test=False):
        self.model = model
        self.logger = logger
        self.model, self.device = model_device(n_gpu= n_gpu, model=self.model)
        self.test = test

    def predict(self,data):
        pbar = ProgressBar(n_total=len(data))
        if self.test: 
            all_logits = []
        else:
            all_logits = None
        self.model.eval()
        targets = []
        with torch.no_grad():
            if self.test:
                for step, batch in enumerate(data):
                    batch = tuple(t.to(self.device) for t in batch)
                    input_ids, input_mask, segment_ids, label_ids = batch
                    # logits = self.model(input_ids, segment_ids, input_mask)
                    logits, h = self.model(input_ids, segment_ids, input_mask, hidden_state = None)
                    all_logits.append(logits.cpu().detach())
                    targets.append(label_ids.cpu().detach())
                #print(f'targets {targets}')                            
                    pbar.batch_step(step=step,info = {},bar_type='Evaluating')
                #print(logits)  
                #targets.append(label_ids.cpu().detach()) 
#                if all_logits is None:
#                    all_logits = logits.detach().cpu().numpy()
#                else:
#                    all_logits = np.concatenate([all_logits,logits.detach().cpu().numpy()],axis = 0)
                    pbar.batch_step(step=step,info = {},bar_type='Testing')
                all_logits = torch.cat(all_logits, dim = 0).cpu().detach()
                targets = torch.cat(targets, dim = 0).cpu().detach()

                if 'cuda' in str(self.device):
                    torch.cuda.empty_cache()
                return all_logits,targets

            else:
                for step, batch in enumerate(data):
                       batch = tuple(t.to(self.device) for t in batch)
                       input_ids, input_mask, segment_ids, label_ids = batch
                       logits, h = self.model(input_ids, segment_ids, input_mask, hidden_state = None)
                      #  logits = self.model(input_ids, segment_ids, input_mask)
                       logits = logits.sigmoid()
                       print(f'logits.shape {logits.shape}')
                       if all_logits is None:
                           all_logits = logits.detach().cpu().numpy()
                       else:
                           all_logits = np.concatenate([all_logits,logits.detach().cpu().numpy()],axis = 0)
                       pbar.batch_step(step=step,info = {},bar_type='Testing')
                if 'cuda' in str(self.device):
                    torch.cuda.empty_cache()
                return all_logits