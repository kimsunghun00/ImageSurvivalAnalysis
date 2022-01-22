import torch
from torch.utils.data import Dataset, DataLoader
import tqdm
import copy
import time
from utils.utils import lifelines_cindex

def model_eval(model, data_loader, device):
    model.eval()
    
    survtime_true = []
    survtime_preds = []
    censors = []
       
     # Iterate over data
    for i_batch, sample_batch in tqdm.tqdm_notebook(enumerate(data_loader), total = len(data_loader)):
        image_batch, censor_batch, survtime_batch = sample_batch['image'], sample_batch['censor'], sample_batch['survtime']
        
        image_batch = image_batch.to(device, dtype = torch.float) # [batch, 5, 3, 512, 512]
        survtime_batch = survtime_batch.to(device, dtype = torch.float)
        censor_batch = censor_batch.to(device, dtype = torch.float)
        
        with torch.no_grad():
            survtime_preds_batch = model(image_batch)
        
        survtime_preds.append(survtime_preds_batch)
        survtime_true.append(survtime_batch)
        censors.append(censor_batch)
    
    survtime_true = torch.cat(survtime_true)
    survtime_preds = torch.cat(survtime_preds)
    censors = torch.cat(censors)
    
    
    test_cindex = lifelines_cindex(survtime_preds, censors, survtime_true)
    print('c-index: {:.4f}'.format(test_cindex))
    
    return survtime_preds, survtime_true
