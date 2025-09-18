import torch as th
import torch.nn as nn

def latent_time_path_reg(z_traj, z_data, ts, ts_data, hinge_value=0.025):
    
    vals = []
    for i in range(z_traj.shape[1]): # loop over time

        _, indices_ts = th.topk(th.cdist(ts[None,i], ts_data[None])[0], 100, 
                                dim=1, largest=False, sorted=True)
        
        cdist, indices = th.topk(th.cdist(z_traj[:,[i]].permute(1,0,2), z_data[indices_ts]), 1,
                                 dim=2, largest=False, sorted=False)
        
        vals.append(cdist[0])
    values = th.cat(vals, dim=1)
    
    return th.sum(nn.functional.relu(values - hinge_value), dim=-1)
