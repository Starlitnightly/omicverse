import torch
import numpy as np
import pandas as pd
import os
import torch.nn as nn


from .neural_net import get_loss

# def process_files(output_folder):
#     smallest_loss = np.Inf
#     best_model_folder_path = None
#     best_mod=None

#     for folder_name in os.listdir(output_folder):
#         folder_path = os.path.join(output_folder, folder_name)

#         if os.path.isdir(folder_path):
#             model_path = os.path.join(folder_path, 'final_model.pt')

#             if os.path.exists(model_path):
#                 try:
#                     mod =  torch.load(model_path)
#                     St=torch.load(os.path.join(folder_path, 'Storch.pt'))
#                     At=torch.load(os.path.join(folder_path, 'Atorch.pt'))
#                     loss = get_loss(mod,St,At)

#                     if loss < smallest_loss:
#                         smallest_loss = loss
#                         best_model_folder_path = folder_path
#                         best_mod=mod
#                 except Exception as e:
#                     raise Exception(f"Error loading model from {model_path}: {str(e)}")
#     print(f'best model: {best_model_folder_path}')
#     if best_model_folder_path:
#         # folder_name = os.path.basename(os.path.dirname(best_model_path))
#         storch_path = os.path.join(best_model_folder_path, 'Storch.pt')
#         atorch_path = os.path.join(best_model_folder_path, 'Atorch.pt')

#         if os.path.exists(storch_path) and os.path.exists(atorch_path):
#             A_torch = torch.load(atorch_path)
#             S_torch = torch.load(storch_path)
            
#             A = A_torch.detach().numpy()
#             S = S_torch.detach().numpy()

#     else:
#         raise Exception("No 'final_model.pt' found in any folder.")

#     return best_mod, A, S

def process_files(output_folder, output_torch=False, epoch_number='final', seed_list=None):
    smallest_loss = np.inf
    best_model_folder_path = None
    best_mod=None

    # Check PyTorch version
    if torch.__version__>='2.6.0':
        is_torch_26_or_later = True
    else:
        is_torch_26_or_later = False

    # only look at specific seeds
    if seed_list is None:
        folder_list=os.listdir(output_folder)
    else:
        folder_list=[f'seed{i}' for i in seed_list]

    for folder_name in folder_list:
        folder_path = os.path.join(output_folder, folder_name)
        if os.path.isdir(folder_path) and 'Storch.pt' in os.listdir(folder_path) and 'Atorch.pt' in os.listdir(folder_path):
            St=torch.load(os.path.join(folder_path, 'Storch.pt'))
            At=torch.load(os.path.join(folder_path, 'Atorch.pt'))

            # check if final_model exists
            if epoch_number!='final':
                final_model_name=f'model_epoch_{epoch_number}.pt'
            else:
                final_model_name='final_model.pt'
                
            final_model_path=os.path.join(folder_path, final_model_name)
            if os.path.exists(final_model_path):
                model_path=final_model_path
            else:
                # find highest epoch model and load
                highest_epoch=-np.inf
                for filename in os.listdir(folder_path):
                    if "model_epoch_" in filename:
                        # Extract the epoch number from the filename
                        epoch_num = int(filename.split('_')[-1][:-3])
                        # Update the highest_epoch and highest_epoch_file if this file has a higher epoch
                        if epoch_num > highest_epoch:
                            highest_epoch = epoch_num
                            highest_epoch_file = filename
                model_path=os.path.join(folder_path, highest_epoch_file)
        
            # Load model based on PyTorch version
            if is_torch_26_or_later:
                try:
                    # First try with weights_only=False
                    mod = torch.load(model_path, weights_only=False)
                except Exception as e:
                    # If that fails, try with safe_globals
                    from torch.serialization import safe_globals
                    from ..gaston.neural_net import GASTON
                    with safe_globals([GASTON]):
                        mod = torch.load(model_path)
            else:
                mod = torch.load(model_path)

            loss = get_loss(mod,St,At)

            # compare against other models with different seeds
            if loss < smallest_loss:
                smallest_loss = loss
                best_model_folder_path = folder_path
                best_mod=mod
                
    print(f'\nbest model: {best_model_folder_path}')
    if best_model_folder_path:
        storch_path = os.path.join(best_model_folder_path, 'Storch.pt')
        atorch_path = os.path.join(best_model_folder_path, 'Atorch.pt')

        if os.path.exists(storch_path) and os.path.exists(atorch_path):
            A_torch = torch.load(atorch_path)
            S_torch = torch.load(storch_path)

            A = A_torch.detach().numpy()
            S = S_torch.detach().numpy()

    else:
        raise Exception("No model found in any folder.")

    if output_torch:
        return best_mod, A, S, A_torch, S_torch
    else:
        return best_mod, A, S

def create_cell_type_df(ct_labels):
    ct_list=np.unique(ct_labels)
    ct_arr=np.zeros( (len(ct_labels), len(ct_list) ))
    
    for i,ct in enumerate(ct_labels):
        ct_arr[i, ct_list==ct]=1
    cell_type_df=pd.DataFrame(ct_arr,columns=ct_list)
    return cell_type_df
