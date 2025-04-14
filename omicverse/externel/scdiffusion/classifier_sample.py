"""
Like image_sample.py, but use a noisy image classifier to guide the sampling
process towards more realistic images.
"""

import argparse

import numpy as np
import torch as th
import torch.distributed as dist
import torch.nn.functional as F

from .guided_diffusion import dist_util, logger
from .guided_diffusion.script_util import (   
    NUM_CLASSES,
    model_and_diffusion_defaults,
    classifier_and_diffusion_defaults,
    create_model_and_diffusion,
    create_classifier,
    add_dict_to_argparser,
    args_to_dict,
)
import scanpy as sc
import torch
from VAE.VAE_model import VAE

def load_VAE(ae_dir, num_gene):
    autoencoder = VAE(
        num_genes=num_gene,
        device='cuda',
        seed=0,
        hidden_dim=128,
        decoder_activation='ReLU',
    )
    autoencoder.load_state_dict(torch.load(ae_dir))
    return autoencoder

def save_data(all_cells, traj, data_dir):
    cell_gen = all_cells
    np.savez(data_dir, cell_gen=cell_gen)
    return

def main(cell_type=[0], multi=False, inter=False, weight=[10,10]):
    args = create_argparser(cell_type, weight).parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    model.eval()

    logger.log("loading classifier...")
    if multi:
        args.num_class = args.num_class1 # how many classes in this condition
        classifier1 = create_classifier(**args_to_dict(args, (['num_class']+list(classifier_and_diffusion_defaults().keys()))[:3]))
        classifier1.load_state_dict(
            dist_util.load_state_dict(args.classifier_path1, map_location="cpu")
        )
        classifier1.to(dist_util.dev())
        classifier1.eval()

        args.num_class = args.num_class2 # how many classes in this condition
        classifier2 = create_classifier(**args_to_dict(args, (['num_class']+list(classifier_and_diffusion_defaults().keys()))[:3]))
        classifier2.load_state_dict(
            dist_util.load_state_dict(args.classifier_path2, map_location="cpu")
        )
        classifier2.to(dist_util.dev())
        classifier2.eval()

    else:
        classifier = create_classifier(**args_to_dict(args, (['num_class']+list(classifier_and_diffusion_defaults().keys()))[:3]))
        classifier.load_state_dict(
            dist_util.load_state_dict(args.classifier_path, map_location="cpu")
        )
        classifier.to(dist_util.dev())
        classifier.eval()

    '''
    control function for Gradient Interpolation Strategy
    '''
    def cond_fn_inter(x, t, y=None, init=None, diffusion=None):
        assert y is not None
        y1 = y[:,0]
        y2 = y[:,1]
        # xt = diffusion.q_sample(th.tensor(init,device=dist_util.dev()),t*th.ones(init.shape[0],device=dist_util.dev(),dtype=torch.long),)
        with th.enable_grad():
            x_in = x.detach().requires_grad_(True)
            logits = classifier(x_in, t)
            log_probs = F.log_softmax(logits, dim=-1)
            selected1 = log_probs[range(len(logits)), y1.view(-1)]
            selected2 = log_probs[range(len(logits)), y2.view(-1)]
            
            grad1 = th.autograd.grad(selected1.sum(), x_in, retain_graph=True)[0] * args.classifier_scale1
            grad2 = th.autograd.grad(selected2.sum(), x_in, retain_graph=True)[0] * args.classifier_scale2

            # l2_loss = ((x_in-xt)**2).mean()
            # grad3 = th.autograd.grad(-l2_loss, x_in, retain_graph=True)[0] * 100

            return grad1+grad2#+grad3

    '''
    control function for multi-conditional generation
    Two conditional generation here
    '''
    def cond_fn_multi(x, t, y=None):
        assert y is not None
        y1 = y[:,0]
        y2 = y[:,1]
        with th.enable_grad():
            x_in = x.detach().requires_grad_(True)
            logits1 = classifier1(x_in, t)
            log_probs1 = F.log_softmax(logits1, dim=-1)
            selected1 = log_probs1[range(len(logits1)), y1.view(-1)]

            logits2 = classifier2(x_in, t)
            log_probs2 = F.log_softmax(logits2, dim=-1)
            selected2 = log_probs2[range(len(logits2)), y2.view(-1)]
            
            grad1 = th.autograd.grad(selected1.sum(), x_in, retain_graph=True)[0] * args.classifier_scale1
            grad2 = th.autograd.grad(selected2.sum(), x_in, retain_graph=True)[0] * args.classifier_scale2
            
            return grad1+grad2

    '''
    control function for one conditional generation
    '''
    def cond_fn_ori(x, t, y=None):
        assert y is not None
        with th.enable_grad():
            x_in = x.detach().requires_grad_(True)
            logits = classifier(x_in, t)
            log_probs = F.log_softmax(logits, dim=-1)
            selected = log_probs[range(len(logits)), y.view(-1)]
            grad = th.autograd.grad(selected.sum(), x_in, retain_graph=True)[0] * args.classifier_scale
            return grad
        
    def model_fn(x, t, y=None, init=None, diffusion=None):
        assert y is not None
        if args.class_cond:
            return model(x, t, y if args.class_cond else None)
        else:
            return model(x, t)
        
    if inter:
        # input real cell expression data as initial noise
        ori_adata = sc.read_h5ad(args.init_cell_path)
        sc.pp.normalize_total(ori_adata, target_sum=1e4)
        sc.pp.log1p(ori_adata)

    logger.log("sampling...")
    all_cell = []
    sample_num = 0
    while sample_num < args.num_samples:
        model_kwargs = {}

        if not multi and not inter:
            classes = (cell_type[0])*th.ones((args.batch_size,), device=dist_util.dev(), dtype=th.long)

        if multi:
            classes1 = (cell_type[0])*th.ones((args.batch_size,), device=dist_util.dev(), dtype=th.long)
            classes2 = (cell_type[1])*th.ones((args.batch_size,), device=dist_util.dev(), dtype=th.long)
            # classes3 = ... if more conditions
            classes = th.stack((classes1,classes2), dim=1)

        if inter:
            classes1 = (cell_type[0])*th.ones((args.batch_size,), device=dist_util.dev(), dtype=th.long)
            classes2 = (cell_type[1])*th.ones((args.batch_size,), device=dist_util.dev(), dtype=th.long)
            classes = th.stack((classes1,classes2), dim=1)

        model_kwargs["y"] = classes
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )

        if inter:
            celltype = ori_adata.obs['period'].cat.categories.tolist()[cell_type[0]]
            adata = ori_adata[ori_adata.obs['period']==celltype].copy()

            start_x = adata.X
            autoencoder = load_VAE(args.ae_dir, args.num_gene)
            start_x = autoencoder(torch.tensor(start_x,device=dist_util.dev()),return_latent=True).detach().cpu().numpy()

            n, m = start_x.shape  
            if n >= args.batch_size:  
                start_x = start_x[:args.batch_size, :]  
            else:  
                repeat_times = args.batch_size // n  
                remainder = args.batch_size % n  
                start_x = np.concatenate([start_x] * repeat_times + [start_x[:remainder, :]], axis=0)  
            
            noise = diffusion.q_sample(th.tensor(start_x,device=dist_util.dev()),args.init_time*th.ones(start_x.shape[0],device=dist_util.dev(),dtype=torch.long),)
            model_kwargs["init"] = start_x
            model_kwargs["diffusion"] = diffusion

        if multi:
            sample, traj = sample_fn(
                model_fn,
                (args.batch_size, args.input_dim),
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
                cond_fn=cond_fn_multi,
                device=dist_util.dev(),
                noise = None,
                start_time=diffusion.betas.shape[0],
                start_guide_steps=args.start_guide_steps,
            )
        elif inter:
            sample, traj = sample_fn(
                model_fn,
                (args.batch_size, args.input_dim),
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
                cond_fn=cond_fn_inter,
                device=dist_util.dev(),
                noise = noise,
                start_time=diffusion.betas.shape[0],
                start_guide_steps=args.start_guide_steps,
            )
        else:
            sample, traj = sample_fn(
                model_fn,
                (args.batch_size, args.input_dim),
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
                cond_fn=cond_fn_ori,
                device=dist_util.dev(),
                noise = None,
            )

        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        if args.filter:
            for sample in gathered_samples:
                if multi:
                    logits1 = classifier1(sample, torch.zeros((sample.shape[0]), device=sample.device))
                    logits2 = classifier2(sample, torch.zeros((sample.shape[0]), device=sample.device))
                    prob1 = F.softmax(logits1, dim=-1)
                    prob2 = F.softmax(logits2, dim=-1)
                    type1 = torch.argmax(prob1, 1)
                    type2 = torch.argmax(prob2, 1)
                    select_index = ((type1 == cell_type[0]) & (type2 == cell_type[1]))
                    all_cell.extend([sample[select_index].cpu().numpy()])
                    sample_num += select_index.sum().item()
                elif inter:
                    logits = classifier(sample, torch.zeros((sample.shape[0]), device=sample.device))
                    prob = F.softmax(logits, dim=-1)
                    left = (prob[:,cell_type[0]] > weight[0]/10-0.15) & (prob[:,cell_type[0]] < weight[0]/10+0.15)
                    right = (prob[:,cell_type[1]] > weight[1]/10-0.15) & (prob[:,cell_type[1]] < weight[1]/10+0.15)
                    select_index = left & right
                    all_cell.extend([sample[select_index].cpu().numpy()])
                    sample_num += select_index.sum().item()
                else:
                    logits = classifier(sample, torch.zeros((sample.shape[0]), device=sample.device))
                    prob = F.softmax(logits, dim=-1)
                    type = torch.argmax(prob, 1)
                    select_index = (type == cell_type[0])
                    all_cell.extend([sample[select_index].cpu().numpy()])
                    sample_num += select_index.sum().item()
            logger.log(f"created {sample_num} samples")
        else:
            all_cell.extend([sample.cpu().numpy() for sample in gathered_samples])
            sample_num = len(all_cell) * args.batch_size
            logger.log(f"created {len(all_cell) * args.batch_size} samples")

    arr = np.concatenate(all_cell, axis=0)
    save_data(arr, traj, args.sample_dir+str(cell_type[0]))

    dist.barrier()
    logger.log("sampling complete")


def create_argparser(celltype=[0], weight=[10,10]):
    defaults = dict(
        clip_denoised=True,
        num_samples=9000,
        batch_size=3000,
        use_ddim=False,
        class_cond=False, 

        model_path="output/diffusion_checkpoint/muris_diffusion/model000000.pt", 

        # ***if commen conditional generation & gradiante interpolation, use this path***
        classifier_path="output/classifier_checkpoint/classifier_muris/model000100.pt",
        # ***if multi-conditional, use this path. replace this to your own classifiers***
        classifier_path1="output/classifier_checkpoint/classifier_muris_ood_type/model200000.pt",
        classifier_path2="output/classifier_checkpoint/classifier_muris_ood_organ/model200000.pt",
        num_class1 = 2,  # set this to the number of classes in your own dataset. this is the first condition (for example cell organ).
        num_class2 = 2,  # this is the second condition (for example cell type).

        # ***if commen conditional generation, use this scale***
        classifier_scale=2,
        # ***in multi-conditional, use this scale. scale1 and scale2 are the weights of two classifiers***
        # ***in Gradient Interpolation, use this scale, too. scale1 and scale2 are the weights of two gradients***
        classifier_scale1=weight[0]*2/10,
        classifier_scale2=weight[1]*2/10,

        # ***if gradient interpolation, replace these base on your own situation***
        ae_dir='output/Autoencoder_checkpoint/WOT/model_seed=0_step=150000.pt', 
        num_gene=19423,
        init_time = 600,    # initial noised state if interpolation
        init_cell_path = 'data/WOT/filted_data.h5ad',   #input initial noised cell state

        sample_dir=f"output/simulated_samples/muris",
        start_guide_steps = 500,     # the time to use classifier guidance
        filter = False,   # filter the simulated cells that are classified into other condition, might take long time

    )
    defaults.update(model_and_diffusion_defaults())
    defaults.update(classifier_and_diffusion_defaults())
    defaults['num_class']=12
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    # for conditional generation
    # main(cell_type=[2])
    for type in range(12):
        main(cell_type=[type])

    # ***for multi-condition, run***
    # muris ood
    # for i in [0,1]:
    #     for j in [0,1]:
    #         main(cell_type=[i,j],multi=True)

    # ***for Gradient Interpolation, run***
    # for i in range(0,11):
    #     main(cell_type=[6,7], inter=True, weight=[10-i,i])
    # for i in range(18):
    #     main(cell_type=[i,i+1], inter=True, weight=[5,5])