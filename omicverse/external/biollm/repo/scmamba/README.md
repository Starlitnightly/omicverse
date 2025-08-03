# scMamba

## Introduction
This is the official codebase for **scMamba**.

## [ckpt] Newly released
**Note:** For running BiMamba (bidirectional mamba), you need to additionally set 'bimamba_type=v1'.
- [2024/03/13] BiMamba: /home/share/huadjyin/home/s_jiangwenjian/proj/scLLM/scGPT/saves/Pretraining/cellxgene/BiMamba/gst_emb_Biv1_attnMVC_mr3
- [2024/03/13] Mamba: /home/share/huadjyin/home/s_jiangwenjian/proj/scLLM/scGPT/saves/Pretraining/cellxgene/Mamba/gst_emb_attnMVC_mr3
- ~~[2024/03/08] BiMamba(Deprecated): /home/share/huadjyin/home/s_jiangwenjian/proj/scLLM/scGPT/saves/Pretraining/cellxgene/BiMamba/gst_Biv2_emb_attnMVC_mr3~~
- ~~[2024/03/08] BiMamba(Deprecated): /home/share/huadjyin/home/s_jiangwenjian/proj/scLLM/scGPT/saves/Pretraining/cellxgene/BiMamba/gst_Biv2_emb_revision~~
- ~~[2024/03/07] BiMamba(Deprecated): /home/share/huadjyin/home/s_jiangwenjian/proj/scLLM/scGPT/saves/Pretraining/cellxgene/BiMamba/gst_ori_initemb~~
- [2024/03/07] Mamba: /home/share/huadjyin/home/s_jiangwenjian/proj/scLLM/scGPT/saves/Pretraining/cellxgene/mamba/gst_ori_initemb_mvc_cls
- [2024/03/03] Mamba(Recommended): /home/share/huadjyin/home/s_jiangwenjian/proj/scLLM/scGPT/saves/Pretraining/cellxgene/mamba/gst_ori_initemb
- ~~[2024/02/27] Mamba(Deprecated): /home/share/huadjyin/home/s_jiangwenjian/proj/scLLM/scGPT/saves/Pretraining/cellxgene/mamba/gst_ori_rdm_HEG~~

## [Log] Updates
- [2024/03/07] Attention-based aggregation for cell embeddings; revision for Bidirectional Mamba Pretraining script.
- [2024/03/06] Bidirectional Mamba for MLM pretrainig.
- [2024/03/01] Integration module has completed. Note that distributed pipeline (DDP) for integration is unavailable, left for future update.

## [Optional] Usage of wandb
We recommend using [wandb](https://wandb.ai/) for logging and visualization.

```bash
$ pip install wandb
```
In the case of first run in your device, you need to login with your API key of [W&B](https://wandb.ai/home).

```bash
wandb login
```

## [Optional] Flash-attention

**Note**: The `flash-attn` dependency usually requires specific GPU and CUDA version. If you encounter any issues, please refer to the [flash-attn](https://github.com/HazyResearch/flash-attention/tree/main) repository for installation instructions. For now, May 2023, we recommend using CUDA 11.7 and flash-attn<1.0.5 due to various issues reported about installing new versions of flash-attn.

You can turn off flash-attn by setting *fast_transformer=False* in your command.
```bash
python3 srcipt.py --fast_transformer False
```

## Running Finetune for downstream task **[For User]**

|           Task           |                               Code                                |   Status   |         Available Dataset         |
|:------------------------:|:-----------------------------------------------------------------:|:----------:|:---------------------------------:|
|     Cell Annotation      |          [scmamba/mamba_CA.py](scmamba/mamba_CA.py)               |  Released  | Zheng68k / hPancreas / M.S. / Mye |
|       Integration        | [scmamba/mamba_intergration.py](scmamba/mamba_intergration.py.py) |  Be Ready  |                                   |
|      GRN inference       |                            Unavailable                            | Developing |                                   |
|       Perturbation       |                            Unavailable                            | Developing |                                   |
| Drug response prediction |                            Unavailable                            | Developing |                                   |

For running cell annotation for example:
```bash
python3 scmamba/mamba_CA.py --model_name mamba --epochs 15 \
--cell_emb_style avg-pool --batch_size 64 --data_name your_data \
--load_model your_model_checkpoint \ 
# train from scratch if ckpt is not provided

--save_dir your_save_dir
--run_name your_run_name
# the result will be save in: your_save_dir/your_model_name/your_data_name/your_run_name

--single_gpu True
--distributed True
# Note: single_gpu and distribute shouldn't be set to True at the same time, you have to choose one from them.
# If both of them are set to False, this script will jump into debug mode.
```
## To-do-list

- [x] scMamba Pretraining with Masked Value Prediction/ Regulatory Role Prediction

- [x] scGPT Pretraining with Masked Value Prediction/ Regulatory Role Prediction

- [x] Cell Annotation Pipeline

- [ ] Integration Pipeline (comming soon)

- [ ] GRN inference Pipeline (Under development)

- [ ] Perturbation (Under development)

- [ ] Drug response prediction (Under development)

- [ ] Publish to huggingface model hub

## Fine-tune scMamba for Cell Annotation **[For Developer]**

Please see example pipeline for Cell Annotation in [scmamba/mamba_CA.py](scmamba/mamba_CA.py).

Set *single_gpu=False, distributed=False* for jumping into the debug mode.

For developing new downstream task, you also need to implement corresponding module in the [dataset](scLLM_utils/dataset.py) and [dataloader](scLLM_utils/dataloader.py).

