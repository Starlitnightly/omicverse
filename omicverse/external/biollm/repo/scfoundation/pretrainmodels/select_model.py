# Copyright 2023 BioMap (Beijing) Intelligence Technology Limited


from .transformer import pytorchTransformerModule
from .performer import PerformerModule
from .mae_autobin import MaeAutobin

def select_module(config, sub_config, module_name):
    if module_name == 'performer':
        return PerformerModule(
            max_seq_len=config['seq_len'],
            dim=sub_config['hidden_dim'],
            depth=sub_config['depth'],
            heads=sub_config['heads'],
            dim_head=sub_config['dim_head'],
            ff_dropout=sub_config.get('ff_dropout',0.0),
            attn_dropout=sub_config.get('attn_dropout',0.0)
        )
    elif module_name == 'transformer':
        return pytorchTransformerModule(
            max_seq_len=config['seq_len'],
            dim=sub_config['hidden_dim'],
            depth=sub_config['depth'],
            heads=sub_config['heads']
        )
    else:
        print('module type error')
        exit(0)

def select_model(config):
    if config["model"] == "mae_autobin":
        encoder_config =config['encoder']
        decoder_config = config['decoder']
        encoder = select_module(config, encoder_config, config['encoder']['module_type'])
        decoder = select_module(config, decoder_config, config['decoder']['module_type'])
        model = MaeAutobin(
            num_tokens=config['n_class'],
            max_seq_len=config['seq_len'],
            embed_dim=config['encoder']['hidden_dim'],
            decoder_embed_dim=config['decoder']['hidden_dim'],
            bin_alpha = config['bin_alpha'],
            bin_num = config['bin_num'],
            pad_token_id = config['pad_token_id'],
            mask_token_id = config['mask_token_id'],
        )
        model.encoder = encoder
        model.decoder = decoder     
    else:
        raise NotImplementedError("Unknown model type!")
    return model

def get_sub_config(config, target):
    """
    获取 包含 target 的 configs
    """
    sub_config = {}
    for k in config.keys():
        if target in k:
            tmp_name = k.replace(target + '_', '')
            sub_config[tmp_name] = config[k]
    return sub_config
