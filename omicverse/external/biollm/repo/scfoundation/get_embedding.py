# Copyright 2023 BioMap (Beijing) Intelligence Technology Limited


from tqdm import tqdm
from .load import *

####################################Settings#################################

def main_gene_selection(X_df, gene_list):
    """
    Describe:
        rebuild the input adata to select target genes encode protein 
    Parameters:
        adata->`~anndata.AnnData` object: adata with var index_name by gene symbol
    Returns:
        adata_new->`~anndata.AnnData` object
        to_fill_columns->list: zero padding gene
    """

    to_fill_columns = list(set(gene_list) - set(X_df.columns))
    padding_df = pd.DataFrame(np.zeros((X_df.shape[0], len(to_fill_columns))), 
                              columns=to_fill_columns, 
                              index=X_df.index)
    X_df = pd.DataFrame(np.concatenate([df.values for df in [X_df, padding_df]], axis=1), 
                        index=X_df.index, 
                        columns=list(X_df.columns) + list(padding_df.columns))
    X_df = X_df[gene_list]
    
    var = pd.DataFrame(index=X_df.columns)
    var['mask'] = [1 if i in to_fill_columns else 0 for i in list(var.index)]
    return X_df, to_fill_columns, var


def embedding(gexpr_feature, model, config, device, emb_mod, batch_size, tgthighres, input_type, pool_type):

    #Set random seed
    random.seed(0)
    np.random.seed(0)  # numpy random generator

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    model.eval()


    geneexpemb = []
    batchcontainer = []

    # Inference
    for i in tqdm(range(gexpr_feature.shape[0])):
        with torch.no_grad():
            # Bulk
            if input_type == 'bulk':
                tmpdata = (gexpr_feature.iloc[i, :]).tolist()
                totalcount = gexpr_feature.iloc[i, :].sum()
                pretrain_gene_x = torch.tensor(tmpdata + [totalcount, totalcount], device=device).unsqueeze(0)
                data_gene_ids = torch.arange(19266, device=pretrain_gene_x.device).repeat(pretrain_gene_x.shape[0], 1)

            # Single cell
            elif input_type == 'singlecell':

                tmpdata = (gexpr_feature.iloc[i, :]).tolist()
                totalcount = gexpr_feature.iloc[i, :].sum()

                # select resolution
                if tgthighres[0] == 'f':
                    pretrain_gene_x = torch.tensor(
                        tmpdata + [np.log10(totalcount * float(tgthighres[1:])), np.log10(totalcount)], device=device).unsqueeze(
                        0)
                elif tgthighres[0] == 'a':
                    pretrain_gene_x = torch.tensor(
                        tmpdata + [np.log10(totalcount) + float(tgthighres[1:]), np.log10(totalcount)], device=device).unsqueeze(
                        0)
                elif tgthighres[0] == 't':
                    pretrain_gene_x = torch.tensor(
                        tmpdata + [float(tgthighres[1:]), np.log10(totalcount)], device=device).unsqueeze(0)
                else:
                    raise ValueError('tgthighres must be start with f, a or t')
                data_gene_ids = torch.arange(19266, device=pretrain_gene_x.device).repeat(pretrain_gene_x.shape[0], 1)

            value_labels = pretrain_gene_x > 0
            x, x_padding = gatherData(pretrain_gene_x, value_labels, config['pad_token_id'])

            #Cell embedding
            if emb_mod=='cell':
                position_gene_ids, _ = gatherData(data_gene_ids, value_labels, config['pad_token_id'])
                x = model.token_emb(torch.unsqueeze(x, 2).float(), output_weight = 0)
                position_emb = model.pos_emb(position_gene_ids)
                x += position_emb
                geneemb = model.encoder(x,x_padding)

                geneemb1 = geneemb[:,-1,:]
                geneemb2 = geneemb[:,-2,:]
                geneemb3, _ = torch.max(geneemb[:,:-2,:], dim=1)
                geneemb4 = torch.mean(geneemb[:,:-2,:], dim=1)
                if pool_type=='all':
                    geneembmerge = torch.concat([geneemb1,geneemb2,geneemb3,geneemb4],axis=1)
                elif pool_type=='max':
                    geneembmerge, _ = torch.max(geneemb, dim=1)
                else:
                    raise ValueError('pool_type must be all or max')
                geneexpemb.append(geneembmerge.detach().cpu().numpy())

            #Gene + expression embedding
            elif emb_mod == 'gene-expression':
                model.to_final = None
                encoder_data, encoder_position_gene_ids, encoder_data_padding, encoder_labels, decoder_data, decoder_data_padding, new_data_raw, data_mask_labels, decoder_position_gene_ids = getEncoerDecoderData(pretrain_gene_x.float(),pretrain_gene_x.float(), config)
                out = model.forward(x=encoder_data, padding_label=encoder_data_padding,
                            encoder_position_gene_ids=encoder_position_gene_ids,
                            encoder_labels=encoder_labels,
                            decoder_data=decoder_data,
                            mask_gene_name=False,
                            mask_labels=None,
                            decoder_position_gene_ids=decoder_position_gene_ids,
                            decoder_data_padding_labels=decoder_data_padding,
                            )
                out = out[:,:19264,:].contiguous()
                geneexpemb.append(out.detach().cpu().numpy())

            #Gene batch embedding
            elif emb_mod == 'gene_batch':
                batchcontainer.append(pretrain_gene_x.float())
                if len(batchcontainer)==gexpr_feature.shape[0]:
                    batchcontainer = torch.concat(batchcontainer,axis=0)
                else:
                    continue
                model.to_final = None
                encoder_data, encoder_position_gene_ids, encoder_data_padding, encoder_labels, decoder_data, decoder_data_padding, new_data_raw, data_mask_labels, decoder_position_gene_ids = getEncoerDecoderData(batchcontainer,batchcontainer,config)
                out = model.forward(x=encoder_data, padding_label=encoder_data_padding,
                            encoder_position_gene_ids=encoder_position_gene_ids,
                            encoder_labels=encoder_labels,
                            decoder_data=decoder_data,
                            mask_gene_name=False,
                            mask_labels=None,
                            decoder_position_gene_ids=decoder_position_gene_ids,
                            decoder_data_padding_labels=decoder_data_padding,
                            )
                geneexpemb = out[:,:19264,:].contiguous().detach().cpu().numpy()

            #Gene_expression
            elif emb_mod == 'expression':
                encoder_data, encoder_position_gene_ids, encoder_data_padding, encoder_labels, decoder_data, decoder_data_padding, new_data_raw, data_mask_labels, decoder_position_gene_ids = getEncoerDecoderData(pretrain_gene_x.float(),pretrain_gene_x.float(),config)
                out = model.forward(x=encoder_data, padding_label=encoder_data_padding,
                            encoder_position_gene_ids=encoder_position_gene_ids,
                            encoder_labels=encoder_labels,
                            decoder_data=decoder_data,
                            mask_gene_name=False,
                            mask_labels=None,
                            decoder_position_gene_ids=decoder_position_gene_ids,
                            decoder_data_padding_labels=decoder_data_padding,
                            )
                out = out[:,:19264].contiguous()
                geneexpemb.append(out.detach().cpu().numpy())
            else:
                raise ValueError('output_type must be cell or gene-expression or gene_batch or expression')

    geneexpemb = np.squeeze(np.array(geneexpemb))
    return geneexpemb
    
