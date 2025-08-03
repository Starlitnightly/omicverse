from .downstream import AnnotationHead, DenoisingHead, PerturbationPredictionHead, PatientClassificationHead, EmbedderHead, ImputationHead
# from .spatial import
from torch import nn

def setup_head(head_type, in_dim, hidden_dim, out_dim, num_layers, dropout, norm, batch_num) -> nn.Module:
    if head_type == 'annotation':
        mod = AnnotationHead(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            num_classes=out_dim,
            num_layers=num_layers,
            dropout=dropout,
            norm=norm,
            batch_num=batch_num,
        )
    elif head_type == 'denoising':
        mod = DenoisingHead(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            num_layers=num_layers,
            dropout=dropout,
            norm=norm,
            batch_num=batch_num,
        )
    elif head_type == 'perturbation_prediction':
        mod = PerturbationPredictionHead(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            num_layers=num_layers,
            dropout=dropout,
            norm=norm,
            batch_num=batch_num,
        )
    elif head_type == 'patient_classification':
        mod = PatientClassificationHead(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            num_classes=out_dim,
            num_layers=num_layers,
            dropout=dropout,
            norm=norm,
            batch_num=batch_num,
        )
    elif head_type == 'imputation':
        mod = ImputationHead(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            num_layers=num_layers,
            dropout=dropout,
            norm=norm,
            batch_num=batch_num,
        )
    elif head_type == 'embedder':
        mod = EmbedderHead(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            num_layers=num_layers,
            dropout=dropout,
            norm=norm,
            batch_num=batch_num,
        )
    else:
        raise NotImplementedError(f'Unsupported model type: {head_type}')
    return mod