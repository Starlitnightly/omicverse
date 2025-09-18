from .trainer import train_vae
from .trainer_nogcn import train_vae_nogcn
from .trainer_anvi import train_anvi
from .trainer_anvi_nogcn import train_anvi_nogcn
#from .trainer_atac import train_atac

def train(model, adata, epochs = 50, learning_rate = 1e-2, batch_size = 200, grad_clip = 1000, shuffle=True, test=0.1, name = '', optimizer='adam', random_seed=42):

    gcn = model.gcn
    annot = model.annot

    if gcn and not annot:
        epochs, val_ae, val_traj = train_vae(model, adata, epochs=epochs,
                  learning_rate=learning_rate, batch_size=batch_size,
                  grad_clip=grad_clip, shuffle=shuffle, test=test, name=name,
                  optimizer=optimizer, random_seed=random_seed)
    
    elif gcn and annot:
        epochs, val_ae, val_traj = train_anvi(model, adata, epochs=epochs,
                  learning_rate=learning_rate, batch_size=batch_size,
                  grad_clip=grad_clip, shuffle=shuffle, test=test, name=name,
                  optimizer=optimizer, random_seed=random_seed)

    elif not gcn and not annot:
        epochs, val_ae, val_traj = train_vae_gcn(model, adata, epochs=epochs,
                  learning_rate=learning_rate, batch_size=batch_size,
                  grad_clip=grad_clip, shuffle=shuffle, test=test, name=name,
                  optimizer=optimizer, random_seed=random_seed)

    elif not gcn and annot:
        epochs, val_ae, val_traj = train_anvi_gcn(model, adata, epochs=epochs,
                  learning_rate=learning_rate, batch_size=batch_size,
                  grad_clip=grad_clip, shuffle=shuffle, test=test, name=name,
                  optimizer=optimizer, random_seed=random_seed)
    
    return epochs, val_ae, val_traj
