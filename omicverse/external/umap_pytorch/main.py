import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.nn.functional import mse_loss
import torch.nn.functional as F
from tqdm import tqdm

from .data import UMAPDataset, MatchDataset
from .modules import get_umap_graph, umap_loss
from .model import default_encoder, default_decoder

from umap.umap_ import find_ab_params
import dill
from umap import UMAP

# Import Colors for colored output
try:
    from ..._settings import Colors
except:
    # Fallback if Colors not available
    class Colors:
        HEADER = '\033[95m'
        OKBLUE = '\033[94m'
        OKCYAN = '\033[96m'
        OKGREEN = '\033[92m'
        WARNING = '\033[93m'
        FAIL = '\033[91m'
        ENDC = '\033[0m'
        BOLD = '\033[1m'
        UNDERLINE = '\033[4m'
        CYAN = '\033[96m'
        GREEN = '\033[92m'

""" Model """


class Model(nn.Module):
    def __init__(
        self,
        lr: float,
        encoder: nn.Module,
        decoder=None,
        beta = 1.0,
        min_dist=0.1,
        reconstruction_loss=F.binary_cross_entropy_with_logits,
        match_nonparametric_umap=False,
    ):
        super().__init__()
        self.lr = lr
        self.encoder = encoder
        self.decoder = decoder
        self.beta = beta # weight for reconstruction loss
        self.match_nonparametric_umap = match_nonparametric_umap
        self.reconstruction_loss = reconstruction_loss
        self._a, self._b = find_ab_params(1.0, min_dist)

    def forward(self, batch):
        """
        Forward pass and loss computation.
        Returns: (total_loss, loss_dict) where loss_dict contains individual losses for logging.
        """
        if not self.match_nonparametric_umap:
            (edges_to_exp, edges_from_exp) = batch
            embedding_to, embedding_from = self.encoder(edges_to_exp), self.encoder(edges_from_exp)
            encoder_loss = umap_loss(embedding_to, embedding_from, self._a, self._b, edges_to_exp.shape[0], negative_sample_rate=5)

            loss_dict = {"umap_loss": encoder_loss.item()}

            if self.decoder:
                recon = self.decoder(embedding_to)
                recon_loss = self.reconstruction_loss(recon, edges_to_exp)
                loss_dict["recon_loss"] = recon_loss.item()
                total_loss = encoder_loss + self.beta * recon_loss
            else:
                total_loss = encoder_loss

            return total_loss, loss_dict

        else:
            data, embedding = batch
            embedding_parametric = self.encoder(data)
            encoder_loss = mse_loss(embedding_parametric, embedding)

            loss_dict = {"encoder_loss": encoder_loss.item()}

            if self.decoder:
                recon = self.decoder(embedding_parametric)
                recon_loss = self.reconstruction_loss(recon, data)
                loss_dict["recon_loss"] = recon_loss.item()
                total_loss = encoder_loss + self.beta * recon_loss
            else:
                total_loss = encoder_loss

            return total_loss, loss_dict
            

""" DataLoader creation """


def create_dataloader(dataset, batch_size, num_workers):
    """Create a DataLoader for training."""
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
    )

class PUMAP():
    def __init__(
        self,
        encoder=None,
        decoder=None,
        n_neighbors=10,
        min_dist=0.1,
        metric="euclidean",
        n_components=2,
        beta=1.0,
        reconstruction_loss=F.binary_cross_entropy_with_logits,
        random_state=None,
        lr=1e-3,
        epochs=10,
        batch_size=64,
        num_workers=1,
        num_gpus=1,
        match_nonparametric_umap=False,
        early_stopping=True,
        patience=10,
        min_delta=1e-4,
        use_pyg='auto',
    ):
        self.encoder = encoder
        self.decoder = decoder
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.metric = metric
        self.n_components = n_components
        self.beta = beta
        self.reconstruction_loss = reconstruction_loss
        self.random_state = random_state
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_gpus = num_gpus
        self.match_nonparametric_umap = match_nonparametric_umap
        self.early_stopping = early_stopping
        self.patience = patience
        self.min_delta = min_delta
        self.use_pyg = use_pyg
        
    def fit(self, X):
        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() and self.num_gpus > 0 else 'cpu')

        print(f"{Colors.CYAN}{'='*60}{Colors.ENDC}")
        print(f"{Colors.BOLD}{Colors.HEADER}🚀 Parametric UMAP Training{Colors.ENDC}")
        print(f"{Colors.CYAN}{'='*60}{Colors.ENDC}")
        print(f"{Colors.GREEN}📊 Device: {Colors.BOLD}{device}{Colors.ENDC}")
        print(f"{Colors.GREEN}📈 Data shape: {Colors.BOLD}{X.shape}{Colors.ENDC}")

        # Initialize encoder and decoder
        encoder = default_encoder(X.shape[1:], self.n_components) if self.encoder is None else self.encoder

        if self.decoder is None or isinstance(self.decoder, nn.Module):
            decoder = self.decoder
        elif self.decoder == True:
            decoder = default_decoder(X.shape[1:], self.n_components)

        # Create model and move to device
        if not self.match_nonparametric_umap:
            print(f"{Colors.GREEN}🔗 Building UMAP graph...{Colors.ENDC}")
            self.model = Model(self.lr, encoder, decoder, beta=self.beta, min_dist=self.min_dist, reconstruction_loss=self.reconstruction_loss)
            graph = get_umap_graph(X, n_neighbors=self.n_neighbors, metric=self.metric, random_state=self.random_state, use_pyg=self.use_pyg)
            dataset = UMAPDataset(X, graph)
        else:
            print(f"{Colors.CYAN}🔍 Fitting Non-parametric UMAP...{Colors.ENDC}")
            non_parametric_umap = UMAP(n_neighbors=self.n_neighbors, min_dist=self.min_dist, metric=self.metric, n_components=self.n_components, random_state=self.random_state, verbose=True)
            non_parametric_embeddings = non_parametric_umap.fit_transform(torch.flatten(X, 1, -1).numpy())
            self.model = Model(self.lr, encoder, decoder, beta=self.beta, reconstruction_loss=self.reconstruction_loss, match_nonparametric_umap=self.match_nonparametric_umap)
            print(f"{Colors.GREEN}🎯 Training NN to match embeddings...{Colors.ENDC}")
            dataset = MatchDataset(X, non_parametric_embeddings)

        self.model = self.model.to(device)

        # Create dataloader
        dataloader = create_dataloader(dataset, self.batch_size, self.num_workers)

        # Initialize optimizer
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)

        # Early stopping variables
        best_loss = float('inf')
        patience_counter = 0
        best_model_state = None

        # Training loop with progress bar
        print(f"\n{Colors.BOLD}{Colors.GREEN}🏋️  Starting Training...{Colors.ENDC}")
        print(f"{Colors.CYAN}{'─'*60}{Colors.ENDC}")

        pbar = tqdm(range(self.epochs), desc=f"{Colors.BOLD}Training{Colors.ENDC}",
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')

        for epoch in pbar:
            self.model.train()
            epoch_losses = {}
            num_batches = 0

            for batch_idx, batch in enumerate(dataloader):
                # Move batch to device
                if isinstance(batch, (list, tuple)):
                    batch = tuple(b.to(device) if isinstance(b, torch.Tensor) else b for b in batch)
                else:
                    batch = batch.to(device)

                # Forward pass
                loss, loss_dict = self.model(batch)

                # Accumulate losses for logging
                for key, value in loss_dict.items():
                    epoch_losses[key] = epoch_losses.get(key, 0) + value
                num_batches += 1

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Calculate average losses
            avg_losses = {key: value / num_batches for key, value in epoch_losses.items()}

            # Get the main loss for early stopping
            main_loss = list(avg_losses.values())[0]

            # Update progress bar with colored loss information
            loss_str = " | ".join([f"{key}: {value:.4f}" for key, value in avg_losses.items()])
            pbar.set_postfix_str(loss_str)

            # Early stopping logic
            if self.early_stopping:
                if main_loss < best_loss - self.min_delta:
                    best_loss = main_loss
                    patience_counter = 0
                    best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                    pbar.write(f"{Colors.GREEN}✓ New best loss: {main_loss:.4f}{Colors.ENDC}")
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        pbar.write(f"\n{Colors.WARNING}⚠️  Early stopping triggered at epoch {epoch + 1}/{self.epochs}{Colors.ENDC}")
                        pbar.write(f"{Colors.WARNING}   No improvement for {self.patience} epochs{Colors.ENDC}")
                        # Restore best model
                        if best_model_state is not None:
                            self.model.load_state_dict(best_model_state)
                            pbar.write(f"{Colors.GREEN}✓ Restored best model (loss: {best_loss:.4f}){Colors.ENDC}")
                        break

        pbar.close()
        print(f"\n{Colors.CYAN}{'─'*60}{Colors.ENDC}")
        print(f"{Colors.BOLD}{Colors.GREEN}✅ Training Completed!{Colors.ENDC}")
        print(f"{Colors.GREEN}📉 Final best loss: {Colors.BOLD}{best_loss:.4f}{Colors.ENDC}")
        print(f"{Colors.CYAN}{'='*60}{Colors.ENDC}\n")
        
    @torch.no_grad()
    def transform(self, X):
        #print(f"Reducing array of shape {X.shape} to ({X.shape[0]}, {self.n_components})")
        # Set model to evaluation mode
        self.model.eval()

        # Ensure model is on the same device as input
        device = X.device if isinstance(X, torch.Tensor) else 'cpu'
        self.model = self.model.to(device)

        # Move input to same device if not already
        if isinstance(X, torch.Tensor):
            X = X.to(device)
        else:
            X = torch.from_numpy(X).to(device)

        return self.model.encoder(X).detach().cpu().numpy()
    
    @torch.no_grad()
    def inverse_transform(self, Z):
        # Set model to evaluation mode
        self.model.eval()

        # Ensure model is on the same device as input
        device = Z.device if isinstance(Z, torch.Tensor) else 'cpu'
        self.model = self.model.to(device)

        # Move input to same device if not already
        if isinstance(Z, torch.Tensor):
            Z = Z.to(device)
        else:
            Z = torch.from_numpy(Z).to(device)

        return self.model.decoder(Z).detach().cpu().numpy()
    
    def save(self, path):
        with open(path, 'wb') as oup:
            dill.dump(self, oup)
        print(f"Pickled PUMAP object at {path}")
        
def load_pumap(path): 
    print("Loading PUMAP object from pickled file.")
    with open(path, 'rb') as inp: return dill.load(inp)

if __name__== "__main__":
    pass