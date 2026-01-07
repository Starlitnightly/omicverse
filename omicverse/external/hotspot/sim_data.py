import numpy as np
from numba import jit
from tqdm import tqdm


def sim_latent(N_CELLS, N_DIM):
    """
    Draws a latent space coordinate for each cell
    randomly from a unit gaussian
    """

    return np.random.normal(size=(N_CELLS, N_DIM))


def sim_umi_counts(N_CELLS, mid, scale):
    """
    Simulates counts from a log-normal distribution
    """

    mu = np.log(mid)
    sd = np.log(mid+scale)-mu

    vals = np.random.normal(loc=mu, scale=sd, size=N_CELLS)
    vals = np.floor(np.exp(vals))

    return vals


def sim_counts_bernoulli(N_CELLS, umi_counts, gene_p):
    """
    For a given transcript probability, simulates detection
    in N_CELLS with `umi_counts` umis per cell

    Note: gene_p is in transcripts / 10,000
    """
    gene_p = gene_p / 10000

    detect_p = 1-(1-gene_p)**umi_counts

    vals = (np.random.rand(N_CELLS) < detect_p).astype('double')

    return vals


def sim_counts_danb(N_CELLS, N_GENES, umi_counts, mean, size):
    """
    Simulates transcript counts under the DANB model
    Uses umi_counts to adjust the mean

    Returns N_GENES replicates -> result is N_GENES x N_CELLS
    """
    umi_scale = umi_counts / umi_counts.mean()

    vals = np.zeros((N_GENES, N_CELLS))

    for i in tqdm(range(N_CELLS)):
        mean_i = mean*umi_scale[i]
        var = mean_i*(1+mean_i/size)
        p = 1 - mean_i/var
        n = mean_i * (1-p) / p
        p = 1-p

        vals[:, i] = np.random.negative_binomial(n=n, p=p, size=N_GENES)

    return vals


def generate_permutation_null(observed_counts, N_REPS):

    out = []
    for i in range(N_REPS):
        out.append(np.random.permutation(observed_counts))

    return np.vstack(out)


def generate_resampled_null(observed_counts, N_REPS):

    out = []
    for i in range(N_REPS):
        out.append(
            np.random.choice(
                observed_counts, size=len(observed_counts), replace=True
            )
        )

    return np.vstack(out)
