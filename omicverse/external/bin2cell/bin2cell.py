from __future__ import annotations

from scanpy import read_10x_h5
from scanpy import logging as logg

import json
from pathlib import Path, PurePath
from typing import BinaryIO, Literal
import pandas as pd
from matplotlib.image import imread
from typing import Tuple, Optional, Literal, Dict
import numpy as np
from scipy import sparse



import math

#actual bin2cell dependencies start here
#the ones above are for read_visium()
#from stardist.plot import render_label
from copy import deepcopy


import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import scipy.spatial
import scipy.sparse
import scipy.stats
import anndata as ad
import scanpy as sc
import numpy as np
import os

from PIL import Image
#setting needed so PIL can load the large TIFFs
#Image.MAX_IMAGE_PIXELS = None

#setting needed so cv2 can load the large TIFFs
#os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(2**40)


#NOTE ON DIMENSIONS WITHIN ANNDATA AND BEYOND
#.obs["array_row"] matches .obsm["spatial"][:,1] matches np.array image [:,0]
#.obs["array_col"] matches .obsm["spatial"][:,0] matches np.array image [:,1]
#array coords start relative to some magical point on the grid (seen bottom left/top right)
#and can require flipping the array row or column to match the image orientation
#row/col do seem to consistenly refer to the stated np.array image dimensions though
#spatial starts in top left corner of image, matching what np.array image is doing
#of note, cv2 treats [:,1] as dim 0 and [:,0] as dim 1, despite working on np.arrays
#also cv2 works with channels in a BGR order, while everything else is RGB



def render_label(labels, img=None):
    """替代 stardist.plot.render_label"""
    from skimage.color import label2rgb
    return label2rgb(labels, image=img, bg_label=0, alpha=0.3)


import numpy as np

def load_image(image_path, gray=False, dtype=np.uint8, backend="opencv"):
    """
    Efficiently load an external image and return it as an RGB numpy array.
    
    Parameters
    ----------
    image_path : str
        Path to image to be loaded.
    gray : bool, optional (default: False)
        Whether to return grayscale image.
    dtype : numpy.dtype, optional (default: np.uint8)
        Data type of the numpy array output.
    backend : {"opencv", "pil", "tifffile"}, optional (default: "opencv")
        Which library to use for reading. "opencv" is fastest, but may fail on huge images.
    
    Returns
    -------
    np.ndarray
        The loaded image.
    """
    img = None

    if backend == "opencv":
        import cv2
        #解除 OpenCV 的最大像素限制
        cv2.setConfigOption("OPENCV_IO_MAX_IMAGE_PIXELS", str(2**63 - 1))
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise IOError(f"OpenCV failed to load image: {image_path}")
        # 默认 BGR → RGB
        if len(img.shape) == 3 and img.shape[2] >= 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if gray:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    elif backend == "pil":
        from PIL import Image
        img = Image.open(image_path)
        img = img.convert("L" if gray else "RGB")
        img = np.array(img, dtype=dtype)

    elif backend == "tifffile":
        import tifffile
        img = tifffile.imread(image_path)
        if gray and img.ndim == 3:
            img = img.mean(axis=-1).astype(dtype)

    else:
        raise ValueError(f"Unsupported backend: {backend}")

    return img.astype(dtype, copy=False)


def normalize(img):
    '''
    Extremely naive reimplementation of default ``cbsdeep.utils.normalize()`` 
    percentile normalisation, with a lower RAM footprint than the original.
    
    Input
    -----
    img : ``numpy.array``
        Numpy array image to normalise
    '''
    eps = 1e-20
    mi = np.percentile(img,3)
    ma = np.percentile(img,99.8)
    return ((img - mi) / ( ma - mi + eps ))

# -*- coding: utf-8 -*-
# Drop-in replacement for bin2cell.stardist using Cellpose with tiled inference.





def _ensure_gray(img: np.ndarray) -> np.ndarray:
    from skimage.color import rgb2gray
    if img.ndim == 3 and img.shape[-1] in (3, 4):
        # skimage.rgb2gray handles RGB/RGBA -> grayscale [0..1]

        g = rgb2gray(img)
        # keep dtype close to original
        if img.dtype.kind in ("u", "i"):
            g = (g * np.iinfo(img.dtype).max).astype(img.dtype)
        return g
    return img

def _tiles(H: int, W: int, block: int, overlap: int, context: int):
    """
    Yield tile windows with overlap+context.
    Returns tuples: (y0,y1,x0,x1) = effective crop (with context),
                    (yy0,yy1,xx0,xx1) = paste window (without context, i.e., core).
    """
    stride = block - overlap
    y_starts = list(range(0, max(H - block, 0) + 1, stride)) or [0]
    x_starts = list(range(0, max(W - block, 0) + 1, stride)) or [0]
    # Ensure last tile reaches image border
    if y_starts[-1] + block < H:
        y_starts.append(H - block)
    if x_starts[-1] + block < W:
        x_starts.append(W - block)

    for ys in y_starts:
        for xs in x_starts:
            core_y0, core_y1 = ys, min(ys + block, H)
            core_x0, core_x1 = xs, min(xs + block, W)

            y0 = max(0, core_y0 - context)
            x0 = max(0, core_x0 - context)
            y1 = min(H, core_y1 + context)
            x1 = min(W, core_x1 + context)

            # core (without context) in global coords:
            yield (y0, y1, x0, x1), (core_y0, core_y1, core_x0, core_x1)

def _merge_labels_overlapping(
    base_core: np.ndarray,
    new_core: np.ndarray,
    iou_thresh: float = 0.5,
) -> Dict[int, int]:
    """
    In overlapping paste area between existing labels (base_core) and new labels (new_core),
    compute IoU and decide a mapping from new IDs -> existing IDs (to avoid duplicates).
    Returns a dict: {new_id: existing_id}
    """
    mapping = {}
    # quickly skip if nearly empty
    if base_core.max() == 0 or new_core.max() == 0:
        return mapping

    # Build contingency of pairs (base_id, new_id) on overlapping pixels where both > 0
    overlap_mask = (base_core > 0) & (new_core > 0)
    if not overlap_mask.any():
        return mapping

    base_ids = base_core[overlap_mask]
    new_ids = new_core[overlap_mask]
    pairs = np.stack([base_ids, new_ids], axis=1)

    # Count intersection per (base_id, new_id)
    # We compress pairs into unique rows and counts
    # Using np.unique with return_counts
    uniq_pairs, counts = np.unique(pairs, axis=0, return_counts=True)
    # Build areas
    base_area = np.bincount(base_core.ravel())
    new_area = np.bincount(new_core.ravel())

    for (b_id, n_id), inter in zip(uniq_pairs, counts):
        if b_id == 0 or n_id == 0:  # safety
            continue
        # IoU = inter / (area_b + area_n - inter)
        if b_id < len(base_area) and n_id < len(new_area):
            union = base_area[b_id] + new_area[n_id] - inter
            if union > 0 and (inter / union) >= iou_thresh:
                # prefer first strong match; if multiple, keep largest IoU by updating only if better
                # We don't precompute IoU ranking for brevity; this heuristic works well in practice.
                if n_id not in mapping:
                    mapping[n_id] = b_id
    return mapping

def _paste_with_relabel(
    global_rows, global_cols, global_vals,
    tile_mask: np.ndarray,
    paste_win: Tuple[int, int, int, int],
    existing_global: Optional[np.ndarray],
    next_label_start: int,
    iou_thresh: float = 0.5,
):
    """
    Accumulate sparse coordinates for tile_mask pasted into global coords with ID relabeling.
    - existing_global: optional small array view (current labels) of the same paste window;
                       if provided, used to compute IoU-based merge mapping.
    Returns new_next_label, and extends lists global_rows, global_cols, global_vals
    """
    y0, y1, x0, x1 = paste_win
    core_h, core_w = (y1 - y0), (x1 - x0)
    core = tile_mask[:core_h, :core_w]  # ensure exact core cut (tile_mask is already cropped to core)

    # Determine mapping using IoU in the paste area
    mapping = {}
    if existing_global is not None:
        mapping = _merge_labels_overlapping(existing_global, core, iou_thresh=iou_thresh)

    # Relabel: new IDs -> either mapped to existing, or shifted by next_label_start
    # Compute relabel array:
    max_id = core.max()
    if max_id == 0:
        return next_label_start

    relabel = np.zeros(max_id + 1, dtype=np.int32)
    for nid in range(1, max_id + 1):
        if nid in mapping:
            relabel[nid] = mapping[nid]
        else:
            relabel[nid] = next_label_start
            next_label_start += 1

    core_re = relabel[core]  # remapped core IDs

    # Append nonzero pixels into global sparse coordinates
    nz = core_re.nonzero()
    if len(nz[0]) == 0:
        return next_label_start
    gy = nz[0] + y0
    gx = nz[1] + x0
    global_rows.append(gy)
    global_cols.append(gx)
    global_vals.append(core_re[nz])

    return next_label_start

def stardist(
    image_path: str,
    labels_npz_path: str,
    stardist_model: str = "2D_versatile_he",
    block_size: int = 2048,
    min_overlap: int = 256,
    context: int = 64,
    # Cellpose-related knobs (can be passed via **kwargs in original calls)
    gpu: bool = False,
    diameter: Optional[float] = None,
    mask_threshold: Optional[float] = None,   # maps from prob_thresh if provided
    flow_threshold: Optional[float] = None,
    iou_merge_threshold: float = 0.5,
    build_sparse_directly: bool = True,
    show_progress: bool = True,
    progress_desc: Optional[str] = None,
    **kwargs,
):
    """
    Drop-in replacement for bin2cell.stardist() using Cellpose with tiled inference.
    Keeps I/O identical: writes a SciPy sparse CSR label matrix to labels_npz_path.

    Parameters kept for compatibility:
      - block_size, min_overlap, context: control tiling; mirrors original StarDist big-predict.

    Additional parameters:
      - show_progress: whether to display a progress bar over tiles.
      - progress_desc: optional description for the progress bar.
    """
    # map Stardist model to Cellpose model
    from cellpose import models
    from skimage.io import imread
    if stardist_model == "2D_versatile_he":
        model_type = "cyto"
    elif stardist_model == "2D_versatile_fluo":
        model_type = "nuclei"
    else:
        model_type = "cyto"

    # map StarDist prob_thresh -> Cellpose mask_threshold if user passed it in kwargs
    if "prob_thresh" in kwargs and mask_threshold is None:
        mask_threshold = kwargs.pop("prob_thresh")

    # load image
    img = imread(image_path)
    img = _ensure_gray(img)
    H, W = img.shape[:2]

    # init model
    cp = models.CellposeModel(gpu=gpu, model_type=model_type)
    # inspect eval signature once for compatibility handling
    import inspect
    eval_params = inspect.signature(cp.eval).parameters
    supports_channels = "channels" in eval_params
    supports_channel_axis = "channel_axis" in eval_params

    # holders for building sparse
    global_rows, global_cols, global_vals = [], [], []
    # For overlap IoU merge we need a small existing view; we won't build a dense global label.
    # Instead, for each paste window, we materialize an "existing view" from what we've already
    # appended by rasterizing only that window to a small dense array. To keep memory small,
    # we do it window-by-window.
    # We'll keep a simple list of pasted windows and their sparse coords to reconstruct a small view.

    pasted_windows = []  # list of tuples (y0,y1,x0,x1, rows_sub, cols_sub, vals_sub)

    next_label = 1

    # Setup progress bar over tiles
    pbar = None
    use_fallback_progress = False
    tiles_done = 0
    print_every = 1
    if show_progress:
        # compute total number of tiles with same logic as _tiles()
        stride = block_size - min_overlap
        y_starts = list(range(0, max(H - block_size, 0) + 1, stride)) or [0]
        x_starts = list(range(0, max(W - block_size, 0) + 1, stride)) or [0]
        if y_starts[-1] + block_size < H:
            y_starts.append(H - block_size)
        if x_starts[-1] + block_size < W:
            x_starts.append(W - block_size)
        total_tiles = len(y_starts) * len(x_starts)
        try:
            from tqdm import tqdm  # type: ignore
            pbar = tqdm(total=total_tiles, desc=progress_desc or "bin2cell.stardist", unit="tile")
        except Exception:
            use_fallback_progress = True
            print_every = max(1, total_tiles // 10) if total_tiles > 0 else 1

    for (y0, y1, x0, x1), (cy0, cy1, cx0, cx1) in _tiles(H, W, block_size, min_overlap, context):
        # crop with context
        tile = img[y0:y1, x0:x1]

        # run cellpose on tile
        # channels=[0,0] meaning single-channel grayscale
        eval_kwargs = {}
        if diameter is not None:
            eval_kwargs["diameter"] = diameter
        if mask_threshold is not None:
            # cellpose versions differ: prefer 'cellprob_threshold'; fall back to 'mask_threshold' if needed
            if "cellprob_threshold" in eval_params:
                eval_kwargs["cellprob_threshold"] = mask_threshold
            elif "mask_threshold" in eval_params:
                eval_kwargs["mask_threshold"] = mask_threshold
        if flow_threshold is not None:
            # only pass if supported by current cellpose version
            if "flow_threshold" in eval_params:
                eval_kwargs["flow_threshold"] = flow_threshold
        # forward supported extra kwargs to Cellpose eval (e.g., min_size, stitch_threshold, tile_overlap, niter, etc.)
        for _k, _v in list(kwargs.items()):
            if _k in eval_params and _k not in ("channels", "channel_axis", "progress") and _k not in eval_kwargs:
                eval_kwargs[_k] = _v

        # guard against very small tiles that can cause mask computation errors in cellpose
        if tile.shape[0] < 11 or tile.shape[1] < 11:
            masks = np.zeros(tile.shape[:2], dtype=np.int32)
            flows, styles, diams = None, None, None
        else:
            # build call-time kwargs for channel specification depending on version
            call_kwargs = {**eval_kwargs}
            # prefer channel_axis in newer versions to avoid deprecation warnings
            if supports_channel_axis:
                call_kwargs["channel_axis"] = None  # 2D grayscale image
            elif supports_channels:
                call_kwargs["channels"] = [0, 0]
            try:
                _out = cp.eval(tile, **call_kwargs)
                # robustly unpack depending on cellpose version
                if isinstance(_out, tuple):
                    if len(_out) == 4:
                        masks, flows, styles, diams = _out
                    elif len(_out) == 3:
                        masks, flows, styles = _out
                        diams = None
                    elif len(_out) == 2:
                        masks, flows = _out
                        styles = diams = None
                    else:
                        masks = _out[0]
                        flows = styles = diams = None
                else:
                    masks = _out
                    flows = styles = diams = None
            except Exception as e:
                # safe fallback: yield empty mask for this tile
                print(f"[bin2cell.stardist] Cellpose eval failed on a tile: {e}. Using empty mask for this tile.")
                masks = np.zeros(tile.shape[:2], dtype=np.int32)
                flows, styles, diams = None, None, None

        # cut back to core (remove context)
        core_y0, core_y1 = cy0 - y0, cy1 - y0
        core_x0, core_x1 = cx0 - x0, cx1 - x0
        core_mask = masks[core_y0:core_y1, core_x0:core_x1]

        # Build a small dense "existing" view just for the paste window to compute IoU
        existing = None
        if pasted_windows:
            # rasterize the existing sparse pixels intersecting this paste window
            ph, pw = (cy1 - cy0), (cx1 - cx0)
            existing = np.zeros((ph, pw), dtype=np.int32)
            for (py0, py1, px0, px1, r, c, v) in pasted_windows:
                # intersection with current paste window
                iy0 = max(py0, cy0)
                ix0 = max(px0, cx0)
                iy1 = min(py1, cy1)
                ix1 = min(px1, cx1)
                if iy0 >= iy1 or ix0 >= ix1:
                    continue
                # select those coords that fall into intersection
                sel = (r >= iy0) & (r < iy1) & (c >= ix0) & (c < ix1)
                if not np.any(sel):
                    continue
                rr = r[sel] - cy0
                cc = c[sel] - cx0
                existing[rr, cc] = v[sel]

        # accumulate sparse coordinates with IoU-based relabel/merge
        next_label = _paste_with_relabel(
            global_rows, global_cols, global_vals,
            core_mask,
            (cy0, cy1, cx0, cx1),
            existing,
            next_label_start=next_label,
            iou_thresh=iou_merge_threshold,
        )

        # cache what we just pasted (for future "existing" rasterization)
        r = global_rows[-1] if global_rows else np.array([], dtype=np.int32)
        c = global_cols[-1] if global_cols else np.array([], dtype=np.int32)
        v = global_vals[-1] if global_vals else np.array([], dtype=np.int32)
        pasted_windows.append((cy0, cy1, cx0, cx1, r, c, v))

        # progress update
        if pbar is not None:
            pbar.update(1)
        elif use_fallback_progress:
            tiles_done += 1
            if (tiles_done % print_every == 0) or (tiles_done == total_tiles):
                print(f"[bin2cell.stardist] Processed {tiles_done}/{total_tiles} tiles")

    # close progress bar if used
    if pbar is not None:
        pbar.close()

    # Build global sparse CSR (H x W) without ever materializing dense full image
    if len(global_rows) == 0:
        labels_csr = sparse.csr_matrix((H, W), dtype=np.int32)
    else:
        rows = np.concatenate(global_rows)
        cols = np.concatenate(global_cols)
        vals = np.concatenate(global_vals)
        labels_csr = sparse.csr_matrix((vals, (rows, cols)), shape=(H, W), dtype=np.int32)

    sparse.save_npz(labels_npz_path, labels_csr)


def view_stardist_labels(image_path, labels_npz_path, crop, **kwargs):
    '''
    Use StarDist's label rendering to view segmentation results in a crop 
    of the input image.
    
    Obsoleted by ``b2c.view_labels()``.
    
    Input
    -----
    image_path : ``filepath``
        Path to image that was segmented.
    labels_npz_path : ``filepath``
        Path to sparse labels generated by ``b2c.stardist()``.
    crop : tuple of ``int``
        A PIL-formatted crop specification - a four integer tuple, 
        provided as (left, upper, right, lower) coordinates.
    kwargs
        Any additional arguments to pass to StarDist's ``render_labels()``. 
        Practically most likely to be ``normalize_img``.
    '''
    #PIL is better at handling crops memory efficiently than cv2
    import cv2
    img = Image.open(image_path)
    img = img.crop(crop)
    #stardist likes images on a 0-1 scale
    img = np.array(img)/255
    #load labels and subset to area of interest
    #crop is (left, upper, right, lower)
    #https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.crop
    labels_sparse = scipy.sparse.load_npz(labels_npz_path)
    #upper:lower, left:right
    labels_sparse = labels_sparse[crop[1]:crop[3], crop[0]:crop[2]]
    #reset labels within crop to start from 1 and go up by 1
    #which makes the stardist view more colourful in objects
    #calling this on [5,7,7,9] yields [1,2,2,3] which is what we want
    labels_sparse.data = scipy.stats.rankdata(labels_sparse.data, method="dense")
    labels = np.array(labels_sparse.todense())
    return render_label(labels, img=img, **kwargs)

def view_labels(image_path, labels_npz_path, 
                crop=None, 
                stardist_normalize=False, 
                fill=False, 
                border=True, 
                fill_palette=None, 
                fill_label_weight=0.5, 
                border_color=[255,255,0]
               ):
    '''
    Render segmentation results in a lightweight manner on a full image 
    level. Can do fills or borders. Returns image as ``np.array``.
    
    Input
    -----
    image_path : ``filepath``
        Path to image that was segmented.
    labels_npz_path : ``filepath``
        Path to sparse labels generated by ``b2c.stardist()``.
    crop : ``None`` or tuple of ``int``, optional (default: ``None``)
        A PIL-formatted crop specification - a four integer tuple, 
        provided as (left, upper, right, lower) coordinates. If ``None``, 
        will render full segmentation results.
    stardist_normalize : ``bool``, optional (default: ``False``)
        If ``True``, percentile normalise the input image prior to rendering 
        the segmentation labels on it.
    fill : ``bool``, optional (default: ``False``)
        If ``True``, render the objects in full, with transparency 
        controlled by ``fill_label_weight``.
    border : ``bool``, optional (default, ``True``)
        If ``True``, render a fully opaque border of a specified colour 
        around the objects
    fill_palette : ``None`` or ``np.array`` of ``np.uint8``, optional (default: ``None``)
        Two-dimensional ``np.array``, with rows corresponding to colours of 
        the palette and columns corresponding to RGB, in ``np.uint8``. If 
        ``None``, will use Seaborn's bright palette, skipping the pink due 
        to it rendering poorly against H&E images. Objects will be coloured 
        based on the remainder of dividing its label ID by the number of 
        colours.
    fill_label_weight : ``float``, optional (default: 0.5)
        Weight to assign the object fill render when constructing output. 
        0 is completely transparent, 1 is completely opaque.
    border_color : ``list`` of ``np.uint8``, optional (default: ``[255, 255, 0]``)
        Border colour in RGB.
    '''
    #load the sparse labels
    import skimage.segmentation
    import skimage
    labels_sparse = scipy.sparse.load_npz(labels_npz_path)
    #determine memory efficient dtype to load the image as
    #if we'll be normalising, we want np.float16 for optimal RAM footprint
    #otherwise use np.uint8
    if stardist_normalize:
        dtype = np.float16
    else:
        dtype = np.uint8
    if crop is None:
        #this will load greyscale as 3 channel, which is what we want here
        img = load_image(image_path, dtype=dtype)
    else:
        #PIL is better at handling crops memory efficiently than cv2
        img = Image.open(image_path)
        #ensure that it's in RGB (otherwise there's a single channel for greyscale)
        img = np.array(img.crop(crop).convert('RGB'), dtype=dtype)
        #subset labels to area of interest
        #crop is (left, upper, right, lower)
        #https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.crop
        #upper:lower, left:right
        labels_sparse = labels_sparse[crop[1]:crop[3], crop[0]:crop[2]]
    #optionally normalise image
    if stardist_normalize:
        img = normalize(img)
        #actually cap the values - currently there are sub 0 and above 1 entries
        img[img<0] = 0
        img[img>1] = 1
        #turn back to uint8 for internal consistency
        img = (255*img).astype(np.uint8)
    #turn labels to COO for ease of position retrieval
    labels_sparse = labels_sparse.tocoo()
    if fill:
        if fill_palette is None:
            #use the seaborn bright palette, but remove the pink from it
            #as it blends too much into the H&E background
            fill_palette = (np.array(sns.color_palette("bright"))*255).astype(np.uint8)
            fill_palette = np.delete(fill_palette, 6, 0)
        #now we have a master list of pixels with objects to show
        #.row is [:,0], .col is [:,1]
        #extract the existing values from the image
        #and simultaneously get a fill colour by doing % on number of fill colours
        #weight the two together to get the new pixel value
        img[labels_sparse.row, labels_sparse.col, :] = \
            (1-fill_label_weight) * img[labels_sparse.row, labels_sparse.col, :] + \
            fill_label_weight * fill_palette[labels_sparse.data % fill_palette.shape[0], :]
    if border:
        #unfortunately the boundary finder wants a dense matrix, so turn our labels to it for a sec
        #turn the output back into a sparse COO, both for memory efficiency and pure convenience
        border_sparse = scipy.sparse.coo_matrix(skimage.segmentation.find_boundaries(np.array(labels_sparse.todense())))
        #can now easily colour the borders similar to what was done for the fill
        img[border_sparse.row, border_sparse.col, :] = border_color
    return img

def overlay_onto_img(img, labels_sparse, cdata, key, common_objects, 
                      fill_label_weight=1, 
                      cat_cmap="tab20",
                      cont_cmap="viridis"
                     ):
    '''
    Helper function used by ``b2c.view_cell_labels()`` to actually overlay 
    the metadata/expression onto the morphology image
    
    Input
    -----
    image_path : ``numpy.array``
        Morphology image in Numpy array form.
    labels_sparse : ``scipy.sparse.coo_matrix``
        Processed to just the pixels to render, 
    cdata : ``AnnData``
        Cell-level VisiumHD object with pertinent gene/metadata. Must have 
        ``.obs_names`` unchanged from ``b2c.bin_to_cell()`` output.
    key : ``str``
        ``.obs`` column (float/integer or categorical) or gene name 
        (expression taken from ``.X``) to colour the ``labels_sparse`` 
        pixels by.
    common_objects : ``numpy.array`` of ``int``
        Object IDs to plot.
    fill_label_weight : ``float``, optional (default: 1)
        Weight to assign the object fill render when constructing output. 
        0 is completely transparent, 1 is completely opaque.
    cat_cmap : ``str``, optional (default: ``"tab20"``)
        Colormap name (must be understood by ``seaborn.color_palette()``) to 
        use for categoricals.
    cont_cmap : ``str``, optional (default: ``"viridis"``)
        Colormap name (must be understood by ``matplotlib.colormaps.get_cmap()``) 
        to use for integers/floats.
    '''
    #are we in obs or in var?
    if key in cdata.obs.columns:
        #continuous or categorical?
        if ("float" in cdata.obs[key].dtype.name) or ("int" in cdata.obs[key].dtype.name):
            #we've got a continous
            #subset on common_objects, so the order matches them
            #also need to turn to string to match cdata.obs_names
            vals = cdata.obs.loc[[str(i) for i in common_objects], key].values
            #we'll continue processing shortly outside the ifs
        elif "category" in cdata.obs[key].dtype.name:
            #we've got a categorical
            #pull out the category indices for each of the common objects
            #and store them in a numpy array that can be indexed on the object in int form
            cats = np.zeros(np.max(common_objects)+1, dtype=np.int32)
            #need to turn common_objects back to string so cdata can be subset on them
            cats[common_objects] = cdata.obs.loc[[str(i) for i in common_objects], key].cat.codes
            #store the original present category codes for legend purposes
            cats_unique_original = np.unique(cats[common_objects])
            #reset the cats[common_objects] to start from 0 and go up by 1
            #which may avoid some palette overlaps in some corner case scenarios
            #calling this on [5,7,7,9] yields [1,2,2,3] which is what we want
            #except then shift it back by 1 so it starts from 0
            cats[common_objects] = scipy.stats.rankdata(cats[common_objects], method="dense") - 1
            #make a cmap with the correct number of colours
            #convert to uint8 for internal consistency
            fill_palette = (np.array(sns.color_palette(cat_cmap, n_colors=len(cats_unique_original)))*255).astype(np.uint8)
            #now we have a master list of pixels with objects to show
            #.row is [:,0], .col is [:,1]
            #extract the existing values from the image
            #pull out the category index by subsetting on the actual object ID
            #no need to do % because we have one entry in the palette per category
            #weight the two together to get the new pixel value
            img[labels_sparse.row, labels_sparse.col, :] = \
                (1-fill_label_weight) * img[labels_sparse.row, labels_sparse.col, :] + \
                fill_label_weight * fill_palette[cats[labels_sparse.data], :]
            #set up legend
            #figsize is largely irrelevant because of bbox_inches='tight' when saving
            fig, ax = plt.subplots(figsize=(5, 2))
            ax.axis("off")
            #alright, there's some stuff going on here. let's explain
            #the categories don't actually get subset when you pull out a chunk of the vector
            #which kinda makes sense but is annoying
            #meanwhile we want to minimise our ID footprint to try to use the colour map nicely
            #as such, earlier we made cats_unique_original, having all the codes from the objects
            #and then we reset the cats to start at 0 and go up by 1
            #now we can get a correspondence of those to cats_unique_original
            #by doing a subset and np.unique(), as this is monotonically preserved
            #once again, no need for % because we have an entry for each category
            legend_patches = [
                matplotlib.patches.Patch(color=fill_palette[i,:]/255.0, 
                                         label=cdata.obs[key].cat.categories[j]
                                        )
                for i, j in zip(np.unique(cats[common_objects]), cats_unique_original)
            ]
            ax.legend(handles=legend_patches, loc="center", title=key, frameon=False)
            #close the thing so it doesn't randomly show. still there though
            plt.close(fig)
            #okay we're happy. return to recycle continuous processing code
            return img, fig
        else:
            #we've got to raise an error
            raise ValueError("``cdata.obs['"+key+"']`` must be a float, int, or categorical")
    elif key in cdata.var_names:
        #gene, continuous
        #fast enough to just subset the cdata, as always turn to string
        #then regardless if it's sparse or dense data need to .toarray().flatten()
        #if it's sparse then this turns it dense
        #if it's dense then it gets out of ArrayView back into a normal array
        #and then flattened regardless
        vals = cdata[[str(i) for i in common_objects]][:,key].X.toarray().flatten()
    else:
        #we've got a whiff
        raise ValueError("'"+key+"' not found in ``cdata.obs`` or ``cdata.var``")
    #we're out here. so we're processing a continuous, be it obs or var
    #set up legend while vals are on their actual scale
    #figsize is largely irrelevant because of bbox_inches='tight' when saving
    fig, ax = plt.subplots(figsize=(5, 2))
    ax.axis("off")
    colormap = matplotlib.colormaps.get_cmap(cont_cmap)
    norm = matplotlib.colors.Normalize(vmin=np.min(vals), vmax=np.max(vals))
    sm = matplotlib.cm.ScalarMappable(cmap=colormap, norm=norm)
    fig.colorbar(sm, ax=ax, orientation="horizontal", label=key)
    #close the thing so it doesn't randomly show. still there though
    plt.close(fig)
    #for continuous operations, we need a 0-1 scaled vector of values
    vals = (vals-np.min(vals))/(np.max(vals)-np.min(vals))
    #construct a fill palette by busting out a colormap
    #and then getting its values at vals, and sending them to a prepared numpy array
    #that can then be subset on the object ID to get its matching RGB of the continuous value
    fill_palette = np.zeros((np.max(common_objects)+1, 3))
    #send to common_objects, matching vals order. also convert to uint8 for consistency
    fill_palette[common_objects, :] = (colormap(vals)[:,:3]*255).astype(np.uint8)
    #now we have a master list of pixels with objects to show
    #.row is [:,0], .col is [:,1]
    #extract the existing values from the image
    #and simultaneously get a fill colour by subsetting the palette on the label ID
    #no fancy % needed here as each object has its own fill value prepared
    #weight the two together to get the new pixel value
    img[labels_sparse.row, labels_sparse.col, :] = \
        (1-fill_label_weight) * img[labels_sparse.row, labels_sparse.col, :] + \
        fill_label_weight * fill_palette[labels_sparse.data, :]
    return img, fig
    
    
def view_cell_labels(image_path, labels_npz_path, cdata, 
                     fill_key=None, 
                     border_key=None, 
                     crop=None, 
                     stardist_normalize=False, 
                     fill_label_weight=1, 
                     thicken_border=True,
                     cat_cmap="tab20",
                     cont_cmap="viridis"
                    ):
    '''
    Colour morphology segmentations by cell-level metadata or gene 
    expression. Can do fills or borders. Returns image as ``np.array``, 
    and a dictionary of legends (as matplotlib figures) named after the 
    keys it used.
    
    Input
    -----
    image_path : ``filepath``
        Path to morphology image that was segmented.
    labels_npz_path : ``filepath``
        Path to sparse morphology labels generated by ``b2c.stardist()``.
    cdata : ``AnnData``
        Cell-level VisiumHD object with pertinent gene/metadata. Must have 
        ``.obs_names`` unchanged from ``b2c.bin_to_cell()`` output.
    fill_key : ``str`` or ``None``, optional (default: ``None``)
        ``.obs`` column (float/integer or categorical) or gene name 
        (expression taken from ``.X``) to colour the fill by. 
        Skipped if ``None``.
    border_key : ``str`` or ``None``, optional (default: ``None``)
        ``.obs`` column (float/integer or categorical) or gene name 
        (expression taken from ``.X``) to colour the border by. 
        Skipped if ``None``.
    crop : ``None`` or tuple of ``int``, optional (default: ``None``)
        A PIL-formatted crop specification - a four integer tuple, 
        provided as (left, upper, right, lower) coordinates. If ``None``, 
        will render full segmentation results.
    stardist_normalize : ``bool``, optional (default: ``False``)
        If ``True``, percentile normalise the input image prior to rendering 
        the segmentation labels on it.
    fill_label_weight : ``float``, optional (default: 1)
        Weight to assign the object fill render when constructing output. 
        0 is completely transparent, 1 is completely opaque.
    thicken_border : ``bool``, optional (default: ``True``)
        The default identified border is thinner than that of 
        ``b2c.view_labels()``. Setting this to ``True``, thickens it for 
        ease of viewing.
    cat_cmap : ``str``, optional (default: ``"tab20"``)
        Colormap name (must be understood by ``seaborn.color_palette()``) to 
        use for categoricals.
    cont_cmap : ``str``, optional (default: ``"viridis"``)
        Colormap name (must be understood by ``matplotlib.colormaps.get_cmap()``) 
        to use for integers/floats.
    '''
    #load the sparse labels
    import skimage.segmentation
    import skimage
    labels_sparse = scipy.sparse.load_npz(labels_npz_path)
    #determine memory efficient dtype to load the image as
    #if we'll be normalising, we want np.float16 for optimal RAM footprint
    #otherwise use np.uint8
    if stardist_normalize:
        dtype = np.float16
    else:
        dtype = np.uint8
    if crop is None:
        #this will load greyscale as 3 channel, which is what we want here
        img = load_image(image_path, dtype=dtype)
    else:
        #PIL is better at handling crops memory efficiently than cv2
        img = Image.open(image_path)
        #ensure that it's in RGB (otherwise there's a single channel for greyscale)
        img = np.array(img.crop(crop).convert('RGB'), dtype=dtype)
        #subset labels to area of interest
        #crop is (left, upper, right, lower)
        #https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.crop
        #upper:lower, left:right
        labels_sparse = labels_sparse[crop[1]:crop[3], crop[0]:crop[2]]
    #optionally normalise image
    if stardist_normalize:
        img = normalize(img)
        #actually cap the values - currently there are sub 0 and above 1 entries
        img[img<0] = 0
        img[img>1] = 1
        #turn back to uint8 for internal consistency
        img = (255*img).astype(np.uint8)
    #turn labels to COO for ease of position retrieval
    labels_sparse = labels_sparse.tocoo()
    #identify overlap between label objects and cdata observations
    #which should be the same nomenclature for the morphology segmentation
    #just as strings, while the labels are ints
    common_objects = np.sort(list(set(np.unique(labels_sparse.data)).intersection(set([int(i) for i in cdata.obs_names]))))
    #kick out filtered out objects from segmentation results
    labels_sparse.data[~np.isin(labels_sparse.data, common_objects)] = 0
    labels_sparse.eliminate_zeros()
    #catch legends to dictionary for returning later
    legends = {}
    #do a fill if requested
    if fill_key is not None:
        #legend comes in the form of a figure
        img, fig = overlay_onto_img(img=img, 
                                    labels_sparse=labels_sparse, 
                                    cdata=cdata, 
                                    key=fill_key, 
                                    common_objects=common_objects, 
                                    fill_label_weight=fill_label_weight,
                                    cat_cmap=cat_cmap,
                                    cont_cmap=cont_cmap
                                   )
        legends[fill_key] = fig
    #do a border if requested
    if border_key is not None:
        #actually get the border
        #unfortunately the boundary finder wants a dense matrix, so turn our labels to it for a sec
        #go for inner borders as that's all we care about, and that's less pixels to worry about
        #keep the border dense because of implementation reasons. it spikes RAM anyway when it's made
        border = skimage.segmentation.find_boundaries(np.array(labels_sparse.todense()), mode="inner")
        #whether we thicken or not, we need nonzero coordinates
        coords = np.nonzero(border)
        if thicken_border:
            #we're thickening. skimage.segmentation.expand_labels() explodes when asked to do this
            #the following gets the job done quicker and with lower RAM footprint
            #take existing nonzero coordinates and move them to the left, right, up and down by 1
            border_rows = np.hstack([
                np.clip(coords[0]-1, a_min=0, a_max=None),
                np.clip(coords[0]+1, a_min=None, a_max=border.shape[0]-1),
                coords[0],
                coords[0]
            ])
            border_cols = np.hstack([
                coords[1],
                coords[1],
                np.clip(coords[1]-1, a_min=0, a_max=None),
                np.clip(coords[1]+1, a_min=None, a_max=border.shape[1]-1)
            ])
            #set the positions to True. this is BLAZING FAST compared to sparse attempts
            #or, surprisingly, trying to np.unique() to get non-duplicate coordinates
            #and the entire reason we kept the borders dense for this process
            border[border_rows, border_cols] = True
            #update our nonzero coordinates
            coords = np.nonzero(border)
        #to assign borders back to objects, subset the object labels to just the border pixels
        #technically need to construct a new COO matrix for it, pulling out values at the border coordinates
        #use (data, (row, col)) constructor
        #also need to turn labels to CSR to be able to pull out their values
        #which results in a 2d numpy matrix, so turn to 1D array or constructor errors
        labels_sparse = scipy.sparse.coo_matrix((np.array(labels_sparse.tocsr()[coords[0], coords[1]]).flatten(), coords), shape=labels_sparse.shape)
        #there will be zeroes from the thickener, border mode="inner" means no thickener no zeros
        labels_sparse.eliminate_zeros()
        #can now run the overlayer, set weights to 1 to have fully opaque borders
        #legend comes in the form of a figure
        img, fig = overlay_onto_img(img=img, 
                                    labels_sparse=labels_sparse, 
                                    cdata=cdata, 
                                    key=border_key, 
                                    common_objects=common_objects, 
                                    fill_label_weight=1,
                                    cat_cmap=cat_cmap,
                                    cont_cmap=cont_cmap
                                   )
        legends[border_key] = fig
    return img, legends

#as PR'd to scanpy: https://github.com/scverse/scanpy/pull/2992
def read_visium(
    path: Path | str,
    genome: str | None = None,
    *,
    count_file: str = "filtered_feature_bc_matrix.h5",
    library_id: str | None = None,
    load_images: bool | None = True,
    source_image_path: Path | str | None = None,
    spaceranger_image_path: Path | str | None = None,
) -> AnnData:
    """\
    Read 10x-Genomics-formatted visum dataset.

    In addition to reading regular 10x output,
    this looks for the `spatial` folder and loads images,
    coordinates and scale factors.
    Based on the `Space Ranger output docs`_.

    See :func:`~scanpy.pl.spatial` for a compatible plotting function.

    .. _Space Ranger output docs: https://support.10xgenomics.com/spatial-gene-expression/software/pipelines/latest/output/overview

    Parameters
    ----------
    path
        Path to directory for visium datafiles.
    genome
        Filter expression to genes within this genome.
    count_file
        Which file in the passed directory to use as the count file. Typically would be one of:
        'filtered_feature_bc_matrix.h5' or 'raw_feature_bc_matrix.h5'.
    library_id
        Identifier for the visium library. Can be modified when concatenating multiple adata objects.
    source_image_path
        Path to the high-resolution tissue image. Path will be included in
        `.uns["spatial"][library_id]["metadata"]["source_image_path"]`.
    spaceranger_image_path
        Path to the folder containing the spaceranger output hires/lowres tissue images. If `None`, 
        will go with the `spatial` folder of the provided `path`.

    Returns
    -------
    Annotated data matrix, where observations/cells are named by their
    barcode and variables/genes by gene name. Stores the following information:

    :attr:`~anndata.AnnData.X`
        The data matrix is stored
    :attr:`~anndata.AnnData.obs_names`
        Cell names
    :attr:`~anndata.AnnData.var_names`
        Gene names for a feature barcode matrix, probe names for a probe bc matrix
    :attr:`~anndata.AnnData.var`\\ `['gene_ids']`
        Gene IDs
    :attr:`~anndata.AnnData.var`\\ `['feature_types']`
        Feature types
    :attr:`~anndata.AnnData.obs`\\ `[filtered_barcodes]`
        filtered barcodes if present in the matrix
    :attr:`~anndata.AnnData.var`
        Any additional metadata present in /matrix/features is read in.
    :attr:`~anndata.AnnData.uns`\\ `['spatial']`
        Dict of spaceranger output files with 'library_id' as key
    :attr:`~anndata.AnnData.uns`\\ `['spatial'][library_id]['images']`
        Dict of images (`'hires'` and `'lowres'`)
    :attr:`~anndata.AnnData.uns`\\ `['spatial'][library_id]['scalefactors']`
        Scale factors for the spots
    :attr:`~anndata.AnnData.uns`\\ `['spatial'][library_id]['metadata']`
        Files metadata: 'chemistry_description', 'software_version', 'source_image_path'
    :attr:`~anndata.AnnData.obsm`\\ `['spatial']`
        Spatial spot coordinates, usable as `basis` by :func:`~scanpy.pl.embedding`.
    """
    
    path = Path(path)
    #if not provided, assume the hires/lowres images are in the same folder as everything
    #except in the spatial subdirectory
    if spaceranger_image_path is None:
        spaceranger_image_path = path / "spatial"
    else:
        spaceranger_image_path = Path(spaceranger_image_path)
    adata = read_10x_h5(path / count_file, genome=genome)

    adata.uns["spatial"] = dict()

    from h5py import File

    with File(path / count_file, mode="r") as f:
        attrs = dict(f.attrs)
    if library_id is None:
        library_id = str(attrs.pop("library_ids")[0], "utf-8")

    adata.uns["spatial"][library_id] = dict()

    if load_images:
        tissue_positions_file = (
            path / "spatial/tissue_positions.csv"
            if (path / "spatial/tissue_positions.csv").exists()
            else path / "spatial/tissue_positions.parquet" if (path / "spatial/tissue_positions.parquet").exists()
            else path / "spatial/tissue_positions_list.csv"
        )
        files = dict(
            tissue_positions_file=tissue_positions_file,
            scalefactors_json_file=path / "spatial/scalefactors_json.json",
            hires_image=spaceranger_image_path / "tissue_hires_image.png",
            lowres_image=spaceranger_image_path / "tissue_lowres_image.png",
        )

        # check if files exists, continue if images are missing
        for f in files.values():
            if not f.exists():
                if any(x in str(f) for x in ["hires_image", "lowres_image"]):
                    logg.warning(
                        f"You seem to be missing an image file.\n"
                        f"Could not find '{f}'."
                    )
                else:
                    raise OSError(f"Could not find '{f}'")

        adata.uns["spatial"][library_id]["images"] = dict()
        for res in ["hires", "lowres"]:
            try:
                adata.uns["spatial"][library_id]["images"][res] = imread(
                    str(files[f"{res}_image"])
                )
            except Exception:
                raise OSError(f"Could not find '{res}_image'")

        # read json scalefactors
        adata.uns["spatial"][library_id]["scalefactors"] = json.loads(
            files["scalefactors_json_file"].read_bytes()
        )

        adata.uns["spatial"][library_id]["metadata"] = {
            k: (str(attrs[k], "utf-8") if isinstance(attrs[k], bytes) else attrs[k])
            for k in ("chemistry_description", "software_version")
            if k in attrs
        }

        # read coordinates
        if files["tissue_positions_file"].name.endswith(".csv"):
            positions = pd.read_csv(
                files["tissue_positions_file"],
                header=0 if tissue_positions_file.name == "tissue_positions.csv" else None,
                index_col=0,
            )
        elif files["tissue_positions_file"].name.endswith(".parquet"):
            positions = pd.read_parquet(files["tissue_positions_file"])
            #need to set the barcode to be the index
            positions.set_index("barcode", inplace=True)
        positions.columns = [
            "in_tissue",
            "array_row",
            "array_col",
            "pxl_col_in_fullres",
            "pxl_row_in_fullres",
        ]

        adata.obs = adata.obs.join(positions, how="left")

        adata.obsm["spatial"] = adata.obs[
            ["pxl_row_in_fullres", "pxl_col_in_fullres"]
        ].to_numpy()
        adata.obs.drop(
            columns=["pxl_row_in_fullres", "pxl_col_in_fullres"],
            inplace=True,
        )

        # put image path in uns
        if source_image_path is not None:
            # get an absolute path
            source_image_path = str(Path(source_image_path).resolve())
            adata.uns["spatial"][library_id]["metadata"]["source_image_path"] = str(
                source_image_path
            )

    return adata

def destripe_counts(adata, counts_key="n_counts", adjusted_counts_key="n_counts_adjusted"):
    '''
    Scale each row (bin) of ``adata.X`` to have ``adjusted_counts_key`` 
    rather than ``counts_key`` total counts.
    
    Input
    -----
    adata : ``AnnData``
        2um bin VisiumHD object. Raw counts, needs to have ``counts_key`` 
        and ``adjusted_counts_key`` in ``.obs``.
    counts_key : ``str``, optional (default: ``"n_counts"``)
        Name of ``.obs`` column with raw counts per bin.
    adjusted_counts_key : ``str``, optional (default: ``"n_counts_adjusted"``)
        Name of ``.obs`` column storing the desired destriped counts per bin.
    '''
    #scanpy's utility function to make sure the anndata is not a view
    #if it is a view then weird stuff happens when you try to write to its .X
    sc._utils.view_to_actual(adata)
    #adjust the count matrix to have n_counts_adjusted sum per bin (row)
    #premultiplying by a diagonal matrix multiplies each row by a value: https://solitaryroad.com/c108.html
    bin_scaling = scipy.sparse.diags(adata.obs[adjusted_counts_key]/adata.obs[counts_key])
    adata.X = bin_scaling.dot(adata.X)

def destripe(adata, quantile=0.99, counts_key="n_counts", factor_key="destripe_factor", adjusted_counts_key="n_counts_adjusted", adjust_counts=True):
    '''
    Correct the raw counts of the input object for known variable width of 
    VisiumHD 2um bins. Scales the total UMIs per bin on a per-row and 
    per-column basis, dividing by the specified ``quantile``. The resulting 
    value is stored in ``.obs[factor_key]``, and is multiplied by the 
    corresponding total UMI ``quantile`` to get ``.obs[adjusted_counts_key]``.
    
    Input
    -----
    adata : ``AnnData``
        2um bin VisiumHD object. Raw counts, needs to have ``counts_key`` in 
        ``.obs``.
    quantile : ``float``, optional (default: 0.99)
        Which row/column quantile to use for the computation.
    counts_key : ``str``, optional (default: ``"n_counts"``)
        Name of ``.obs`` column with raw counts per bin.
    factor_key : ``str``, optional (default: ``"destripe_factor"``)
        Name of ``.obs`` column to hold computed factor prior to reversing to 
        count space.
    adjusted_counts_key : ``str``, optional (default: ``"n_counts_adjusted"``)
        Name of ``.obs`` column for storing the destriped counts per bin.
    adjust_counts : ``bool``, optional (default: ``True``)
        Whether to use the computed adjusted count total to adjust the counts in 
        ``adata.X``.
    '''
    #apply destriping via sequential quantile scaling
    #get specified quantile per row
    quant = adata.obs.groupby("array_row")[counts_key].quantile(quantile)
    #divide each row by its quantile (order of obs[counts_key] and obs[array_row] match)
    adata.obs[factor_key] = adata.obs[counts_key] / adata.obs["array_row"].map(quant)
    #repeat on columns
    quant = adata.obs.groupby("array_col")[factor_key].quantile(quantile)
    adata.obs[factor_key] /= adata.obs["array_col"].map(quant)
    #propose adjusted counts as the global quantile multipled by the destripe factor
    adata.obs[adjusted_counts_key] = adata.obs[factor_key] * np.quantile(adata.obs[counts_key], quantile)
    #correct the count space unless told not to
    if adjust_counts:
        destripe_counts(adata, counts_key=counts_key, adjusted_counts_key=adjusted_counts_key)

def check_array_coordinates(adata, row_max=3349, col_max=3349):
    '''
    Assess the relationship between ``.obs["array_row"]``/``.obs["array_col"]`` 
    and ``.obsm["spatial"]``, as the array coordinates tend to have their 
    origin in places that isn't the top left of the image. Array coordinates 
    are deemed to be flipped or not based on the Pearson correlation with the 
    corresponding spatial dimension. Creates ``.uns["bin2cell"]["array_check"]`` 
    that is used by ``b2c.grid_image()``, ``b2c.insert_labels()`` and 
    ``b2c.get_crop()``.
    
    Input
    -----
    adata : ``AnnData``
        2um bin VisiumHD object.
    row_max : ``int``, optional (default: 3349)
        Maximum possible ``array_row`` value.
    col_max : ``int``, optional (default: 3349)
        Maximum possible ``array_col`` value.
    '''
    #store the calls here
    if not "bin2cell" in adata.uns:
        adata.uns["bin2cell"] = {}
    adata.uns["bin2cell"]["array_check"] = {}
    #we'll need to check both the rows and columns
    for axis in ["row", "col"]:
        #we may as well store the maximum immediately
        adata.uns["bin2cell"]["array_check"][axis] = {}
        if axis == "row":
            adata.uns["bin2cell"]["array_check"][axis]["max"] = row_max
        elif axis == "col":
            adata.uns["bin2cell"]["array_check"][axis]["max"] = col_max
        #are we going to be extracting values for a single col or row?
        #set up where we'll be looking to get values to correlate
        if axis == "col":
            single_axis = "row"
            #spatial[:,0] matches axis_col (note at start)
            spatial_axis = 0
        elif axis == "row":
            single_axis = "col"
            #spatial[:,1] matches axis_row (note at start)
            spatial_axis = 1
        #get the value of the other axis with the highest number of bins present
        val = adata.obs['array_'+single_axis].value_counts().index[0]
        #get a boolean mask of the bins of that value
        mask = (adata.obs['array_'+single_axis] == val)
        #use the mask to get the spatial and array coordinates to compare
        array_vals = adata.obs.loc[mask,'array_'+axis].values
        spatial_vals = adata.obsm['spatial'][mask, spatial_axis]
        #check whether they're positively or negatively correlated
        if scipy.stats.pearsonr(array_vals, spatial_vals)[0] < 0:
            adata.uns["bin2cell"]["array_check"][axis]["flipped"] = True
        else:
            adata.uns["bin2cell"]["array_check"][axis]["flipped"] = False

def grid_image(adata, val, log1p=False, mpp=2, sigma=None, save_path=None):
    '''
    Create an image of a specified ``val`` across the array coordinate grid. 
    Orientation matches the morphology image and spatial coordinates.
    
    Input
    -----
    adata : ``AnnData``
        2um bin VisiumHD object. Should have array coordinates evaluated by 
        calling ``b2c.check_array_coordinates()``.
    val : ``str``
        ``.obs`` column or variable name to visualise.
    log1p : ``bool``, optional (default: ``False``)
        Whether to log1p-transform the values in the image.
    mpp : ``float``, optional (default: 2)
        Microns per pixel of the output image.
    sigma : ``float`` or ``None``, optional (default: ``None``)
        If not ``None``, will run the final image through 
        ``skimage.filters.gaussian()`` with the provided sigma value.
    save_path : ``filepath`` or ``None``, optional (default: ``None``)
        If specified, will save the generated image to this path (e.g. for 
        StarDist use). If not provided, will return image.
    '''
    #pull out the values for the image. start by checking .obs
    import skimage
    import cv2
    if val in adata.obs.columns:
        vals = adata.obs[val].values.copy()
    elif val in adata.var_names:
        #if not in obs, it's presumably in the feature space
        vals = adata[:, val].X
        #may be sparse
        if scipy.sparse.issparse(vals):
            vals = vals.todense()
        #turn it to a flattened numpy array so it plays nice
        vals = np.asarray(vals).flatten()
    else:
        #failed to find
        raise ValueError('"'+val+'" not located in ``.obs`` or ``.var_names``')
    #make the values span from 0 to 255
    vals = (255 * (vals-np.min(vals))/(np.max(vals)-np.min(vals))).astype(np.uint8)
    #optionally log1p
    if log1p:
        vals = np.log1p(vals)
        vals = (255 * (vals-np.min(vals))/(np.max(vals)-np.min(vals))).astype(np.uint8)
    #spatial coordinates match what's going on in the image, array coordinates may not
    #have we checked if the array row/col need flipping?
    if not "bin2cell" in adata.uns:
        check_array_coordinates(adata)
    elif not "array_check" in adata.uns["bin2cell"]:
        check_array_coordinates(adata)
    #can now create an empty image the shape of the grid and stick the values in based on the coordinates
    #need to nudge up the dimensions by 1 as python is zero-indexed
    img = np.zeros((adata.uns["bin2cell"]["array_check"]["row"]["max"]+1, 
                    adata.uns["bin2cell"]["array_check"]["col"]["max"]+1), 
                   dtype=np.uint8)
    img[adata.obs['array_row'], adata.obs['array_col']] = vals
    #check if the row or column need flipping
    if adata.uns["bin2cell"]["array_check"]["row"]["flipped"]:
        img = np.flip(img, axis=0)
    if adata.uns["bin2cell"]["array_check"]["col"]["flipped"]:
        img = np.flip(img, axis=1)
    #resize image to appropriate mpp. bins are 2um apart, so current mpp is 2
    #need to reverse dimensions relative to the array for cv2, and turn to int
    if mpp != 2:
        dim = np.round(np.array(img.shape) * 2/mpp).astype(int)[::-1]
        img = cv2.resize(img, dim, interpolation=cv2.INTER_CUBIC)
    #run through the gaussian filter if need be
    if sigma is not None:
        img = skimage.filters.gaussian(img, sigma=sigma)
        img = (255 * (img-np.min(img))/(np.max(img)-np.min(img))).astype(np.uint8)
    #save or return image
    if save_path is not None:
        cv2.imwrite(save_path, img)
    else:
        return img

def check_bin_image_overlap(adata, img, overlap_threshold=0.9):
    '''
    Assess the number of bins that fall within the source image coordinate 
    space. If an insufficient proportion are captured then throw an informative 
    error.
    
    Obsoleted by ``b2c.actual_vs_inferred_image_shape()``.
    
    Input
    -----
    adata : ``AnnData``
        2um bin Visium object.
    img : ``np.array``
        Loaded full resolution morphology image, prior to any cropping/scaling.
    overlap_threshold : ``float``, optional (default: 0.9)
        Throw the error if fewer than this fraction of bin spatial coordinates 
        fall within the dimensions of the image.
    '''
    #spatial[:,1] matches img[:,0] and spatial[:,0] matches img[:,1]
    #check how many fall within the dimensions, and get a fraction of total bin count
    overlap = np.sum((adata.obsm["spatial"][:,1] < img.shape[0]) & (adata.obsm["spatial"][:,0] < img.shape[1])) / adata.shape[0]
    if overlap < overlap_threshold:
        #something is amiss. print a bunch of diagnostics
        print("Source image dimensions: "+ str(img.shape))
        #the end user does not need to know about the messiness of the representations
        #pre-format the spatial maxima to match the dimensions of the image
        print("Corresponding ``.obsm['spatial']`` maxima: "+str(np.max(adata.obsm["spatial"], axis=0)[::-1]))
        raise ValueError("Only "+str(100*overlap)+"% of bins fall within image. Are you running with ``source_image_path`` set to the full resolution morphology image, as used for ``--image`` in Spaceranger?")

def actual_vs_inferred_image_shape(adata, img, ratio_threshold=0.99):
    '''
    Compare the shape of the actual morphology image versus what the shape of 
    the morphology image that was used for Spaceranger appears to be from 
    information stored for the hires. If there's a mismatch throw an 
    informative error with both sets of dimensions.
    
    Input
    -----
    adata : ``AnnData``
        2um bin Visium object.
    img : ``np.array``
        Loaded full resolution morphology image, prior to any cropping/scaling.
    ratio_threshold : ``float``, optional (default: 0.99)
        Throw the error if any ratio of corresponding actual and inferred 
        dimensions falls below this value.
    '''
    #identify name of spatial key for subsequent access of fields
    library = list(adata.uns['spatial'].keys())[0]
    #infer the dimensions as the shape of the hires tissue image
    #divided by the hires scale factor
    inferred_dim = np.array(adata.uns['spatial'][library]['images']['hires'].shape)[:2] / adata.uns['spatial'][library]['scalefactors']['tissue_hires_scalef']
    #retrieve actual dimension as we have the full morphology image loaded
    actual_dim = np.array(img.shape)[:2]
    #do the two match, within some tolerance of rounding etc?
    #divide both ways just in case
    if np.min(np.hstack((actual_dim/inferred_dim, inferred_dim/actual_dim))) < ratio_threshold:
        raise ValueError("Morphology image dimension mismatch. Dimensions inferred from Spaceranger output: "+str(inferred_dim)+", actual image dimensions: "+str(actual_dim)+". Are you running with ``source_image_path`` set to the full resolution morphology image, as used for ``--image`` in Spaceranger?")

def mpp_to_scalef(adata, mpp):
    '''
    Compute a scale factor for a specified mpp value.
    
    Input
    -----
    adata : ``AnnData``
        2um bin Visium object.
    mpp : ``float``
        Microns per pixel to report scale factor for.
    '''
    #identify name of spatial key for subsequent access of fields
    library = list(adata.uns['spatial'].keys())[0]
    #get original image mpp value
    mpp_source = adata.uns['spatial'][library]['scalefactors']['microns_per_pixel']
    #our scale factor is the original mpp divided by the new mpp
    return mpp_source/mpp

def get_mpp_coords(adata, basis="spatial", spatial_key="spatial", mpp=None):
    '''
    Get an mpp-adjusted representation of spatial or array coordinates of the 
    provided object. Origin in top left, dimensions correspond to ``np.array()`` 
    representation of image (``[:,0]`` is up-down, ``[:,1]`` is left-right). 
    The resulting coordinates are integers for ease of retrieval of labels from 
    arrays or defining crops.
    
    adata : ``AnnData``
        2um bin VisiumHD object.
    basis : ``str``, optional (default: ``"spatial"``)
        Whether to get ``"spatial"`` or ``"array"`` coordinates. The former is 
        the source morphology image, the latter is a GEX-based grid representation.
    spatial_key : ``str``, optional (default: ``"spatial"``)
        Only used with ``basis="spatial"``. Needs to be present in ``.obsm``. 
        Rounded coordinates will be used to represent each bin when retrieving 
        labels.
    mpp : ``float`` or ``None``, optional (default: ``None``)
        The mpp value. Mandatory for GEX (``basis="array"``), if not provided 
        with morphology (``basis="spatial"``) will assume full scale image.
    '''
    #if we're using array coordinates, is there an mpp provided?
    if basis == "array" and mpp is None:
        raise ValueError("Need to specify mpp if working with array coordinates.")
    if basis == "spatial":
        if mpp is not None:
            #get necessary scale factor
            scalef = mpp_to_scalef(adata, mpp=mpp)
        else:
            #no mpp implies full blown morphology image, so scalef is 1
            scalef = 1
        #get the matching coordinates, rounding to integers makes this agree
        #need to reverse them here to make the coordinates match the image, as per note at start
        #multiply by the scale factor to account for possible custom mpp morphology image
        coords = (adata.obsm[spatial_key]*scalef).astype(int)[:,::-1]
    elif basis == "array":
        #generate the pixels in the GEX image at the specified mpp
        #which actually correspond to the locations of the bins
        #easy to define scale factor as starting array mpp is 2
        scalef = 2/mpp
        coords = np.round(adata.obs[['array_row','array_col']].values*scalef).astype(int)
        #need to flip axes maybe
        #need to scale up maximum appropriately
        if adata.uns["bin2cell"]["array_check"]["row"]["flipped"]:
            coords[:,0] = np.round(adata.uns["bin2cell"]["array_check"]["row"]["max"]*scalef).astype(int) - coords[:,0]
        if adata.uns["bin2cell"]["array_check"]["col"]["flipped"]:
            coords[:,1] = np.round(adata.uns["bin2cell"]["array_check"]["col"]["max"]*scalef).astype(int) - coords[:,1]
    return coords

def get_crop(adata, basis="spatial", spatial_key="spatial", mpp=None, buffer=0):
    '''
    Get a PIL-formatted crop tuple from a provided object and coordinate 
    representation.
    
    Input
    -----
    adata : ``AnnData``
        2um bin VisiumHD object.
    basis : ``str``, optional (default: ``"spatial"``)
        Whether to use ``"spatial"`` or ``"array"`` coordinates. The former is 
        the source morphology image, the latter is a GEX-based grid representation.
    spatial_key : ``str``, optional (default: ``"spatial"``)
        Only used with ``basis="spatial"``. Needs to be present in ``.obsm``. 
        Rounded coordinates will be used to represent each bin when retrieving 
        labels.
    mpp : ``float`` or ``None``, optional (default: ``None``)
        The micron per pixel value to use. Mandatory for GEX (``basis="array"``), 
        if not provided with morphology (``basis="spatial"``) will assume full scale 
        image.
    buffer : ``int``, optional (default: 0)
        How many extra pixels to include to each side the cropped grid for 
        extra visualisation.
    '''
    #get the appropriate coordinates, be they spatial or array, at appropriate mpp
    coords = get_mpp_coords(adata, basis=basis, spatial_key=spatial_key, mpp=mpp)
    #PIL crop is defined as a tuple of (left, upper, right, lower) coordinates
    #coords[:,0] is up-down, coords[:,1] is left-right
    #don't forget to add/remove buffer, and to not go past 0
    return (np.max([np.min(coords[:,1])-buffer, 0]), 
            np.max([np.min(coords[:,0])-buffer, 0]), 
            np.max(coords[:,1])+buffer, 
            np.max(coords[:,0])+buffer
           )

def scaled_he_image(adata, mpp=1, crop=True, buffer=150, 
                    spatial_cropped_key=None, store=True, 
                    img_key=None, save_path=None,
                    backend='tifffile'):
    '''
    Create a custom microns per pixel render of the full scale H&E image for 
    visualisation and downstream application. Store resulting image and its 
    corresponding size factor in the object. If cropping to just the spatial 
    grid, also store the cropped spatial coordinates. Optionally save to file.
    
    Input
    -----
    adata : ``AnnData``
        2um bin VisiumHD object. Path to high resolution H&E image provided via 
        ``source_image_path`` to ``b2c.read_visium()``.
    mpp : ``float``, optional (default: 1)
        Microns per pixel of the desired H&E image to create.
    crop : ``bool``, optional (default: ``True``)
        If ``True``, will limit the image to the actual spatial coordinate area, 
        with ``buffer`` added to each dimension.
    buffer : ``int``, optional (default: 150)
        Only used with ``crop=True``. How many extra pixels (in original 
        resolution) to include on each side of the captured spatial grid.
    spatial_cropped_key : ``str`` or ``None``, optional (default: ``None``)
        Only used with ``crop=True``. ``.obsm`` key to store the adjusted 
        spatial coordinates in. If ``None``, defaults to 
        ``"spatial_cropped_X_buffer"``, where ``X`` is the value of ``buffer``.
    store : ``bool``, optional (default: ``True``)
        Whether to store the generated image within the object.
    img_key : ``str`` or ``None``, optional (default: ``None``)
        Only used with ``store=True``. The image key to store the image 
        under in the object. If ``None``, defaults to ``"X_mpp_Y_buffer"``, 
        where ``X`` is the value of ``mpp`` and ``Y`` is the value of 
        ``buffer`` in the instance of ``crop=True``. If no cropping is to 
        be done, defaults to ``"X_mpp"``.
    save_path : ``filepath`` or ``None``, optional (default: ``None``)
        If specified, will save the generated image to this path (e.g. for 
        StarDist use).
    '''
    #identify name of spatial key for subsequent access of fields
    import cv2
    library = list(adata.uns['spatial'].keys())[0]
    #retrieve specified source image path and load it
    img = load_image(adata.uns['spatial'][library]['metadata']['source_image_path'],
                     backend=backend)
    #assess that the image dimensions match what they're supposed to be
    #if not, inform the user what image they should retrieve and use
    actual_vs_inferred_image_shape(adata, img)
    #crop image if necessary
    if crop:
        crop_coords = get_crop(adata, basis="spatial", spatial_key="spatial", mpp=None, buffer=buffer)
        #this is already capped at a minimum of 0, so can just subset freely
        #left, upper, right, lower; image is up-down, left-right
        img = img[crop_coords[1]:crop_coords[3], crop_coords[0]:crop_coords[2], :]
        #set up the spatial cropped key if one is not passed
        if spatial_cropped_key is None:
            spatial_cropped_key = "spatial_cropped_"+str(buffer)+"_buffer"
        #need to move spatial so it starts at the new crop top left point
        #spatial[:,1] is up-down, spatial[:,0] is left-right
        adata.obsm[spatial_cropped_key] = adata.obsm["spatial"].copy()
        adata.obsm[spatial_cropped_key][:,0] -= crop_coords[0]
        adata.obsm[spatial_cropped_key][:,1] -= crop_coords[1]
        #print off the spatial cropped key just in case
        print("Cropped spatial coordinates key: "+spatial_cropped_key)
    #reshape image to desired microns per pixel
    #get necessary scale factor for the custom mpp
    #multiply dimensions by this to get the shrunken image size
    #multiply .obsm['spatial'] by this to get coordinates matching the image
    scalef = mpp_to_scalef(adata, mpp=mpp)
    #need to reverse dimension order and turn to int for cv2
    dim = (np.array(img.shape[:2])*scalef).astype(int)[::-1]
    img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    #we have everything we need. store in object
    if store:
        if img_key is None:
            img_key = str(mpp)+"_mpp"
            if crop:
                img_key = img_key+"_"+str(buffer)+"_buffer"
        adata.uns['spatial'][library]['images'][img_key] = img
        #the scale factor needs to be prefaced with "tissue_"
        adata.uns['spatial'][library]['scalefactors']['tissue_'+img_key+"_scalef"] = scalef
        #print off the image key just in case
        print("Image key: "+img_key)
    if save_path is not None:
        #cv2 expects BGR channel order, we're working with RGB
        cv2.imwrite(save_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

def scaled_if_image(adata, channel, mpp=1, crop=True, buffer=150, spatial_cropped_key=None, store=True, img_key=None, save_path=None):
    '''
    Create a custom microns per pixel render of the full scale IF image for 
    visualisation and downstream application. Store resulting image and its 
    corresponding size factor in the object. If cropping to just the spatial 
    grid, also store the cropped spatial coordinates. Optionally save to file.
    
    Input
    -----
    adata : ``AnnData``
        2um bin VisiumHD object. Path to high resolution IF image provided via 
        ``source_image_path`` to ``b2c.read_visium()``.
    channel : ``int``
        The channel of the IF image holding the DAPI capture.
    mpp : ``float``, optional (default: 1)
        Microns per pixel of the desired IF image to create.
    crop : ``bool``, optional (default: ``True``)
        If ``True``, will limit the image to the actual spatial coordinate area, 
        with ``buffer`` added to each dimension.
    buffer : ``int``, optional (default: 150)
        Only used with ``crop=True``. How many extra pixels (in original 
        resolution) to include on each side of the captured spatial grid.
    spatial_cropped_key : ``str`` or ``None``, optional (default: ``None``)
        Only used with ``crop=True``. ``.obsm`` key to store the adjusted 
        spatial coordinates in. If ``None``, defaults to 
        ``"spatial_cropped_X_buffer"``, where ``X`` is the value of ``buffer``.
    store : ``bool``, optional (default: ``True``)
        Whether to store the generated image within the object.
    img_key : ``str`` or ``None``, optional (default: ``None``)
        Only used with ``store=True``. The image key to store the image 
        under in the object. If ``None``, defaults to ``"X_mpp_Y_buffer"``, 
        where ``X`` is the value of ``mpp`` and ``Y`` is the value of 
        ``buffer`` in the instance of ``crop=True``. If no cropping is to 
        be done, defaults to ``"X_mpp"``.
    save_path : ``filepath`` or ``None``, optional (default: ``None``)
        If specified, will save the generated image to this path (e.g. for 
        StarDist use).
    '''
    #identify name of spatial key for subsequent access of fields
    import cv2
    import tifffile as tf
    library = list(adata.uns['spatial'].keys())[0]
    #pull out specified channel from IF tiff via tifffile
    #pretype to float32 for space while working with plots (float16 does not)
    img = tf.imread(adata.uns['spatial'][library]['metadata']['source_image_path'], key=channel).astype(np.float32)
    #assess that the image dimensions match what they're supposed to be
    #if not, inform the user what image they should retrieve and use
    actual_vs_inferred_image_shape(adata, img)
    #this can be dark, apply stardist normalisation to fix
    img = normalize(img)
    #actually cap the values - currently there are sub 0 and above 1 entries
    img[img<0] = 0
    img[img>1] = 1
    #crop image if necessary
    if crop:
        crop_coords = get_crop(adata, basis="spatial", spatial_key="spatial", mpp=None, buffer=buffer)
        #this is already capped at a minimum of 0, so can just subset freely
        #left, upper, right, lower; image is up-down, left-right
        img = img[crop_coords[1]:crop_coords[3], crop_coords[0]:crop_coords[2]]
        #set up the spatial cropped key if one is not passed
        if spatial_cropped_key is None:
            spatial_cropped_key = "spatial_cropped_"+str(buffer)+"_buffer"
        #need to move spatial so it starts at the new crop top left point
        #spatial[:,1] is up-down, spatial[:,0] is left-right
        adata.obsm[spatial_cropped_key] = adata.obsm["spatial"].copy()
        adata.obsm[spatial_cropped_key][:,0] -= crop_coords[0]
        adata.obsm[spatial_cropped_key][:,1] -= crop_coords[1]
        #print off the spatial cropped key just in case
        print("Cropped spatial coordinates key: "+spatial_cropped_key)
    #reshape image to desired microns per pixel
    #get necessary scale factor for the custom mpp
    #multiply dimensions by this to get the shrunken image size
    #multiply .obsm['spatial'] by this to get coordinates matching the image
    scalef = mpp_to_scalef(adata, mpp=mpp)
    #need to reverse dimension order and turn to int for cv2
    dim = (np.array(img.shape[:2])*scalef).astype(int)[::-1]
    img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    #we have everything we need. store in object
    if store:
        if img_key is None:
            img_key = str(mpp)+"_mpp"
            if crop:
                img_key = img_key+"_"+str(buffer)+"_buffer"
        adata.uns['spatial'][library]['images'][img_key] = img
        #the scale factor needs to be prefaced with "tissue_"
        adata.uns['spatial'][library]['scalefactors']['tissue_'+img_key+"_scalef"] = scalef
        #print off the image key just in case
        print("Image key: "+img_key)
    if save_path is not None:
        #cv2 expects BGR channel order, we have a greyscale image
        #oh also we should make it a uint8 as otherwise stuff won't work
        cv2.imwrite(save_path, cv2.cvtColor((255*img).astype(np.uint8), cv2.COLOR_GRAY2BGR))

def insert_labels(adata, labels_npz_path, basis="spatial", spatial_key="spatial", mpp=None, labels_key="labels"):
    '''
    Load StarDist segmentation results and store them in the object. Labels 
    will be stored as integers, with 0 being unassigned to an object.
    
    Input
    -----
    adata : ``AnnData``
        2um bin VisiumHD object.
    labels_npz_path : ``filepath``
        Path to sparse labels generated by ``b2c.stardist()``.
    basis : ``str``, optional (default: ``"spatial"``)
        Whether the image represents ``"spatial"`` or ``"array"`` coordinates. 
        The former is the source morphology image, the latter is a GEX-based grid 
        representation.
    spatial_key : ``str``, optional (default: ``"spatial"``)
        Only used with ``basis="spatial"``. Needs to be present in ``.obsm``. 
        Rounded coordinates will be used to represent each bin when retrieving 
        labels.
    mpp : ``float`` or ``None``, optional (default: ``None``)
        The mpp value that was used to generate the segmented image. Mandatory 
        for GEX (``basis="array"``), if not provided with morphology 
        (``basis="spatial"``) will assume full scale image.
    labels_key : ``str``, optional (default: ``"labels"``)
        ``.obs`` key to store the labels under.
    '''
    #load sparse segmentation results
    labels_sparse = scipy.sparse.load_npz(labels_npz_path)
    #may as well stash that path in .uns['bin2cell'] since we have it
    if "bin2cell" not in adata.uns:
        adata.uns["bin2cell"] = {}
    if "labels_npz_paths" not in adata.uns["bin2cell"]:
        adata.uns["bin2cell"]["labels_npz_paths"] = {}
    #store as absolute path if it's relative
    if labels_npz_path[0] != "/":
        npz_prefix = os.getcwd()+"/"
    else:
        npz_prefix = ""
    adata.uns["bin2cell"]["labels_npz_paths"][labels_key] = npz_prefix + labels_npz_path
    #get the appropriate coordinates, be they spatial or array, at appropriate mpp
    coords = get_mpp_coords(adata, basis=basis, spatial_key=spatial_key, mpp=mpp)
    #there is a possibility that some coordinates will fall outside labels_sparse
    #start by pregenerating an obs column of all zeroes so all bins are covered
    adata.obs[labels_key] = 0
    #can now construct a mask defining which coordinates fall within range
    #apply the mask to the coords and the obs to just go for the relevant bins
    mask = ((coords[:,0] >= 0) & 
            (coords[:,0] < labels_sparse.shape[0]) & 
            (coords[:,1] >= 0) & 
            (coords[:,1] < labels_sparse.shape[1])
           )
    #pull out the cell labels for the coordinates, can just index the sparse matrix with them
    #insert into bin object, need to turn it into a 1d numpy array from a 1d numpy matrix first
    adata.obs.loc[mask, labels_key] = np.asarray(labels_sparse[coords[mask,0], coords[mask,1]]).flatten()

def expand_labels(adata, labels_key="labels", expanded_labels_key="labels_expanded", algorithm="max_bin_distance", max_bin_distance=2, volume_ratio=4, k=4, subset_pca=True):
    '''
    Expand StarDist segmentation results to bins a maximum distance away in 
    the array coordinates. In the event of multiple equidistant bins with 
    different labels, ties are broken by choosing the closest bin in a PCA 
    representation of gene expression. The resulting labels will be integers, 
    with 0 being unassigned to an object.
    
    Input
    -----
    adata : ``AnnData``
        2um bin VisiumHD object. Raw or destriped counts.
    labels_key : ``str``, optional (default: ``"labels"``)
        ``.obs`` key holding the labels to be expanded. Integers, with 0 being 
        unassigned to an object.
    expanded_labels_key : ``str``, optional (default: ``"labels_expanded"``)
        ``.obs`` key to store the expanded labels under.
    algorithm : ``str``, optional (default: ``"max_bin_distance"``)
        Toggle between ``max_bin_distance`` or ``volume_ratio`` based label 
        expansion.
    max_bin_distance : ``int`` or ``None``, optional (default: 2)
        Maximum number of bins to expand the nuclear labels by.
    volume_ratio : ``float``, optional (default: 4)
        A per-label expansion distance will be proposed as 
        ``ceil((volume_ratio**(1/3)-1) * sqrt(n_bins/pi))``, where 
        ``n_bins`` is the number of bins for the corresponding pre-expansion 
        label. Default based on cell line 
        `data <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8893647/>`_
    k : ``int``, optional (default: 4)
        Number of assigned spatial coordinate bins to find as potential nearest 
        neighbours for each unassigned bin.
    subset_pca : ``bool``, optional (default: ``True``)
        If ``True``, will obtain the PCA representation of just the bins 
        involved in the tie breaks rather than the full bin space. Results in 
        a slightly different embedding at a lower resource footprint.
    '''
    #this is where the labels will go
    adata.obs[expanded_labels_key] = adata.obs[labels_key].values.copy()
    #get out our array grid, and preexisting labels
    coords = adata.obs[["array_row","array_col"]].values
    labels = adata.obs[labels_key].values
    #we'll be splitting the space in two - the bins with labels, and those without
    object_mask = (labels != 0)
    #get their indices in cell space
    full_reference_inds = np.arange(adata.shape[0])[object_mask]
    full_query_inds = np.arange(adata.shape[0])[~object_mask]
    #for each unassigned bin, we'll find its k nearest neighbours in the assigned space
    #build a reference using the assigned bins' coordinates
    ckd = scipy.spatial.cKDTree(coords[object_mask, :])
    #query it using the unassigned bins' coordinates
    dists, hits = ckd.query(x=coords[~object_mask,:], k=k, workers=-1)
    #convert the identified indices back to the full cell space
    hits = full_reference_inds[hits]
    #get the label calls for each of the hits
    calls = labels[hits]
    #get the area (bin count) of each object
    label_values, label_counts = np.unique(labels, return_counts=True)
    #this is how the algorithm was toggled early on
    #switched to an argument to avoid potential future spaghetti
    if max_bin_distance is None:
        raise ValueError("Use ``algorithm`` to toggle between algorithms")
    if algorithm == "volume_ratio":
        #compute the object's sphere's radius as sqrt(nbin/pi)
        #scale to radius of cell by multiplying by volume_ratio^(1/3)
        #and subtract away the original radius to account for presence of nucleus
        #do a ceiling to compensate for possible reduction of area in slice
        label_distances = np.ceil((volume_ratio**(1/3)-1) * np.sqrt(label_counts/np.pi))
        #get an array where you can index on object and get the distance
        #needs +1 as the max value of label_values is actually present in the data
        label_distance_array = np.zeros((np.max(label_values)+1,))
        label_distance_array[label_values] = label_distances
    elif algorithm == "max_bin_distance":
        #just use the provided value
        label_distance_array = np.ones((np.max(label_values)+1,)) * max_bin_distance
    else:
        raise ValueError("``algorithm`` must be ``'max_bin_distance'`` or ``'volume_ratio'``")
    #construct a matching dimensionality array of max distance allowed per call
    max_call_distance = label_distance_array[calls]
    #mask bins too far away from call with arbitrary high value
    dist_mask = 1000
    dists[dists > max_call_distance] = dist_mask
    #evaluate the minima in each row. start by getting said minima
    min_per_bin = np.min(dists, axis=1)[:,None]
    #now get positions in each row that have the minimum (and aren't the mask)
    is_hit = (dists == min_per_bin) & (min_per_bin < dist_mask)
    #case one - we have a solitary hit of the minimum
    clear_mask = (np.sum(is_hit, axis=1) == 1)
    #get out the indices of the bins
    clear_query_inds = full_query_inds[clear_mask]
    #np.argmin(axis=1) finds the column of the minimum per row
    #subsequently retrieve the matching hit from calls
    clear_query_labels = calls[clear_mask, np.argmin(dists[clear_mask, :], axis=1)]
    #insert calls into object
    adata.obs.loc[adata.obs_names[clear_query_inds], expanded_labels_key] = clear_query_labels
    #case two - 2+ assigned bins are equidistant
    ambiguous_mask = (np.sum(is_hit, axis=1) > 1)
    if np.sum(ambiguous_mask) > 0:
        #get their indices in the original cell space
        ambiguous_query_inds = full_query_inds[ambiguous_mask]
        if subset_pca:
            #in preparation of PCA, get a master list of all the bins to PCA
            #we've got two sets - the query bins, and their k hits
            #the hits needs to be .flatten()ed after masking to become 1d again
            #np.unique sorts in an ascending fashion, which is convenient
            smol = np.unique(np.concatenate([hits[ambiguous_mask,:].flatten(), ambiguous_query_inds]))
            #prepare a PCA as a representation of the GEX space for solving ties
            #can just run straight on an array to get a PCA matrix back. convenient!
            #keep the object's X raw for subsequent cell creation
            pca_smol = sc.pp.pca(np.log1p(adata.X[smol, :]))
            #mock up a "full-scale" PCA matrix to not have to worry about different indices
            pca = np.zeros((adata.shape[0], pca_smol.shape[1]))
            pca[smol, :] = pca_smol
        else:
            #just run a full space PCA
            pca = sc.pp.pca(np.log1p(adata.X))
        #compute the distances between the expression profiles of the undecided bin and the neighbours
        #np.linalg.norm is the fastest way to get euclidean, subtract two point sets beforehand
        #pca[hits[ambiguous_mask, :]] is bins by k by num_pcs
        #pca[ambiguous_query_inds, :] is bins by num_pcs
        #add the [:, None, :] and it's bins by 1 by num_pcs, and subtracts as you'd hope
        eucl_input = pca[hits[ambiguous_mask, :]] - pca[ambiguous_query_inds, :][:, None, :]
        #can just do this along axis=2 and get all the distances at once
        eucl_dists = np.linalg.norm(eucl_input, axis=2)
        #mask ineligible bins with arbitrary high value
        eucl_mask = 1000
        eucl_dists[~is_hit[ambiguous_mask, :]] = eucl_mask
        #define calls based on euclidean minimum
        #same argmin/mask logic as with clear before
        ambiguous_query_labels = calls[ambiguous_mask, np.argmin(eucl_dists, axis=1)]
        #insert calls into object
        adata.obs.loc[adata.obs_names[ambiguous_query_inds], expanded_labels_key] = ambiguous_query_labels

def salvage_secondary_labels(adata, primary_label="labels_he_expanded", secondary_label="labels_gex", labels_key="labels_joint"):
    '''
    Create a joint ``labels_key`` that takes the ``primary_label`` and fills in 
    unassigned bins based on calls from ``secondary_label``. Only objects that do not 
    overlap with any bins called as part of ``primary_label`` are transferred over.
    
    Input
    -----
    adata : ``AnnData``
        2um bin VisiumHD object. Needs ``primary_key`` and ``secodary_key`` in ``.obs``.
    primary_label : ``str``, optional (default: ``"labels_he_expanded"``)
        ``.obs`` key holding the main labels. Integers, with 0 being unassigned to an 
        object.
    secondary_label : ``str``, optional (default: ``"labels_gex"``)
        ``.obs`` key holding the labels to be inserted into unassigned bins. Integers, 
        with 0 being unassigned to an object.
    labels_key : ``str``, optional (default: ``"labels_joint"``)
        ``.obs`` key to store the combined label information into. Will also add a 
        second column with ``"_source"`` appended to differentiate whether the bin was 
        tagged from the primary or secondary label.
    '''
    #these are the bins that have the primary label assigned
    primary = adata.obs.loc[adata.obs[primary_label] > 0, :]
    #these are the bins that lack the primary label, but have the secondary label
    secondary = adata.obs.loc[adata.obs[primary_label] == 0, :]
    secondary = secondary.loc[secondary[secondary_label] > 0, :]
    #kick out any secondary labels that appear in primary-labelled bins
    #we are just interested in ones that are unique to bins without primary labelling
    secondary_to_take = np.array(list(set(secondary[secondary_label]).difference(set(primary[secondary_label]))))
    #both of these labels are integers, starting from 1
    #offset the new secondary labels by however much the maximum primary label is
    offset = np.max(adata.obs[primary_label])
    #use the primary labels as a basis
    adata.obs[labels_key] = adata.obs[primary_label].copy()
    #flag any bins that are assigned to our secondary labels of interest
    mask = np.isin(adata.obs[secondary_label], secondary_to_take)
    adata.obs.loc[mask, labels_key] = adata.obs.loc[mask, secondary_label] + offset
    #store information on origin of call
    adata.obs[labels_key+"_source"] = "none"
    adata.obs.loc[adata.obs[primary_label]>0, labels_key+"_source"] = "primary"
    adata.obs.loc[mask, labels_key+"_source"] = "secondary"
    #stash secondary label offset as that seems potentially useful
    if "bin2cell" not in adata.uns:
        adata.uns["bin2cell"] = {}
    if "secondary_label_offset" not in adata.uns["bin2cell"]:
        adata.uns["bin2cell"]["secondary_label_offset"] = {}
    adata.uns["bin2cell"]["secondary_label_offset"][labels_key] = offset
    #notify of how much was salvaged
    print("Salvaged "+str(len(secondary_to_take))+" secondary labels")

def bin_to_cell(adata, labels_key="labels_expanded", spatial_keys=["spatial"], diameter_scale_factor=None):
    '''
    Collapse all bins for a given nonzero ``labels_key`` into a single cell. 
    Gene expression added up, array coordinates and ``spatial_keys`` averaged out. 
    ``"spot_diameter_fullres"`` in the scale factors multiplied by 
    ``diameter_scale_factor`` to reflect increased unit size. Returns cell level AnnData, 
    including ``.obs["bin_count"]`` reporting how many bins went into creating the cell.
    
    Input
    -----
    adata : ``AnnData``
        2um bin VisiumHD object. Raw or destriped counts. Needs ``labels_key`` in ``.obs`` 
        and ``spatial_keys`` in ``.obsm``.
    labels_key : ``str``, optional (default: ``"labels_expanded"``)
        Which ``.obs`` key to use for grouping 2um bins into cells. Integers, with 0 being 
        unassigned to an object. If an extra ``"_source"`` column is detected as a result 
        of ``b2c.salvage_secondary_labels()`` calling, its info will be propagated per 
        label.
    spatial_keys : list of ``str``, optional (default: ``["spatial"]``)
        Which ``.obsm`` keys to average out across all bins falling into a cell to get a 
        cell's respective spatial coordinates.
    diameter_scale_factor : ``float`` or ``None``, optional (default: ``None``)
        The object's ``"spot_diameter_fullres"`` will be multiplied by this much to reflect 
        the change in unit per observation. If ``None``, will default to the square root of 
        the mean of the per-cell bin counts.
    '''
    #a label of 0 means there's nothing there, ditch those bins from this operation
    adata = adata[adata.obs[labels_key]!=0]
    #use the newly inserted labels to make pandas dummies, as sparse because the data is huge
    cell_to_bin = pd.get_dummies(adata.obs[labels_key], sparse=True)
    #take a quick detour to save the cell labels as they appear in the dummies
    #they're likely to be integers, make them strings to avoid complications in the downstream AnnData
    cell_names = [str(i) for i in cell_to_bin.columns]
    #then pull out the actual internal sparse matrix (.sparse) as a scipy COO one, turn to CSR
    #this has bins as rows, transpose so cells are as rows (and CSR becomes CSC for .dot())
    cell_to_bin = cell_to_bin.sparse.to_coo().tocsr().T
    #can now generate the cell expression matrix by adding up the bins (via matrix multiplication)
    #cell-bin * bin-gene = cell-gene
    #(turn it to CSR at the end as somehow it comes out CSC)
    X = cell_to_bin.dot(adata.X).tocsr()
    #create object, stash stuff
    cell_adata = ad.AnnData(X, var = adata.var)
    cell_adata.obs_names = cell_names
    #turn the cell names back to int and stash that as metadata too
    cell_adata.obs['object_id'] = [int(i) for i in cell_names]
    #need to bust out deepcopy here as otherwise altering the spot diameter gets back-propagated
    cell_adata.uns['spatial'] = deepcopy(adata.uns['spatial'])
    #getting the centroids (means of bin coords) involves computing a mean of each cell_to_bin row
    #premultiplying by a diagonal matrix multiplies each row by a value: https://solitaryroad.com/c108.html
    #use that to divide each row by it sum (.sum(axis=1)), then matrix multiply the result by bin coords
    #stash the sum into a separate variable for subsequent object storage
    #cell-cell * cell-bin * bin-coord = cell-coord
    bin_count = np.asarray(cell_to_bin.sum(axis=1)).flatten()
    row_means = scipy.sparse.diags(1/bin_count)
    cell_adata.obs['bin_count'] = bin_count
    #take the thing out for a spin with array coordinates
    cell_adata.obs["array_row"] = row_means.dot(cell_to_bin).dot(adata.obs["array_row"].values)
    cell_adata.obs["array_col"] = row_means.dot(cell_to_bin).dot(adata.obs["array_col"].values)
    #generate the various spatial coordinate systems
    #just in case a single is passed as a string
    if type(spatial_keys) is not list:
        spatial_keys = [spatial_keys]
    for spatial_key in spatial_keys:
        cell_adata.obsm[spatial_key] = row_means.dot(cell_to_bin).dot(adata.obsm[spatial_key])
    #of note, the default scale factor bin diameter at 2um resolution stops rendering sensibly in plots
    #by default estimate it as the sqrt of the bin count mean
    if diameter_scale_factor is None:
        diameter_scale_factor = np.sqrt(np.mean(bin_count))
    #bump it up to something a bit more sensible
    library = list(adata.uns['spatial'].keys())[0]
    cell_adata.uns['spatial'][library]['scalefactors']['spot_diameter_fullres'] *= diameter_scale_factor
    #if we can find a source column, transfer that
    if labels_key+"_source" in adata.obs.columns:
        #hell of a one liner. the premise is to turn two columns of obs into a translation dictionary
        #so pull them out, keep unique rows, turn everything to string (as labels are strings in cells)
        #then set the index to be the label names, turn the thing to dict
        #pd.DataFrame -> dict makes one entry per column (even if we just have the one column here)
        #so pull out our column's entry and we have what we're after
        mapping = adata.obs[[labels_key,labels_key+"_source"]].drop_duplicates().astype(str).set_index(labels_key).to_dict()[labels_key+"_source"]
        #translate the labels from the cell object
        cell_adata.obs[labels_key+"_source"] = [mapping[i] for i in cell_adata.obs_names]
    return cell_adata