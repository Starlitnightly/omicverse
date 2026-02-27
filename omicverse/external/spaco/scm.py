from typing import List

import matplotlib


def pal(palette_name:str, n:int) -> List[str]:
    """
    API to retrieve palettes hex code from Scientific Colour Map (SCM) by Crameri et al.'s work, or dittoSeq colors.

    Reference:
    Crameri, F. (2018). Scientific colour maps. Zenodo. http://doi.org/10.5281/zenodo.1243862

    Bunis, Daniel G., et al. 2021. "dittoSeq: Universal User-Friendly Single-Cell and Bulk RNA
    Sequencing Visualization Toolkit." Bioinformatics 36 (22-23): 5535-36.

    Args:
        palette_name (str): name of a palette in SCM or dittoSeq.
        n_colors (int): number of colors to retrive from the SCM palette.

    Returns:
        List[str]: a list of hex values for the retrived palette.
    """
    
    if palette_name == "ditto":
        ditto_pal = [
            "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2",
            "#D55E00", "#CC79A7", "#666666", "#AD7700", "#1C91D4",
            "#007756", "#D5C711", "#005685", "#A04700", "#B14380",
            "#4D4D4D", "#FFBE2D", "#80C7EF", "#00F6B3", "#F4EB71",
            "#06A5FF", "#FF8320", "#D99BBD", "#8C8C8C"
        ]
        return ditto_pal[:n]

    try:
        from cmcrameri import cm
    except ImportError:
        print("Please install the brilliant cmcrameri package to use Scientific colour maps.")

    return [matplotlib.colors.to_hex(i) for i in eval("cm."+palette_name+".colors")][:n]