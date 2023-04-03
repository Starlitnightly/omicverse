import numpy as np
from matplotlib.colors import LinearSegmentedColormap
sc_color=['#7CBB5F','#368650','#A499CC','#5E4D9A','#78C2ED','#866017', '#9F987F','#E0DFED',
 '#EF7B77', '#279AD7','#F0EEF0', '#1F577B', '#A56BA7', '#E0A7C8', '#E069A6', '#941456', '#FCBC10',
 '#EAEFC5', '#01A0A7', '#75C8CC', '#F0D7BC', '#D5B26C', '#D5DA48', '#B6B812', '#9DC3C3', '#A89C92', '#FEE00C', '#FEF2A1']
sc_color_cmap = LinearSegmentedColormap.from_list('Custom', sc_color, len(sc_color))

def pyomic_palette():
    """
    Returns a dictionary of colors for various plots used in pyomic package.

    Returns
    -------
    - sc_color: `list`
        List containing the hex codes as values.
    """ 
    return sc_color

def plot_text_set(text,text_knock=2,text_maxsize=20):
    """
    Formats the text to fit in a plot by adding line breaks.

    Parameters
    ----------
    - text : `str`
        Text to format.
    - text_knock : `int`, optional
        Number of words to skip between two line breaks, by default 2.
    - text_maxsize : `int`, optional
        Maximum length of the text before formatting, by default 20.

    Returns
    -------
    - text: `str`
        Formatted text.
    """
    #print(text)
    text_len=len(text)
    if text_len>text_maxsize:
        ty=text.split(' ')
        ty_len=len(ty)
        if ty_len%2==1:
            ty_mid=(ty_len//text_knock)+1
        else:
            ty_mid=(ty_len//text_knock)
        #print(ty_mid)

        if ty_mid==0:
            ty_mid=1

        res=''
        ty_len_max=np.max([i%ty_mid for i in range(ty_len)])
        if ty_len_max==0:
            ty_len_max=1
        for i in range(ty_len):
            #print(ty_mid,i%ty_mid,i,ty_len_max)
            if (i%ty_mid)!=ty_len_max:
                res+=ty[i]+' '
            else:
                res+='\n'+ty[i]+' '
        return res
    else:
        return text