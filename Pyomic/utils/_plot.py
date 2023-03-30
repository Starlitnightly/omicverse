import numpy as np

def plot_text_set(text,text_knock=2,text_maxsize=20):
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