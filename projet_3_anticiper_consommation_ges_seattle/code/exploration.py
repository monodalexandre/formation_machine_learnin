import pandas as pd
from tabulate import tabulate
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from math import radians, cos, sin, asin, sqrt


import squarify
import matplotlib.colors

def analyse_forme(df,thresh_na=60, all=False):
    
    statList = {'Taille':[df.size],'Nb lignes':[df.shape[0]],
                'Nb colonnes':[df.shape[1]],
                '% de NaN':[round(100.0 * (df.isna().sum().sum())/df.size,2)],
                'Nb duplicats':df.duplicated().sum()}
    statsValues = pd.DataFrame().from_dict(statList, orient='columns')
    print(tabulate(statsValues, headers = 'keys', tablefmt = 'psql'))    

    if(all):
        # print(df.head(10))
    
        plt.figure(figsize=(16,8))

        ax1 = plt.subplot(1,2,1)
        vc = df.dtypes.value_counts()

        # create a color palette, mapped to these values
        cmap = matplotlib.cm.Blues
        mini=min(vc)
        maxi=max(vc)
        norm = matplotlib.colors.Normalize(vmin=mini, vmax=maxi)
        colors = [cmap(norm(value)) for value in vc]

        squarify.plot(ax = ax1, sizes=list(vc.values), label=pd.Series(vc.keys()).to_string(index=False).replace(" ", "").splitlines(), alpha=.8, value=round(vc/vc.sum(),2), pad=0.05, color=colors,  text_kwargs={'fontsize': 22, 'fontfamily' : 'sans-serif'})
        plt.axis('off')
        ax1.set_title('Répartition valeurs quantitatives/qualitatives',fontsize=20,weight='bold')

        ax2 = plt.subplot(1,2,2)
        perc = (df.isnull().sum()/df.shape[0])*100
        perc = perc.sort_values(ascending=False)
        perc.index = np.arange(0,df.shape[1],1)

        ax2 = sns.barplot(x=perc.index,y=perc, palette=sns.dark_palette("#69d", reverse=True))
        plt.axhline(y=thresh_na, color='r', linestyle='-')
        plt.text(len(df.isnull().sum()/len(df))/1.7, thresh_na+12.5, 'Columns with more than %s%s missing values' %(thresh_na, '%'), fontsize=12,weight='bold', color='crimson',
             ha='left' ,va='top')
        plt.text(len(df.isnull().sum()/len(df))/1.7, thresh_na - 5, 'Columns with less than %s%s missing values' %(thresh_na, '%'), fontsize=12,weight='bold', color='blue',
             ha='left' ,va='top')

        ax2.set_title('NaN par colonnes',fontsize=20, weight='bold')
        ax2.set_xlabel('Colonnes',fontsize=20)
        ax2.set_ylabel('% de NaN',fontsize=20)
        ax2.set_xticks(np.arange(0,df.shape[1],5))
        ax2.set_yticks(np.arange(0,101,5))

        plt.show()

        

def premier_quartile(data_frame,colonne):
    """Pour une variables quantitative. Retourne la valeur du premier quartile. Colonne est le nom de la colonne."""
    return data_frame[colonne].quantile(q=0.25)


def troisieme_quartile(data_frame,colonne):
    """Pour une variables quantitative. Retourne la valeur du troisième quartile. Colonne est le nom de la colonne"""
    return data_frame[colonne].quantile(q=0.75)


def inter_quartile(data_frame,colonne):
    """Retourne l'écart inter-quartile."""
    return troisieme_quartile(data_frame,colonne)-premier_quartile(data_frame,colonne)

def correlations(data, method):

    correlation = data.select_dtypes(include=['int64','float64']).corr(method=method) * 100
    cleaning_mask = np.zeros_like(correlation)
    upper_triangle = np.triu_indices_from(correlation)
    cleaning_mask[upper_triangle] = 1

    fig, axes = plt.subplots(nrows=1, figsize=(15,12))

    sns.heatmap(correlation, cmap="RdBu_r", mask = cleaning_mask, 
                annot = True, fmt=".0f", cbar=False)

    axes.set_title(f"Matrice de corrélations de {method} en %")
  

def correlations_small(data, method):

    correlation = data.select_dtypes(include=['int64','float64']).corr(method=method) * 100
    cleaning_mask = np.zeros_like(correlation)
    upper_triangle = np.triu_indices_from(correlation)
    cleaning_mask[upper_triangle] = 1

    fig, axes = plt.subplots(nrows=1, figsize=(3,2))

    sns.heatmap(correlation, cmap="RdBu_r", mask = cleaning_mask, 
                annot = True, fmt=".0f", cbar=False)

    axes.set_title(f"Matrice de corrélations de {method} en %")

def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance in kilometers between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))

    r = 6371 # Radius of earth in kilometers. Use 3956 for miles. Determines return value units.
    return c * r
    
    
    
    
