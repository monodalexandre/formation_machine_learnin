# Matrice de corrélation avec plusieurs méthodes

def correlations(data, methods):
    """
    Plot correlation matrix for quantitatives values.
        
        Args:
        data(dataframe): A dataframe
        
        method(string): CHosen method to compute correlation. Might be "pearson", "spearman" or "kendall"
        
    Returns:
        Correlation matrix
    """
    
    correlations = []
    cleanning_masks = []
    for method in methods: 
        correlation = data.select_dtypes(include=['int64','float64']).corr(method=method) * 100
        cleanning_mask = np.zeros_like(correlation)
        upper_triangle = np.triu_indices_from(correlation)
        cleanning_mask[upper_triangle] = 1
        correlations.append(correlation)
        cleanning_masks.append(cleanning_mask)

        
    fig, axes = plt.subplots(nrows=len(methods), figsize=(30,40))
    fig.subplots_adjust(wspace=0.2)

    for i in range(len(axes)):
        sns.heatmap(correlations[i], ax=axes[i], cmap="RdBu_r", mask = cleanning_masks[i], 
                   annot = True, fmt=".0f", cbar=False)

        axes[i].set_title(f"Matrice de corrélations de {methods[i]} en %")
  
def correlations(data, method):

    correlation = data.select_dtypes(include=['int64','float64']).corr(method=method) * 100
    cleaning_mask = np.zeros_like(correlation)
    upper_triangle = np.triu_indices_from(correlation)
    cleaning_mask[upper_triangle] = 1

    fig, axes = plt.subplots(nrows=1, figsize=(15,12))

    sns.heatmap(correlation, cmap="RdBu_r", mask = cleaning_mask, 
                annot = True, fmt=".0f", cbar=False)

    axes.set_title(f"Matrice de corrélations de {method} en %")