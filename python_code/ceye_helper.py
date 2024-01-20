
import pandas as pd
import numpy as np
import seaborn as sns
import math
import warnings
import plotly.express as px
import plotly.graph_objects as go
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA


### Helper Function Estimating Size Factors Needed for Normalization ###
def estimate_size_factors(gene_expression, sample_names):
    '''
    Normalizes the raw gene expression counts for the sample list provided using the calculated size factors
    based on the DESeq2 median-of-ratios normalization method

    Parameters
    ----------
    gene_expression : Pandas DataFrame 
        A Pandas DataFrame containing raw gene expression values (float)
    sample_names : list
        A 1D list of sample names (str) to be used for normalization

    Returns
    -------
    normalized_counts
        A Pandas DataFrame containing the normalized read counts
    '''

    print("Estimating size factors for normalization...")
    counts = gene_expression[sample_names]
    np.seterr(divide = 'ignore') 
    counts_log = np.log(counts)
    np.seterr(divide = 'warn') 
    loggeomeans = counts_log.mean(axis = 1)

    size_factors = []
    for column in counts:
        cnts = counts[column]
        log_cnts = counts_log[column]
        keep_index = (~(loggeomeans.isin([-np.inf]))) & (cnts > 0)
        sf = math.exp((log_cnts - loggeomeans)[keep_index].median())
        size_factors.append(sf)

    size_factors = pd.DataFrame(size_factors, index = sample_names, columns = ["size_factor"]).T
    print("Done!")
    return size_factors


### Normalization Function for Making Read Counts Comparable ###
def normalize_counts(gene_expression, sample_names):
    '''
    Normalizes the raw gene expression counts for the sample list provided using the calculated size factors
    based on the DESeq2 median-of-ratios normalization method

    Parameters
    ----------
    gene_expression : Pandas DataFrame 
        A Pandas DataFrame containing raw gene expression values (float)
    sample_names : list
        A 1D list of sample names (str) to be used for normalization

    Returns
    -------
    normalized_counts
        A Pandas DataFrame containing the normalized read counts
    '''
    print("Normalizing raw read counts...")
    size_factors = estimate_size_factors(gene_expression, sample_names)
    normalized_counts = gene_expression[sample_names].div(size_factors.iloc[0], axis = "columns")
    normalized_counts = pd.concat([gene_expression[["gene_name", "gene_biotype"]], normalized_counts], axis = 1)
    print("Done!")
    return normalized_counts


### PCA Plots for Sample Similarity ###
def plot_pca(normalized_counts, sample_names, top_n, sample_condition):
    '''
    Plots a Principal Component Analysis (PCA) plot using the top_n genes in terms of gene expression variation
    for the sample_names samples considered and colored using the sample_condition list

    Parameters
    ----------
    normalized_counts : Pandas DataFrame 
        A Pandas DataFrame containing normalized gene expression values (float)
    sample_names : list
        A 1D list of sample names (str) to be used for PCA plot 
    top_n : int
        The number of top N genes to be used to plot the PCA plot based on gene expression variation
    sample_condition : list
        A 1D list containing the conditions for each sample which will be used to color each sample in the PCA accordingly

    Returns
    -------
    fig
        A Plotly scatter plot (PCA)
    '''

    print("Plotting PCA plot...")
    df = normalized_counts[sample_names]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df_gid = list(normalized_counts.var(axis = 1).sort_values(ascending = False).index[:top_n])
    df_plot = df.loc[df_gid].T

    pca = PCA(n_components = 2)
    components = pca.fit_transform(df_plot)
    pca_var = pca.explained_variance_ratio_ * 100
    total_var = pca_var.sum()

    fig = px.scatter(components, x = 0, y = 1, color = sample_condition,
                        title = f'PCA Plot - Total Explained Variance: {total_var:.2f}%',
                        labels = {'0': f'PC 1 - Var {pca_var[0]:.2f}%', '1': f'PC 2 - Var {pca_var[1]:.2f}%'})
    print("Done!")
    return fig


### Venn Diagram for Set Comparison ###
def plot_venn_diagram(set1, set2, set1_name, set2_name, title):
    '''
    Plots a Venn diagram using 2 lists of values (i.e. set1 and set2), 
    determines the shared and unique element groups

    Parameters
    ----------
    set1 : list
        A 1D list containing the members of the first set
    set2 : list
        A 1D list containing the members of the second set
    set1_name : str
        The x-axis label for the scatter plot
    set2_name : str
        The y-axis label for the scatter plot
    title : str
        The title for the Venn diagram

    Returns
    -------
    fig
        A Plotly Venn diagram
    '''

    print("Plotting Venn diagram: " + title + "...")
    set_intersect = [value for value in set1 if value in set2]
    set1_only_size = len(set1) - len(set_intersect)
    set2_only_size = len(set2) - len(set_intersect)
    set_intersect_size = len(set_intersect)

    fig = go.Figure()

    # Create scatter trace of text labels
    fig.add_trace(go.Scatter(
        x = [1, 1.75, 2.5, 1, 2.5],
        y = [1, 1, 1, 2.25, 2.25],
        text = [str(set1_only_size), 
                str(set_intersect_size), 
                str(set2_only_size),
                set1_name,
                set2_name],
        mode = "text",
        textfont = dict(
            color = "black",
            size = 18,
            family = "Arail",
        )
    ))

    # Update axes properties
    fig.update_xaxes(
        showticklabels = False,
        showgrid = False,
        zeroline = False,
    )

    fig.update_yaxes(
        showticklabels = False,
        showgrid = False,
        zeroline = False,
    )

    # Add circles
    fig.add_shape(type = "circle",
        line_color = "blue", fillcolor = "blue",
        x0 = 0, y0 = 0, x1 = 2, y1 = 2
    )
    fig.add_shape(type = "circle",
        line_color = "gray", fillcolor = "gray",
        x0 = 1.5, y0 = 0, x1 = 3.5, y1 = 2
    )
    fig.update_shapes(opacity = 0.3, xref = "x", yref = "y")

    fig.update_layout(
        title = title,
        margin = dict(l = 20, r = 20, b = 100),
        height = 600, width = 800,
        plot_bgcolor = "white"
    )
    print("Done!")
    return fig


### Boxplots for Visualizing the Expression of a Gene Across Conditions ###
def plot_gene_expression_boxplot(normalized_counts, gene_id, sample_names, sample_condition, condition_name):
    '''
    Plots a gene expression boxplot showing the normalized read counts for a single gene for the specified sample set 
    and sample groups

    Parameters
    ----------
    normalized_counts : Pandas DataFrame 
        A Pandas DataFrame containing normalized gene expression values (float)
    gene_id : str
        The gene id for the gene whose expression will be plotted. This id should be present in the normalized_counts DataFrame
    sample_names : list
        A 1D list of sample names (str) to be used for gene expression boxplot 
    sample_condition : str
        A 1D list containing the conditions for each sample which will be used to determine the sample groups for boxplots
    condition_name : str
        The number of top N genes to be used to plot the PCA plot based on gene expression variation

    Returns
    -------
    fig
        A Plotly boxplot
    '''
    
    print("Plotting gene expression boxplot for gene: " + gene_id + "...")
    df = np.log(normalized_counts[sample_names].T + 1)
    df[condition_name] = sample_condition
    df = df.loc[:,[gene_id, condition_name]]

    gene_name = normalized_counts.loc[gene_id, ]["gene_name"]

    fig = px.box(df, x = condition_name, y = gene_id, points = "all", color = condition_name,
                    title = "Gene Expression Boxplot - " + gene_id + " - " + gene_name,
                    labels = {condition_name : condition_name, gene_id : 'Log2 Normalized Gene Expression'})
    print("Done!")
    return fig


### Volcano Plots for Visualizing the Significance and Magnitude of Gene Expression Change ###
def plot_volcano(de_result, pvalue_threshold, lfc_threshold, title):
    '''
    Plots a volcano plot providing an overview of the differentially expressed genes in terms of statistical 
    significance and magnitude of change based on the provided p-value and log2 fold change thresholds used

    Parameters
    ----------
    de_result : Pandas DataFrame 
        A Pandas DataFrame containing the differential expression results for the specific comparison
    pvalue_threshold : float
        The p-value threshold that will be used to determine statistically significant changes in gene expression
    lfc_threshold : float
        The log2 fold change threshold that will be used to determine the genes with strong changes in gene expression between conditions
    title : str
        The title of the volcano plot

    Returns
    -------
    fig
        A Plotly scatter plot (volcano)
    '''

    print("Plotting volcano plot: " + title + "...")
    p_test = de_result["padj"] < pvalue_threshold
    fc_up_test = de_result["log2FoldChange"] > lfc_threshold
    fc_down_test = de_result["log2FoldChange"] < -lfc_threshold
    
    status = ["Not DEG" for x in range(0, de_result.shape[0])] 
    for i in range(0, de_result.shape[0]):
        if p_test[i] and fc_up_test[i]:
            status[i] = "Up Reg"
        if p_test[i] and fc_down_test[i]:
            status[i] = "Down Reg"

    de_result["DE Status"] = status
    de_result["neg_log10_pval"] = -np.log10(de_result["pvalue"])

    fig = px.scatter(de_result, x = "log2FoldChange", y = "neg_log10_pval", 
                        color = "DE Status", 
                        hover_data = ['baseMean', 'log2FoldChange', 'pvalue', 'padj', 'gene_name', 'gene_biotype', 'DE Status'],
                        title = title,
                        labels = {"log2FoldChange" : "Log2 Fold Change", "neg_log10_pval" : 'Negative Log10 P-value'},
                        color_discrete_map={"Not DEG": "silver",
                                            "Up Reg": "#cc212f",
                                            "Down Reg": "#1052bd"})
    print("Done!")
    return fig


### MA Plots for Visualizing the Expression vs Magnitude of Gene Expression Change ###
def plot_ma(de_result, pvalue_threshold, lfc_threshold, title):
    '''
    Plots an MA plot providing an overview of the differentially expressed genes in terms of average expression 
    and magnitude of change based on the provided p-value and log2 fold change thresholds used

    Parameters
    ----------
    de_result : Pandas DataFrame 
        A Pandas DataFrame containing the differential expression results for the specific comparison
    pvalue_threshold : float
        The p-value threshold that will be used to determine statistically significant changes in gene expression
    lfc_threshold : float
        The log2 fold change threshold that will be used to determine the genes with strong changes in gene expression between conditions
    title : str
        The title of the volcano plot

    Returns
    -------
    fig
        A Plotly scatter plot (MA)
    '''

    print("Plotting MA plot: " + title + "...")
    p_test = de_result["padj"] < pvalue_threshold
    fc_up_test = de_result["log2FoldChange"] > lfc_threshold
    fc_down_test = de_result["log2FoldChange"] < -lfc_threshold
    
    status = ["Not DEG" for x in range(0, de_result.shape[0])] 
    for i in range(0, de_result.shape[0]):
        if p_test[i] and fc_up_test[i]:
            status[i] = "Up Reg"
        if p_test[i] and fc_down_test[i]:
            status[i] = "Down Reg"

    de_result["DE Status"] = status
    de_result["log2_expression"] = np.log2(de_result["baseMean"] + 1)

    fig = px.scatter(de_result, x = "log2_expression", y = "log2FoldChange", 
                        color = "DE Status", 
                        hover_data = ['gene_name', 'gene_biotype', 'DE Status', 'baseMean', 'log2FoldChange', 'pvalue', 'padj'],
                        title = title,
                        labels = {"log2_expression" : 'Log2 Mean Normalized Read Counts', "log2FoldChange" : "Log2 Fold Change"},
                        color_discrete_map={"Not DEG": "silver",
                                            "Up Reg": "#cc212f",
                                            "Down Reg": "#1052bd"})
    print("Done!")
    return fig


### Heatmap Plots for Visualizing the Expression of Multiple Genes over Samples ###
def plot_heatmap(normalized_counts, sample_names, gene_list, 
                    title, output_folder, file_name,
                    cluster_genes = True, cluster_samples = False):
    '''
    Plots a gene expression heatmap (based on normalized read counts) of the provided gene list
    across the sample_names samples 

    Parameters
    ----------
    normalized_counts : Pandas DataFrame 
        A Pandas DataFrame containing normalized gene expression values (float)
    sample_names : list
        A 1D list of sample names (str) to be used for the heatmap
    gene_list : list
        A 1D list of gene ids (str) to specify the genes to be used for the heatmap
    title : str
        The title of the heatmap plot
    output_folder : str
        The path to the folder where the heatmap will be saved as an SVG image
    file_name : str
        The filename to be used to save the heatmap 
    cluster_genes : boolean
        A boolean indicating whether the genes (rows of the heatmap) should be ordered based on hierarchical clustering
    cluster_samples : boolean
        A boolean indicating whether the samples (columns of the heatmap) should be ordered based on hierarchical clustering

    Returns
    -------
    Null
        A SVG image is saved to the specified path as the output
    '''

    print("Plotting gene expression heatmap plot: " + title + "...")
    df_count = normalized_counts.loc[gene_list]

    df = df_count[sample_names]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p = sns.clustermap(df, 
                            z_score = 0, 
                            cmap = "viridis", 
                            linewidth = 0.3, 
                            row_cluster = cluster_genes, 
                            col_cluster = cluster_samples)
    p.fig.suptitle(title)
    p.ax_heatmap.set_xlabel("TCGA Samples")
    p.ax_heatmap.set_ylabel("Ensembl Gene ID")
    plt.savefig(output_folder + "/" + file_name + ".svg", dpi = 300)
    print("Done!")


### Heatmap Plots for Visualizing the Expression of Differentially Expressed Genes ###
def plot_deg_heatmap(normalized_counts, sample_names, 
                        de_result, pvalue_threshold, lfc_threshold, 
                        title, output_folder, file_name, 
                        cluster_genes = True, cluster_samples = False):
    '''
    Plots a gene expression heatmap (based on normalized read counts) of all the differentially expressed genes 
    across the sample_names samples based on the provided p-value and log2 fold change thresholds used

    Parameters
    ----------
    normalized_counts : Pandas DataFrame 
        A Pandas DataFrame containing normalized gene expression values (float)
    sample_names : list
        A 1D list of sample names (str) to be used for the heatmap
    de_result : Pandas DataFrame 
        A Pandas DataFrame containing the differential expression results for the specific comparison
    pvalue_threshold : float
        The p-value threshold that will be used to determine statistically significant changes in gene expression
    lfc_threshold : float
        The log2 fold change threshold that will be used to determine the genes with strong changes in gene expression between conditions
    title : str
        The title of the heatmap plot
    output_folder : str
        The path to the folder where the heatmap will be saved as an SVG image
    file_name : str
        The filename to be used to save the heatmap 
    cluster_genes : boolean
        A boolean indicating whether the genes (rows of the heatmap) should be ordered based on hierarchical clustering
    cluster_samples : boolean
        A boolean indicating whether the samples (columns of the heatmap) should be ordered based on hierarchical clustering

    Returns
    -------
    Null
        A SVG image is saved to the specified path as the output
    '''

    print("Plotting differentially expressed gene heatmap plot: " + title + "...")
    p_test = de_result["padj"] < pvalue_threshold
    fc_up_test = de_result["log2FoldChange"] > lfc_threshold
    fc_down_test = de_result["log2FoldChange"] < -lfc_threshold

    status = ["Not DEG" for x in range(0, de_result.shape[0])] 
    for i in range(0, de_result.shape[0]):
        if p_test[i] and fc_up_test[i]:
            status[i] = "Up Reg"
        if p_test[i] and fc_down_test[i]:
            status[i] = "Down Reg"

    de_result["DE Status"] = status

    gene_list = de_result.index[de_result["DE Status"] != "Not DEG"]
    df_count = normalized_counts.loc[gene_list]
    df_count["DE Status"] = de_result["DE Status"][de_result["DE Status"] != "Not DEG"]

    row_palette = {"Not DEG": "silver", "Up Reg": "#cc212f", "Down Reg": "#1052bd"}
    row_colors = {}
    for gid in df_count.index:
        s = df_count.loc[gid, "DE Status"]
        row_colors[gid] = row_palette[s]
    row_colors = pd.Series(row_colors)

    df = df_count[sample_names]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p = sns.clustermap(df, z_score = 0, cmap = "viridis", 
                            row_cluster = cluster_genes, 
                            col_cluster = cluster_samples,
                            row_colors = row_colors)

    p.fig.suptitle(title)
    p.ax_heatmap.set_xlabel("TCGA Samples")
    p.ax_heatmap.set_ylabel("Ensembl Gene ID")
    plt.savefig(output_folder + "/" + file_name + ".svg", dpi = 300)
    print("Done!")


### Scatter Plots for Correlation ###
def plot_compare_traits(trait1, trait2, title, x_label, y_label):
    '''
    Plots a scatter plot using 2 numerical variables (i.e. trait1 and trait2), 
    calculates the Pearson correlation and the best fit line using OLS

    Parameters
    ----------
    trait1 : list
        A 1D list containing numerical (int) values for the first trait
    trait2 : list
        A 1D list containing numerical (int) values for the second trait
    title : str
        The title for the scatter plot
    x_label : str
        The x-axis label for the scatter plot
    y_label : str
        The y-axis label for the scatter plot

    Returns
    -------
    fig
        A Plotly scatter plot
    '''

    print("Plotting scatter plot: " + title + "...")
    corr = np.corrcoef(x = trait1, y = trait2)
    fig = px.scatter(x = trait1,  y = trait2, 
                        title = title + " - Pearson Cor: %.3f" % (corr[0, 1]), 
                        labels = dict(x = x_label, y = y_label),
                        trendline = "ols")
    print("Done!")
    return fig
