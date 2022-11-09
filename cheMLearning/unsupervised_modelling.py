import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering


class unsupervised_class():
    """
    Unsupervised class provides high-level tools commonly applied in unsupervised multivariate analysis on untargeted metabolomics.
    The idea is providing of simple tools to carry out chemometrics analysis in Python with short amount of code. Outputs can be accessed 
    via class instances.

    Methods:
    --------
    PCA_ready
    score_viz
    loading_barplot
    loading_contourplot
    HCA_dendogram
    Cluster_plot
    """

    def __init__(self):
        self.pca = PCA(n_components=.95)
        self.var = None
        self.scores = None
        self.loadings = None

    def PCA_ready(self, df: np.array or pd.DataFrame):
        """
        PCA_ready method computes PCA on data centered and scaled to unit variance. The data frame should be transposed and only contain numerical values
        releted to the features. Scores, loadings, variance and PCA model are stored as attributes of the class; which are initially empty.

        Parameters:
        ----------
        df: pandas data frame or array.

        Return:
        -------
        self.var
        self.scores
        self.loadings

        References:
        ----------
        Documentation on PCA algorithm, read: https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html.
        """
        # setting up pipeline with standarization step as well as pca computation
        pca_pipe = Pipeline([("scale", StandardScaler()), ("pca", self.pca)])
        # computing scores and saving it as instance
        self.scores = pca_pipe.fit_transform(df)
        # computing loadings
        self.loadings = self.pca.components_.T * \
            np.sqrt(self.pca.explained_variance_ratio_)
        # obtaining variance
        self.var = self.pca.explained_variance_ratio_*100

    def score_viz(self,
                  pcx: int,
                  pcy: int,
                  color=None,
                  color_discrete_sequence=None,
                  label=None,
                  label_position=None,
                  symbol=None,
                  symbol_sequence=None,
                  hover_name=None,
                  width=900,
                  height=700,
                  title=None):
        """
        Score_viz method enable the visualizaton of PCA score plot by a Plotly.Express scatter plot.
        Default dimensions: width=900 x height=700.

        Parameters:
        ----------
        pcx: PC to be displayed on x axis. Usage: scores_array[:, pcx].
        pcy: PC to be displayed on y axis. Usage: scores_array[:, pcy].
        color: data frame column or array or list containing categorical variable used to assing color. Usage: df["column_name"].
        color_discrete_sequence: list of color to be assigned. Ref:https://plotly.com/python/discrete-color/.
        label: data frame column or array or list containing categorical variable used to display the label of scores. Usage: df["column_name"].
        label_postionn: can be: 'top center/left/right', 'middle center/left/right'.
        symbol: data frame column or array or list containing categorical variable to assign a symbol to.
        symbol_sequence: list specifying the kind of plotly symbols to use. Ref: https://plotly.com/python/marker-style/ .
        hover_name: data frame column or array or list containing names to show in HTML format.
        width: default is 900.
        height: default is 700.
        title: default None, pass string.

        Return:
        -------
        Plotly express scatter plot object. To visualize, use object.show(). The object which can be further modify with "px.update_xxx()" functions.
        """
        # generating score plot by passing self.scores also for the xaxis, otherwise one cannot customize axis label
        score_plot = px.scatter(data_frame=self.scores, x=self.scores[:, pcx], y=self.scores[:, pcy], symbol=symbol,
                                color=color,
                                labels={
            'x': f'PC{pcx+1} ({self.var[pcx]:.1f}%)', 'y': f'PC{pcy+1} ({self.var[pcy]:.1f}%)'},
            template="presentation", width=width, height=height, text=label,
            title=title, symbol_sequence=symbol_sequence, color_discrete_sequence=color_discrete_sequence, hover_name=hover_name)
        # adding legend title, and title position
        score_plot.update_layout(legend_title="Exposure", title={
                                 "xanchor": "center", "x": .42}, font=dict(size=10))
        # modifying label font size as well as position
        score_plot.update_traces(
            textfont_size=10, textposition=label_position)

        return score_plot

    def loading_barplot(self,
                        pcx: int):
        """
        Loading_barplot method provides visualization of loading of a single PC as bar plot. Usefult when that PC is driven by the differencies of groups control and exposed.

        Parameter:
        ----------
        pcx: PC to be displayed in the x axis.
        """
        loading_barplot = px.bar(data_frame=self.loadings[:, pcx], template="presentation",
                                 labels={"index": "Features",
                                         "variable": "Loadings Value"},
                                 title=f"Loading Bar Plot of PC{pcx+1}")
        loading_barplot.update_layout(showlegend=False)
        return loading_barplot

    def loading_contourplot(self,
                            pcx: int,
                            pcy: int,
                            **kwargs):
        """
        Loading_contoutplot method provides contour (density plot) visualization of loadings corresponding to two desired PC. If needed, after passing a parameter it can display histograms of the PCs.

        Parameters:
        -----------
        pcx: PC to be displayed in the x axis. Usage: loadigs_array[:, pcx].
        pcy: PC to be displayed in the y axis. Usage: loadigs_array[:, pcy].

        Return:
        ------
        Plotly express scatter plot object.
        """
        histogram = kwargs.get("histogram", None)
        contour_plot = go.Figure()
        contour_plot.add_trace(go.Histogram2dContour(
            x=self.loadings[:, pcx], y=self.loadings[:, pcy], colorscale="Jet", xaxis="x", yaxis="y"))
        contour_plot.add_trace(go.Scatter(
            x=self.loadings[:, pcx], y=self.loadings[:, pcy], mode="markers", marker=dict(size=3), xaxis="x", yaxis="y"))
        contour_plot.update_layout(
            template="presentation", height=700, width=900, hovermode=False, font=dict(size=10))
        # add title!!!!

        if histogram:
            contour_plot.add_trace(go.Histogram(
                x=self.loadings[:, pcx], yaxis="y2"))
            contour_plot.add_trace(go.Histogram(
                y=self.loadings[:, pcy], xaxis="x2"))
            contour_plot.update_layout(
                xaxis=dict(zeroline=False, showgrid=False, domain=[0, .85]),
                yaxis=dict(zeroline=False, showgrid=False, domain=[0, .85]),
                xaxis2=dict(zeroline=False, showgrid=False, domain=[.85, 1]),
                yaxis2=dict(zeroline=False, showgrid=False, domain=[.85, 1]),
                height=700, width=900, showlegend=False, font=dict(size=10))
            return contour_plot
        else:
            return contour_plot

    def HCA_dendrogram(self, label):
        """
        HCA_dendrogram computes and visualize HCA dendrogram on PC scores.

        Parameter:
        ----------
        label: data frame column containing categorical variable to use as labels. Usage: df["column_name"].

        Return:
        -------
        Matplotlib object.

        References:
        ----------
        Documentation on HCA algorithm: https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.dendrogram.html.
        """
        plt.figure(figsize=(10, 10))
        dendrogram = sch.dendrogram(sch.linkage(
            self.scores, method='ward'), labels=label.tolist(), orientation='left')
        return plt

    def cluster_viz(self,
                    pcx: int,
                    pcy: int,
                    label=None,
                    symbol=None,
                    symbol_sequence=None,
                    color_discrete_sequence=None,
                    **kwargs):
        """

        cluster_viz allows to visualize score plot from PCA where the color corresponds to cluster groups from HCA.

        Parameters:
        -----------
        pcx: PC to be displayed on the X axis.
        pcy: PC to be displayed on the y axis.
        label: column index used to display as label of the samples.

        Return:
        -------
        Plotly express scatter plot object.
        """
        n_cluster = kwargs.get("n_cluster", None)
        if n_cluster:
            cluster = AgglomerativeClustering(
                affinity='euclidean', linkage='ward', n_clusters=n_cluster)
        else:
            cluster = AgglomerativeClustering(
                affinity='euclidean', linkage='ward')

        label_cluster = cluster.fit_predict(self.scores)
        label_group = []
        for i in label_cluster:
            label_group.append(f'Group {i}')
        scores_cluster = np.column_stack((self.scores, label_cluster))
        #scores_cluster[:, len(self.scores[1])]
        cluster_plot = px.scatter(data_frame=scores_cluster, x=scores_cluster[:, pcx], y=scores_cluster[:, pcy],
                                  color=label_group, color_discrete_sequence=color_discrete_sequence,
                                  labels={
            'x': f'PC{pcx+1} ({self.var[pcx]:.2f}%)', 'y': f'PC{pcy+1} ({self.var[pcy]:.2f}%)'},
            color_discrete_map={"1": "orange", "0": "red"},
            template="presentation", width=900, height=700, text=label,
            title=f'Cluster Plot', symbol=symbol, symbol_sequence=symbol_sequence)
        cluster_plot.update_layout(
            legend_title="Cluster Group", font=dict(size=10))
        cluster_plot.update_traces(
            textfont_size=10, textposition='top center')
        return cluster_plot

    @staticmethod
    def PQN(path: str):
        """
        PQN is a wrapper function for median-based Probabilistic Quotient Normalization.

        As input, file named as "xxx.csv" which contains tranposed data, where the first column is the sample names.
        Quality controls and diluted quality controls have to be named as "*QC*" and "dQC*" respectively.

        The output is the normalised data named as "PQN_xxx1".

        Parameter:
        ----------
        path: path of the file to be normalised.

        References:
        ----------
        Frank Dieterle, Alfred Ross, Götz Schlotterbeck, Hans Senn. Probabilistic Quotient Normalization as Robust Method to Account 
        for Dilution of Complex Biological Mixtures. Application in 1H NMR Metabonomics. Anal. Chem. 2006, 78, 4281-4290.

        """
        # importing the data frame from path, passing the first column as index to use filter over it
        df = pd.read_csv(path, sep=",", index_col=0)
        # filtering QCs to concatenate with the normalized df later
        qcs = df.filter(axis=0, regex="^QC")
        # filtering df to obtain only QCs and computing the reference vector which is the mean across all QCs samples
        ref_vector = np.asarray(df.filter(axis=0, regex="^QC*").median(axis=0))
        # filtering the data to exclude QCs and using apply to compute the quotients, variables divided by ref_vector
        quotients = df.filter(axis=0, regex="(?m)^(?!QC)").apply(
            lambda x: x/ref_vector, axis=1, result_type="expand")
        # compting the coeficient vector for each sample, the scalar.
        coef_vector = np.asarray(quotients.median(axis=1))
        # filtering to exclude QC samples and computing normalization of each sample dividing by coef_vector
        pqn_df = df.filter(axis=0, regex="(?m)^(?!QC)").apply(
            lambda x: x/coef_vector, axis=0, result_type="expand")
        # concatenating pqn_df with QC samples to obtain a final df to be exported
        export_df = pd.concat([pqn_df, qcs])
        # setting file name
        file_name = f'PQN_{path.split("_")[0]}.csv'
        # exporting file to csv format
        export_df.to_csv(file_name, header=True, index=True)
        print("All done")
        return export_df
