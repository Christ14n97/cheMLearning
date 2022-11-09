from typing import Any
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, roc_curve, auc
from xlrd import open_workbook_xls


def ConfusionMat_viz(y, y_hat, classes_names: Any or None = None, title: str or None = None, text_auto=True, width=600, height=600, color_continuous_scale="YlGnBu"):
    """
    ConfusionMat_viz function is designed to compute and visulize confusion matrix for multilabel classification problem.
    It takes two Y matrices, one for true y and one for predicted y, and obtains the index where the maximum argument
    is located in the axis 1.

    Parameters:
    ----------
    y: true y, dummy matrix for multilabel classification of shape [n_samples, n_classes].
    y_hat: predicted y of shape [n_samples, n_classes].
    classes_names: array containing the name of the classes [n_classes].
    tittle: string to be display as title.
    text_auto: is set to default to show the value of each cell.
    width: default 600.
    height: default 600.
    colo_continuous_scale: default "YlGnBu".

    Returns:
    --------
    Plotly express object imshow.
    """

    cm = confusion_matrix(y_true=np.argmax(y, axis=1),
                          y_pred=np.argmax(y_hat, axis=1))
    cm_viz = px.imshow(cm, labels=dict(x="Predicted Class",
                       y="True Class"), x=classes_names, y=classes_names, text_auto=text_auto, width=width, height=height, color_continuous_scale=color_continuous_scale)
    cm_viz.update_layout(title=title)
    return cm_viz


def ROC_AUC_viz(n_classes, y_classes, y_hat, width=1000, height=500):
    """
    ROC_AUC_viz function computes Receiver Operation Curve as well as Area Under the Curve for classification problem and returns it's visualization.

    Parameters:
    ----------
    n_classes: array containing the number of classes with shape [n_classes].
    y_classes: dummy matrix of shape [n_samples, n_classes].
    y_hat: can be either a matrix of shape [n_samples, n_classes] or [n_samples,].

    Return:
    ------
    Plotly goFigure object of the visualization for ROC.
    """
    fig = go.Figure()
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    def roc_computation(class_index):
        if y_hat.shape[1] == 1:
            fpr[class_index], tpr[class_index], _ = roc_curve(
                y_true=y_classes[:, class_index], y_score=y_hat)
        else:
            fpr[class_index], tpr[class_index], _ = roc_curve(
                y_true=y_classes[:, class_index], y_score=y_hat[:, class_index])
        roc_auc[class_index] = auc(fpr[class_index], tpr[class_index])

    list(map(roc_computation, np.arange(len(n_classes))))

    def add_trace(class_index):
        name = f'ROC curve of class {n_classes[class_index]} (area = {roc_auc[class_index]:.2f})'
        fig.add_trace(go.Scatter(
            x=fpr[class_index], y=tpr[class_index], name=name, mode="lines"))
    list(map(add_trace, np.arange(len(n_classes))))
    fig.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)
    fig.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
                      yaxis=dict(scaleratio=1), xaxis=dict(scaleratio=1),
                      width=width, height=height, template="presentation", hovermode=False)

    return fig


def Read_MetRank_xls(path: str,
                     sheet_name=1,
                     col_names=None):
    """
    Function to read MetExplore, MetaboRank excel files.

    Parameters:
    ----------
    Path: path to file.
    sheet_name: int, default to 1.
    col_names: list containing the names of teh columns to retain.

    Return:
    ------
    Pandas data frame

    Example:
    -------
    >>> col_names=["Name", "Identifier", "Formula", "Monoisotopic_mass","chebi", "hmdb", "inchi", "inchikey","kegg", "pubchem", "Fingerprint", "MR_OUT", "MR_IN"]
    >>> mr_output = Read_MetRank_xls("MR_noChemBackg.xls", col_names=col_names, sheet_name=3)
    """
    wk = open_workbook_xls(filename=path, ignore_workbook_corruption=True)
    file = pd.read_excel(wk, sheet_name=sheet_name, names=col_names)
    return file


def calculate_MassAccuracy(
        theoretical_df: pd.DataFrame,
        experimental_df: pd.DataFrame,
        colTheo_mass="Monoisotopic_mass",
        colExp_mass="Neutral mass (Da)",
        tolerance=5):
    """
    Description
    """
    if colExp_mass == "Neutral mass (Da)":
        # Getting the neutral mass for those features that where not annotated ("Accepted ID") and have MS fragmentation data
        expe_data = np.asanyarray(experimental_df[pd.isna(experimental_df["Accepted ID"])][~pd.isna(
            experimental_df["MSMS info available"])][["Neutral mass (Da)"]].dropna())
    else:
        expe_data = experimental_df[colExp_mass]

    def iterator(theor_vector):
        if expe_data[abs(1e6*(expe_data-theor_vector)/expe_data) < tolerance].any() == True:
            # obtaining the expected m/z value
            exp_hit = np.r_[
                expe_data[abs(1e6*(expe_data-theor_vector)/expe_data) < tolerance].values]
            # computing ppm mass error
            err_ppm = np.round(
                abs(1e6*(exp_hit-theor_vector)/exp_hit), decimals=2)
            theo_hit = np.r_[theor_vector]
            # data manipulation
            if colTheo_mass == "Monoisotopic_mass":
                theo_filtered = theoretical_df[theoretical_df[colTheo_mass].isin(theo_hit)].loc[:, [
                    "Name", "Formula", "Monoisotopic_mass"]].sort_values(by="Monoisotopic_mass")
                exp_filtered = experimental_df[experimental_df[colExp_mass].isin(
                    exp_hit)].loc[:, ["Compound", "Neutral mass (Da)"]].sort_values(by="Neutral mass (Da)")
                final_df = pd.concat([theo_filtered.reset_index(drop=True), exp_filtered.reset_index(drop=True)], axis="columns").loc[:, [
                    "Name", "Formula", "Monoisotopic_mass", "Neutral mass (Da)", "Compound"]]
            else:
                theo_filtered = theoretical_df[theoretical_df[colTheo_mass].isin(theo_hit)].loc[:, [
                    "Name", "adduct", "Monoisotopic_mass", "Formula", "adduct mass", "smiles"]].sort_values(by="adduct mass")
                exp_filtered = experimental_df[experimental_df[colExp_mass].isin(
                    exp_hit)].loc[:, ["Compound", "m/z"]].sort_values(by="m/z")
                final_df = pd.concat([theo_filtered.reset_index(drop=True), exp_filtered.reset_index(drop=True)], axis="columns").loc[:, [
                    "Name", "Formula", "adduct mass", "m/z", "adduct", "Compound", "smiles"]]
            final_df["error_ppm"] = err_ppm
            return final_df
    l = list(map(iterator, theoretical_df[colTheo_mass]))
    result = [i for i in l if i is not None]
    return result
