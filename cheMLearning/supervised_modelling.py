import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import r2_score, classification_report
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import LeaveOneOut


class supervised_class():

    def __init__(self):
        self.pls_scores = None
        self.predict = None
        self.pls_coef_determ = None
        self.pls_expl_var = None
        self.r2_nested = []
        self.q2_nested = []
        self.l_v = []
        self.rf_ra_coef_det = None

    def PLS_model(self,
                  X: pd.DataFrame or np.array,
                  y_variable: pd.Series or np.array,
                  n_components: int):
        """
        PLS model method uses NIPALS algotrithm to perfor either PLS-DA or PLS-RA.

        Parameters:
        -----------
        X: data frame or data matrix containing variables of interest.
        y_variable: variable response; it can be either a dummy matrix for DA of [n_samples, n_classes] or a continuous vector for RA of [n_samples].
        n_components: number of latent variables to be computed.

        Returns:
        --------
        self.pls_scores 
        self.pls_coef_determ
        self.pls_expl_var

        """
        pls_model = PLSRegression(
            n_components=n_components, scale=True).fit(X, y_variable)
        # obtaining scores
        self.pls_scores = pls_model.transform(X)
        # obtaining coeficient of determination
        self.pls_coef_determ = pls_model.score(X, y_variable)
        # predicting response, Y
        self.y_pred = pls_model.predict(X)

        def ptc_var(xcores, yloadings, y_variable):
            """
            ptc_var computes explained variance by LV of a NIPALS-PLS-DA model.

            Parameters:
            ----------
            xcores: x_scores_.
            yloadings: y_loadings.
            classes: array of [n_classes,] or unitary-dummy matrix for classes [n_samples, n_classes].

            Returns:
            --------
            PLS explained variance per latent variable of the model.
            """
            c = xcores.shape[1]
            var_explained = []
            for i in range(c):
                y_hat = np.dot(xcores[:, i].reshape(-1, 1), yloadings[:, i].reshape(-1,
                               1).T) * y_variable.std(axis=0, ddof=1) + y_variable.mean(axis=0)
                var_explained.append(r2_score(y_variable, y_hat)*100)
            return var_explained
        self.pls_expl_var = ptc_var(
            xcores=pls_model._x_scores, yloadings=pls_model.y_loadings_, y_variable=y_variable)

        return pls_model

    def score_viz(self,
                  lv_1: int,
                  lv_2: int,
                  color=None,
                  color_discrete_sequence=None,
                  label=None,
                  symbol=None,
                  symbol_sequence=None,
                  width=900,
                  height=700):
        """
        score_viz method allows visualization of the scores belonging to desired latent variables.
        Default dimensions: width=900 x height=700.

        Parameters:
        -----------
        lv_1: int, first latent variable to plot in x axis.
        lv_2: int, second latent variable to plot in y axis.
        color: list containing names to assign a color to.
        color_discrete_sequence: list of color to be assigned. Ref:https://plotly.com/python/discrete-color/
        symbol: list containing the names to assign symbols to.
        symbol_sequence: list specifying the kind of plotly symbols to use. Ref: https://plotly.com/python/marker-style/ .

        Returns:
        --------
        Plotly express scatter plot object. It can be called by the method 'plot()'. The object which can be further modify with "px.update_xxx()" functions.
        """
        plot = px.scatter(data_frame=self.pls_scores, x=lv_1, y=lv_2, color=color, text=label,
                          width=width, height=height, symbol=symbol, template="presentation",
                          symbol_sequence=symbol_sequence, color_discrete_sequence=color_discrete_sequence)
        plot.update_traces(textposition="top center", textfont_size=10)
        plot.update_layout(font=dict(
            size=10), xaxis_title=f'LV {lv_1+1} ({self.pls_expl_var[lv_1]:.2f}%)', yaxis_title=f'LV {lv_2+1} ({self.pls_expl_var[lv_2]:.2f}%)',
            legend_title="Exposure")
        return plot

    def PLS_goodness_metrics(self,
                             X: pd.DataFrame or np.array,
                             y_variable: pd.Series or np.array,
                             n_models: int,
                             vip_metrics=False,
                             threshold=None,
                             n_jobs=None):
        """
        PLS_goodness_metrics method compute R2 and Q2 for nested PLS models; each time including one more LV until the maximun number of LV is reached.

        Parameters:
        -----------
        X: data frame or numpy array containing the variables of interest.
        y_variable: either a dummy matrix or a continuous vector for the variable response.
        n_models: must be the number of latent variables the original PLS computed.
        threshold: int, VIP threshold value.
        vip_metrics: booler, to either compute or not the goodness metrics for each model at each VIP threshold.
        n_jobs: int, number of cores to use for the job.

        Returns:
        --------
        List of computed values and a list containing the number of latent variables. It corresponds to : attributes self.r2_nested, self.q2_nested and
        self.l_v.

        """

        r2_vip_metrics = {}
        q2_vip_metrics = {}
        vip_betas = {}
        for i in np.arange(n_models):
            self.l_v.append(i+1)
            # Fitting PLS model
            pls_model = PLSRegression(
                n_components=i+1, scale=1).fit(X=X, Y=y_variable)
            y_hat_train = pls_model.predict(X)
            # r2 computation
            r2 = r2_score(y_true=y_variable, y_pred=y_hat_train)

            # cross-validation
            cv = LeaveOneOut()
            y_hat_cv = cross_val_predict(
                estimator=pls_model, cv=cv, X=X, y=y_variable, n_jobs=n_jobs)

            # q2 computation
            q2 = r2_score(y_true=y_variable, y_pred=y_hat_cv)

            if vip_metrics != True:
                self.r2_nested.append(r2)
                self.q2_nested.append(q2)
            else:
                r2_vip_metrics[f'r2_tr_{threshold}_LVs_{i+1}'] = r2
                q2_vip_metrics[f'q2_tr_{threshold}_LVs_{i+1}'] = q2
                vip_betas[f'beta_coef_tr_{threshold}_LVs_{i+1}'] = pls_model.coef_
        if vip_metrics:
            return [r2_vip_metrics, q2_vip_metrics, vip_betas]

    def PLS_goodness_metrics_viz(self,
                                 height=700,
                                 width=720,
                                 **kwargs):
        """
        PLS_goodness_metrics_viz method generates a scatter plot to visualize on the value of both R2 and Q2 as a function of the number of latent variables.
        Default dimensions: height=700, width=720.

        Parameters:
        ----------
        **kwargs for update_layout() methods to check wheter I can pass a title whenever want.

        Return:
        ------
        Plotly go.Figure, scatter plot.
        """

        plot = make_subplots(
            rows=1, cols=1, y_title="R2 & Q2", x_title="Latent Variables")
        plot.add_trace(go.Scatter(x=self.l_v, y=self.r2_nested, name="R²"))
        plot.add_trace(go.Scatter(x=self.l_v, y=self.q2_nested, name="Q²"))
        plot.update_layout(legend_title_text="Model validation metrics",
                           hovermode="x unified", template="presentation", height=height, width=width, **kwargs)
        return plot

    @staticmethod
    def VIP(x: pd.DataFrame or np.array,
            model):
        """
        Computes PLS Variable Importance in Projection (VIP).

        Parameter
        ---------
        x: data frames.
        model: scikit-learn PLS model.

        Return
        ------
        VIP scores.

        References
        ---------
        Tahir Mehmood, Kristian Hovde Liland, Lars Snipen, and Solve Sæbø. A review of variable
        selection methods in partial least squares regression. Chemometrics and intelligent laboratory
        systems, 118:62–69, 2012.

        Author
        ------
        https://github.com/scikit-learn/scikit-learn/issues/7050
        """
        t = model.x_scores_
        w = model.x_weights_
        q = model.y_loadings_

        m, p = x.shape
        _, h = t.shape

        vips = np.zeros((p,))

        s = np.diag(t.T @ t @ q.T @ q).reshape(h, -1)
        total_s = np.sum(s)

        for i in range(p):
            weight = np.array(
                [(w[i, j] / np.linalg.norm(w[:, j]))**2 for j in range(h)])
            vips[i] = np.sqrt(p*(s.T @ weight)/total_s)

        return vips

    @staticmethod
    def VIP_subset(vips, threshold, X):
        """
        Performs df extraction from original df according to the threshold value passed.

        Parameter
        --------
        vips: data frame or data matrix containing VIP values.
        threshold: threshod value, it can be integer or float.
        X: data frame containing annotated metabolites from where extraction is carried out based on the VIP threshold and values.

        Return
        ------
        Data frame containing the extracted metabolites of interest. The content is the values of the original data frame.

        Author:
        ------
        Christian Peralta

        """
        # vip marix is converted to a pd.DataFrame and transpose, so later the variable name can be included in the output
        v = pd.DataFrame(vips).T
        # filtering according to a selected threshold
        v = v[v >= threshold].dropna(axis=1)
        # DF-v now contraind the variables of interest according to the VIP threshold.
        # Variable name is set to object and stored in a variable for iteration
        features_subset = v.columns.astype("object")
        # setting an empty dictionary which will be converted to a DF
        d = {}
        # loop setting as key:value argument as variable_position:column of the DF-X
        for i in features_subset:
            d[i] = X[str(i)]
        d = pd.DataFrame(d)
        return d

    def RandomForest_Classifier(self,
                                X: pd.DataFrame or np.array,
                                y_variable: pd.Series or np.array,
                                **kwargs):
        """"
        RandomForest_Classifier method is a wrap-up method from the corresponding sklearn method which fits the model directly
        and returns the model as well as the score for classification, the accuracy.

        Parameters:
        ----------
        X: data frame.
        y_variable: response variable suited for classification analysis either [n_samples,] or [n_samples, n_classes].

        Returns:
        -------
        The model  itself and the accuracy score as self.rf_ra_accuracy.
        """

        rf_da = RandomForestClassifier(
            **kwargs, oob_score=True).fit(X=X, y=y_variable)
        self.rf_da_accuracy = rf_da.score(X=X, y=y_variable)
        return rf_da

    def RandomForest_Regressor(self,
                               X: pd.DataFrame or np.array,
                               y_variable: pd.Series or np.array,
                               **kwargs):
        """"
        RandomForest_Classifier method is a wrap-up method from the corresponding sklearn method which fits the model directly
        and returns the model as well as the coefficient of determination R².

        Parameters:
        ----------
        df: data frame.
        y_variable: response variable suited for regression analysis should be [n_samples].

        Returns:
        -------
        The model  itself and the coefficient of determination as self.rf_ra_accuracy.
        """

        rf_ra = RandomForestRegressor(
            **kwargs, oob_score=True).fit(X=X, y=y_variable)
        self.rf_ra_coef_det = rf_ra.score(X=X, y=y_variable)
        return rf_ra

    @staticmethod
    def Classification_report(
            y_variable,
            y_hat,
            target_names=None):
        """
        How to specify target named:
        target_names=encoder.classes_
        """
        print(classification_report(y_true=np.argmax(y_variable, axis=1),
              y_pred=np.argmax(y_hat, axis=1), target_names=target_names))

    @staticmethod
    def distanceMatrix(estimator,
                       X):
        """
        Computes distance matrix to map samples in the forest.

        Parameters:
        ----------

        estimator: random forest classifier model.
        X: data frame or data matrix used to fit the model.

        Return:
        ------
        Matrix containing the distance between observations in the forest.
        """
        terminals = estimator.apply(X)
        nTrees = terminals.shape[1]

        a = terminals[:, 0]

        # assest whether values are equal or not and by multiplying by 1 it obtains a binary matrix
        proxMat = 1*np.equal.outer(a, a)

        # initialitzing the first freq count
        for i in range(0, nTrees):
            a = terminals[:, i]
            proxMat += 1*np.equal.outer(a, a)

        distanceMat = 1-(proxMat / nTrees)

        return distanceMat
