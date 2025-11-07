"""Plotly plotting backend."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import plotly.graph_objects as go

    from dataeval.outputs import Output


class PlotlyBackend:
    """Plotly implementation of plotting backend with interactive visualizations."""

    def plot(self, output: Output, **kwargs: Any) -> go.Figure | list[go.Figure]:
        """
        Route to appropriate plot method based on output type.

        Parameters
        ----------
        output : Output
            DataEval output object
        **kwargs
            Plotting parameters

        Returns
        -------
        Figure or list[Figure]
            Plotly figure object(s)

        Raises
        ------
        NotImplementedError
            If plotting not implemented for output type
        """
        # Import all output types
        from dataeval.outputs import (
            BalanceOutput,
            BaseStatsOutput,
            CoverageOutput,
            DiversityOutput,
            DriftMVDCOutput,
            SufficiencyOutput,
        )

        # Route to appropriate plotting function
        if isinstance(output, CoverageOutput):
            return self._plot_coverage(output, **kwargs)
        elif isinstance(output, BalanceOutput):
            return self._plot_balance(output, **kwargs)
        elif isinstance(output, DiversityOutput):
            return self._plot_diversity(output, **kwargs)
        elif isinstance(output, SufficiencyOutput):
            return self._plot_sufficiency(output, **kwargs)
        elif isinstance(output, BaseStatsOutput):
            return self._plot_base_stats(output, **kwargs)
        elif isinstance(output, DriftMVDCOutput):
            return self._plot_drift_mvdc(output, **kwargs)
        else:
            raise NotImplementedError(f"Plotting not implemented for {type(output).__name__}")

    def _plot_coverage(
        self,
        output: Any,  # CoverageOutput
        images: Any = None,  # Images | Dataset
        top_k: int = 6,
    ) -> Any:  # go.Figure
        """
        Plot the top k images together for visualization.

        Parameters
        ----------
        output : CoverageOutput
            The coverage output object to plot
        images : Images or Dataset
            Original images (not embeddings) in (N, C, H, W) or (N, H, W) format
        top_k : int, default 6
            Number of images to plot (plotting assumes groups of 3)

        Returns
        -------
        plotly.graph_objects.Figure
        """
        import base64
        from io import BytesIO

        import numpy as np
        import plotly.graph_objects as go
        from PIL import Image
        from plotly.subplots import make_subplots

        from dataeval.data._images import Images
        from dataeval.protocols import Dataset
        from dataeval.utils._array import as_numpy, channels_first_to_last

        if images is None:
            raise ValueError("images parameter is required for coverage plotting")

        images_obj = Images(images) if isinstance(images, Dataset) else images
        if np.max(output.uncovered_indices) > len(images_obj):
            raise ValueError(
                f"Uncovered indices {output.uncovered_indices} specify images "
                f"unavailable in the provided number of images {len(images_obj)}."
            )

        # Determine which images to plot
        selected_indices = output.uncovered_indices[:top_k]
        num_images = min(top_k, len(selected_indices))

        rows = int(np.ceil(num_images / 3))
        cols = min(3, num_images)

        # Create subplots
        fig = make_subplots(
            rows=rows,
            cols=cols,
            subplot_titles=[f"Image {i}" for i in range(num_images)],
        )

        for idx, img in enumerate(images_obj[:num_images]):
            img_np = channels_first_to_last(as_numpy(img))

            # Normalize to 0-255 range if needed
            if img_np.max() <= 1.0:
                img_np = (img_np * 255).astype(np.uint8)
            else:
                img_np = img_np.astype(np.uint8)

            # Convert to PIL Image
            pil_img = Image.fromarray(img_np)

            # Convert to base64
            buffered = BytesIO()
            pil_img.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()

            row = idx // 3 + 1
            col = idx % 3 + 1

            # Add image to subplot
            fig.add_layout_image(
                dict(
                    source=f"data:image/png;base64,{img_str}",
                    xref=f"x{idx + 1}" if idx > 0 else "x",
                    yref=f"y{idx + 1}" if idx > 0 else "y",
                    x=0,
                    y=1,
                    sizex=1,
                    sizey=1,
                    sizing="stretch",
                    layer="below",
                ),
                row=row,
                col=col,
            )

            # Hide axes for this subplot
            fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False, row=row, col=col)
            fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False, row=row, col=col)

        fig.update_layout(
            title=f"Top {num_images} Uncovered Images",
            showlegend=False,
            height=300 * rows,
            width=300 * cols,
        )

        return fig

    def _plot_balance(
        self,
        output: Any,  # BalanceOutput
        row_labels: Any = None,  # Sequence[Any] | NDArray[Any] | None
        col_labels: Any = None,  # Sequence[Any] | NDArray[Any] | None
        plot_classwise: bool = False,
    ) -> Any:  # go.Figure
        """
        Plot a heatmap of balance information.

        Parameters
        ----------
        output : BalanceOutput
            The balance output object to plot
        row_labels : ArrayLike or None, default None
            List/Array containing the labels for rows in the histogram
        col_labels : ArrayLike or None, default None
            List/Array containing the labels for columns in the histogram
        plot_classwise : bool, default False
            Whether to plot per-class balance instead of global balance

        Returns
        -------
        plotly.graph_objects.Figure
        """
        import numpy as np
        import plotly.graph_objects as go

        if plot_classwise:
            if row_labels is None:
                row_labels = output.class_names
            if col_labels is None:
                col_labels = output.factor_names

            data = output.classwise
            xlabel = "Factors"
            ylabel = "Class"
            title = "Classwise Balance"
        else:
            # Combine balance and factors results
            data = np.concatenate(
                [
                    output.balance[np.newaxis, 1:],
                    output.factors,
                ],
                axis=0,
            )
            # Create a mask for the upper triangle
            mask = np.triu(data + 1, k=0) < 1
            data = np.where(mask, np.nan, data)[:-1]

            if row_labels is None:
                row_labels = output.factor_names[:-1]
            if col_labels is None:
                col_labels = output.factor_names[1:]

            xlabel = ""
            ylabel = ""
            title = "Balance Heatmap"

        # Create heatmap with annotations
        text = [[f"{val:.2f}" if not np.isnan(val) else "" for val in row] for row in data]

        fig = go.Figure(
            data=go.Heatmap(
                z=data,
                x=[str(label) for label in col_labels],
                y=[str(label) for label in row_labels],
                colorscale="Viridis",
                zmin=0,
                zmax=1,
                text=text,
                texttemplate="%{text}",
                textfont={"size": 10},
                colorbar=dict(title="Normalized Mutual Information"),
                hovertemplate="Row: %{y}<br>Col: %{x}<br>Value: %{z:.2f}<extra></extra>",
            )
        )

        fig.update_layout(
            title=title,
            xaxis_title=xlabel,
            yaxis_title=ylabel,
            width=600,
            height=600,
            xaxis=dict(tickangle=-45),
        )

        return fig

    def _plot_diversity(
        self,
        output: Any,  # DiversityOutput
        row_labels: Any = None,  # Sequence[Any] | NDArray[Any] | None
        col_labels: Any = None,  # Sequence[Any] | NDArray[Any] | None
        plot_classwise: bool = False,
    ) -> Any:  # go.Figure
        """
        Plot a heatmap or bar chart of diversity information.

        Parameters
        ----------
        output : DiversityOutput
            The diversity output object to plot
        row_labels : ArrayLike or None, default None
            List/Array containing the labels for rows in the histogram
        col_labels : ArrayLike or None, default None
            List/Array containing the labels for columns in the histogram
        plot_classwise : bool, default False
            Whether to plot per-class balance instead of global balance

        Returns
        -------
        plotly.graph_objects.Figure
        """
        from dataclasses import asdict

        import plotly.graph_objects as go

        if plot_classwise:
            if row_labels is None:
                row_labels = output.class_names
            if col_labels is None:
                col_labels = output.factor_names

            data = output.classwise
            method = asdict(output.meta())["arguments"]["method"].title()

            # Create heatmap with annotations
            text = [[f"{val:.2f}" for val in row] for row in data]

            fig = go.Figure(
                data=go.Heatmap(
                    z=data,
                    x=[str(label) for label in col_labels],
                    y=[str(label) for label in row_labels],
                    colorscale="Viridis",
                    zmin=0,
                    zmax=1,
                    text=text,
                    texttemplate="%{text}",
                    textfont={"size": 10},
                    colorbar=dict(title=f"Normalized {method} Index"),
                    hovertemplate="Row: %{y}<br>Col: %{x}<br>Value: %{z:.2f}<extra></extra>",
                )
            )

            fig.update_layout(
                title="Classwise Diversity",
                xaxis_title="Factors",
                yaxis_title="Class",
                width=600,
                height=600,
                xaxis=dict(tickangle=-45),
            )
        else:
            # Bar chart for diversity indices
            heat_labels = ["class_labels"] + list(output.factor_names)

            fig = go.Figure(
                data=go.Bar(
                    x=heat_labels,
                    y=output.diversity_index,
                    marker=dict(color=output.diversity_index, colorscale="Viridis", showscale=True),
                    text=[f"{val:.3f}" for val in output.diversity_index],
                    textposition="outside",
                    hovertemplate="Factor: %{x}<br>Diversity: %{y:.3f}<extra></extra>",
                )
            )

            fig.update_layout(
                title="Diversity Index by Factor",
                xaxis_title="Factors",
                yaxis_title="Diversity Index",
                width=700,
                height=500,
                xaxis=dict(tickangle=-45),
            )

        return fig

    def _plot_sufficiency(
        self,
        output: Any,  # SufficiencyOutput
        class_names: Any = None,  # Sequence[str] | None
        show_error_bars: bool = True,
        show_asymptote: bool = True,
        reference_outputs: Any = None,  # Sequence[SufficiencyOutput] | SufficiencyOutput | None
    ) -> list[Any]:  # list[go.Figure]
        """
        Plotting function for data sufficiency tasks.

        Parameters
        ----------
        output : SufficiencyOutput
            The sufficiency output object to plot
        class_names : Sequence[str] | None, default None
            List of class names
        show_error_bars : bool, default True
            True if error bars should be plotted, False if not
        show_asymptote : bool, default True
            True if asymptote should be plotted, False if not
        reference_outputs : Sequence[SufficiencyOutput] | SufficiencyOutput, default None
            Singular or multiple SufficiencyOutput objects to include in plots

        Returns
        -------
        list[plotly.graph_objects.Figure]
            List of Plotly figures for each measure
        """
        import numpy as np
        import plotly.graph_objects as go

        # Extrapolation parameters
        last_X = output.steps[-1]
        geomshape = (0.01 * last_X, last_X * 4, len(output.steps))
        projection = np.geomspace(*geomshape).astype(np.int64)

        # Wrap reference
        if reference_outputs is None:
            reference_outputs = []
        elif not isinstance(reference_outputs, (list, tuple)):
            reference_outputs = [reference_outputs]

        figures = []

        # Helper function to create projection curve
        def project_steps(params: Any, proj: Any) -> Any:
            return 1 - (params[0] * proj ** (-params[1]) + params[2])

        for name, measures in output.averaged_measures.items():
            if measures.ndim > 1:
                # Multi-class plotting
                if class_names is not None and len(measures) != len(class_names):
                    raise IndexError("Class name count does not align with measures")

                for i, values in enumerate(measures):
                    class_name = str(i) if class_names is None else class_names[i]

                    fig = go.Figure()

                    # Projection curve
                    proj_values = project_steps(output.params[name][i], projection)
                    fig.add_trace(
                        go.Scatter(
                            x=projection,
                            y=proj_values,
                            mode="lines",
                            name="Potential Model Results",
                            line=dict(width=2),
                            hovertemplate="Step: %{x}<br>Value: %{y:.4f}<extra></extra>",
                        )
                    )

                    # Actual measurements
                    error_y = None
                    if show_error_bars and name in output.measures:
                        error = np.std(output.measures[name][:, :, i], axis=0)
                        error_y = dict(type="data", array=error, visible=True)

                    fig.add_trace(
                        go.Scatter(
                            x=output.steps,
                            y=values,
                            mode="markers",
                            name="Model Results",
                            marker=dict(size=10),
                            error_y=error_y,
                            hovertemplate="Step: %{x}<br>Value: %{y:.4f}<extra></extra>",
                        )
                    )

                    # Add asymptote if requested
                    if show_asymptote:
                        bound = 1 - output.params[name][i][2]
                        fig.add_trace(
                            go.Scatter(
                                x=[projection[0], projection[-1]],
                                y=[bound, bound],
                                mode="lines",
                                name=f"Asymptote: {bound:.4g}",
                                line=dict(dash="dash", width=2),
                                hovertemplate="Asymptote: %{y:.4f}<extra></extra>",
                            )
                        )

                    fig.update_layout(
                        title=f"{name} Sufficiency - Class {class_name}",
                        xaxis_title="Steps",
                        yaxis_title=name,
                        xaxis_type="log",
                        width=700,
                        height=500,
                        hovermode="closest",
                    )

                    figures.append(fig)
            else:
                # Single-class plotting
                fig = go.Figure()

                # Projection curve
                proj_values = project_steps(output.params[name], projection)
                fig.add_trace(
                    go.Scatter(
                        x=projection,
                        y=proj_values,
                        mode="lines",
                        name="Potential Model Results",
                        line=dict(width=2),
                        hovertemplate="Step: %{x}<br>Value: %{y:.4f}<extra></extra>",
                    )
                )

                # Actual measurements
                error_y = None
                if show_error_bars and name in output.measures:
                    error = np.std(output.measures[name], axis=0)
                    error_y = dict(type="data", array=error, visible=True)

                fig.add_trace(
                    go.Scatter(
                        x=output.steps,
                        y=measures,
                        mode="markers",
                        name="Model Results",
                        marker=dict(size=10),
                        error_y=error_y,
                        hovertemplate="Step: %{x}<br>Value: %{y:.4f}<extra></extra>",
                    )
                )

                # Add asymptote if requested
                if show_asymptote:
                    bound = 1 - output.params[name][2]
                    fig.add_trace(
                        go.Scatter(
                            x=[projection[0], projection[-1]],
                            y=[bound, bound],
                            mode="lines",
                            name=f"Asymptote: {bound:.4g}",
                            line=dict(dash="dash", width=2),
                            hovertemplate="Asymptote: %{y:.4f}<extra></extra>",
                        )
                    )

                fig.update_layout(
                    title=f"{name} Sufficiency",
                    xaxis_title="Steps",
                    yaxis_title=name,
                    xaxis_type="log",
                    width=700,
                    height=500,
                    hovermode="closest",
                )

                figures.append(fig)

        return figures

    def _plot_base_stats(
        self,
        output: Any,  # BaseStatsOutput
        log: bool = True,
        channel_limit: int | None = None,
        channel_index: Any = None,  # int | Iterable[int] | None
    ) -> Any:  # go.Figure
        """
        Plots the statistics as a set of histograms.

        Parameters
        ----------
        output : BaseStatsOutput
            The stats output object to plot
        log : bool, default True
            If True, plots the histograms on a logarithmic scale.
        channel_limit : int or None, default None
            The maximum number of channels to plot. If None, all channels are plotted.
        channel_index : int, Iterable[int] or None, default None
            The index or indices of the channels to plot. If None, all channels are plotted.

        Returns
        -------
        plotly.graph_objects.Figure
        """
        import math

        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        max_channels, ch_mask = output._get_channels(channel_limit, channel_index)
        factors = output.factors(exclude_constant=True)

        if not factors:
            return go.Figure()

        if max_channels == 1:
            # Single channel histogram
            num_metrics = len(factors)
            rows = math.ceil(num_metrics / 3)
            cols = min(num_metrics, 3)

            fig = make_subplots(
                rows=rows,
                cols=cols,
                subplot_titles=list(factors.keys()),
            )

            for idx, (metric_name, metric_values) in enumerate(factors.items()):
                row = idx // 3 + 1
                col = idx % 3 + 1

                fig.add_trace(
                    go.Histogram(
                        x=metric_values.flatten(),
                        nbinsx=20,
                        name=metric_name,
                        showlegend=False,
                        hovertemplate="Value: %{x}<br>Count: %{y}<extra></extra>",
                    ),
                    row=row,
                    col=col,
                )

                fig.update_xaxes(title_text="Values", row=row, col=col)
                fig.update_yaxes(
                    title_text="Counts", type="log" if log else "linear", row=row, col=col
                )

            fig.update_layout(height=300 * rows, width=300 * cols, title="Base Statistics Histograms")

        else:
            # Multi-channel histogram
            channelwise_metrics = [
                "mean",
                "std",
                "var",
                "skew",
                "zeros",
                "brightness",
                "contrast",
                "darkness",
                "entropy",
            ]
            data_keys = [key for key in factors if key in channelwise_metrics]

            num_metrics = len(data_keys)
            rows = math.ceil(num_metrics / 3)
            cols = min(num_metrics, 3)

            fig = make_subplots(
                rows=rows,
                cols=cols,
                subplot_titles=data_keys,
            )

            for idx, metric_name in enumerate(data_keys):
                row = idx // 3 + 1
                col = idx % 3 + 1

                # Reshape for channel-wise data
                data = factors[metric_name][ch_mask].reshape(-1, max_channels)

                for ch_idx in range(max_channels):
                    fig.add_trace(
                        go.Histogram(
                            x=data[:, ch_idx],
                            nbinsx=20,
                            name=f"Channel {ch_idx}",
                            opacity=0.7,
                            showlegend=(idx == 0),
                            legendgroup=f"ch{ch_idx}",
                            hovertemplate=f"Channel {ch_idx}<br>Value: %{{x}}<br>Count: %{{y}}<extra></extra>",
                        ),
                        row=row,
                        col=col,
                    )

                fig.update_xaxes(title_text="Values", row=row, col=col)
                fig.update_yaxes(
                    title_text="Counts", type="log" if log else "linear", row=row, col=col
                )

            fig.update_layout(
                height=300 * rows,
                width=300 * cols,
                title="Base Statistics Histograms (Multi-Channel)",
                barmode="overlay",
            )

        return fig

    def _plot_drift_mvdc(
        self,
        output: Any,  # DriftMVDCOutput
    ) -> Any:  # go.Figure
        """
        Render the roc_auc metric over the train/test data in relation to the threshold.

        Parameters
        ----------
        output : DriftMVDCOutput
            The drift MVDC output object to plot

        Returns
        -------
        plotly.graph_objects.Figure
        """
        import numpy as np
        import plotly.graph_objects as go

        resdf = output.to_dataframe()

        if resdf.shape[0] < 3:
            fig = go.Figure()
            fig.add_annotation(
                text="Insufficient data for drift detection plot",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
            )
            return fig

        fig = go.Figure()

        # Prepare data
        trndf = resdf[resdf["chunk"]["period"] == "reference"]
        tstdf = resdf[resdf["chunk"]["period"] == "analysis"]
        driftx = np.where(resdf["domain_classifier_auroc"]["alert"].values)[0]  # type: ignore

        # Threshold lines
        fig.add_trace(
            go.Scatter(
                x=resdf.index,
                y=resdf["domain_classifier_auroc"]["upper_threshold"],
                mode="lines",
                name="Threshold Upper",
                line=dict(dash="dash", color="red", width=2),
                hovertemplate="Index: %{x}<br>Threshold: %{y:.4f}<extra></extra>",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=resdf.index,
                y=resdf["domain_classifier_auroc"]["lower_threshold"],
                mode="lines",
                name="Threshold Lower",
                line=dict(dash="dash", color="red", width=2),
                hovertemplate="Index: %{x}<br>Threshold: %{y:.4f}<extra></extra>",
            )
        )

        # Train data
        fig.add_trace(
            go.Scatter(
                x=trndf.index,
                y=trndf["domain_classifier_auroc"]["value"],
                mode="lines",
                name="Train",
                line=dict(color="blue", width=2),
                hovertemplate="Index: %{x}<br>ROC AUC: %{y:.4f}<extra></extra>",
            )
        )

        # Test data
        fig.add_trace(
            go.Scatter(
                x=tstdf.index,
                y=tstdf["domain_classifier_auroc"]["value"],
                mode="lines",
                name="Test",
                line=dict(color="green", width=2),
                hovertemplate="Index: %{x}<br>ROC AUC: %{y:.4f}<extra></extra>",
            )
        )

        # Drift markers
        if len(driftx) > 0:
            fig.add_trace(
                go.Scatter(
                    x=resdf.index.values[driftx],
                    y=resdf["domain_classifier_auroc"]["value"].values[driftx],
                    mode="markers",
                    name="Drift",
                    marker=dict(symbol="diamond", size=10, color="magenta"),
                    hovertemplate="Drift at Index: %{x}<br>ROC AUC: %{y:.4f}<extra></extra>",
                )
            )

        fig.update_layout(
            title="Domain Classifier, Drift Detection",
            xaxis_title="Chunk Index",
            yaxis_title="ROC AUC",
            yaxis=dict(range=[0, 1.1]),
            width=900,
            height=500,
            hovermode="closest",
        )

        return fig
