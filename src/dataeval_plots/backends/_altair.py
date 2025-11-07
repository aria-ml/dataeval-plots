"""Altair plotting backend."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import altair as alt

    from dataeval.outputs import Output


class AltairBackend:
    """Altair implementation of plotting backend."""

    def plot(self, output: Output, **kwargs: Any) -> alt.Chart | alt.VConcatChart | alt.HConcatChart | list[Any]:
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
        Chart or list[Chart]
            Altair chart object(s)

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
    ) -> Any:  # alt.VConcatChart | alt.HConcatChart
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
        alt.VConcatChart or alt.HConcatChart
            Altair chart with image grid
        """
        import altair as alt
        import numpy as np
        import pandas as pd

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

        # Convert images to base64 for Altair
        import base64
        from io import BytesIO

        from PIL import Image

        image_data = []
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

            image_data.append({
                "index": idx,
                "row": idx // 3,
                "col": idx % 3,
                "image": f"data:image/png;base64,{img_str}"
            })

        df = pd.DataFrame(image_data)

        # Create the chart
        chart = alt.Chart(df).mark_image(
            width=200,
            height=200
        ).encode(
            url='image:N',
            x=alt.X('col:O', axis=None),
            y=alt.Y('row:O', axis=None)
        ).properties(
            title=f"Top {num_images} Uncovered Images"
        )

        return chart

    def _plot_balance(
        self,
        output: Any,  # BalanceOutput
        row_labels: Any = None,  # Sequence[Any] | NDArray[Any] | None
        col_labels: Any = None,  # Sequence[Any] | NDArray[Any] | None
        plot_classwise: bool = False,
    ) -> Any:  # alt.Chart
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
        alt.Chart
            Altair heatmap chart
        """
        import altair as alt
        import numpy as np
        import pandas as pd

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

        # Convert to long format for Altair
        rows, cols = data.shape
        heatmap_data = []
        for i in range(rows):
            for j in range(cols):
                if not np.isnan(data[i, j]):
                    heatmap_data.append({
                        "row": str(row_labels[i]),
                        "col": str(col_labels[j]),
                        "value": float(data[i, j])
                    })

        df = pd.DataFrame(heatmap_data)

        # Create heatmap
        chart = alt.Chart(df).mark_rect().encode(
            x=alt.X("col:N", title=xlabel, axis=alt.Axis(labelAngle=-45)),
            y=alt.Y("row:N", title=ylabel),
            color=alt.Color("value:Q", scale=alt.Scale(scheme="viridis", domain=[0, 1]),
                          title="Normalized Mutual Information"),
            tooltip=["row:N", "col:N", alt.Tooltip("value:Q", format=".2f")]
        ).properties(
            width=400,
            height=400,
            title=title
        )

        # Add text labels
        text = alt.Chart(df).mark_text(baseline="middle").encode(
            x=alt.X("col:N"),
            y=alt.Y("row:N"),
            text=alt.Text("value:Q", format=".2f"),
            color=alt.condition(
                alt.datum.value > 0.5,
                alt.value("white"),
                alt.value("black")
            )
        )

        return chart + text

    def _plot_diversity(
        self,
        output: Any,  # DiversityOutput
        row_labels: Any = None,  # Sequence[Any] | NDArray[Any] | None
        col_labels: Any = None,  # Sequence[Any] | NDArray[Any] | None
        plot_classwise: bool = False,
    ) -> Any:  # alt.Chart
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
        alt.Chart
            Altair chart (heatmap or bar chart)
        """
        from dataclasses import asdict

        import altair as alt
        import pandas as pd

        if plot_classwise:
            if row_labels is None:
                row_labels = output.class_names
            if col_labels is None:
                col_labels = output.factor_names

            # Create heatmap similar to balance
            data = output.classwise
            rows, cols = data.shape
            heatmap_data = []
            for i in range(rows):
                for j in range(cols):
                    heatmap_data.append({
                        "row": str(row_labels[i]),
                        "col": str(col_labels[j]),
                        "value": float(data[i, j])
                    })

            df = pd.DataFrame(heatmap_data)
            method = asdict(output.meta())["arguments"]["method"].title()

            chart = alt.Chart(df).mark_rect().encode(
                x=alt.X("col:N", title="Factors", axis=alt.Axis(labelAngle=-45)),
                y=alt.Y("row:N", title="Class"),
                color=alt.Color("value:Q", scale=alt.Scale(scheme="viridis", domain=[0, 1]),
                              title=f"Normalized {method} Index"),
                tooltip=["row:N", "col:N", alt.Tooltip("value:Q", format=".2f")]
            ).properties(
                width=400,
                height=400,
                title="Classwise Diversity"
            )

            text = alt.Chart(df).mark_text(baseline="middle").encode(
                x=alt.X("col:N"),
                y=alt.Y("row:N"),
                text=alt.Text("value:Q", format=".2f"),
                color=alt.condition(
                    alt.datum.value > 0.5,
                    alt.value("white"),
                    alt.value("black")
                )
            )

            return chart + text
        else:
            # Bar chart for diversity indices
            heat_labels = ["class_labels"] + list(output.factor_names)
            df = pd.DataFrame({
                "factor": heat_labels,
                "diversity": output.diversity_index
            })

            chart = alt.Chart(df).mark_bar().encode(
                x=alt.X("factor:N", title="Factors", axis=alt.Axis(labelAngle=-45)),
                y=alt.Y("diversity:Q", title="Diversity Index"),
                tooltip=["factor:N", alt.Tooltip("diversity:Q", format=".3f")]
            ).properties(
                width=500,
                height=400,
                title="Diversity Index by Factor"
            )

            return chart

    def _plot_sufficiency(
        self,
        output: Any,  # SufficiencyOutput
        class_names: Any = None,  # Sequence[str] | None
        show_error_bars: bool = True,
        show_asymptote: bool = True,
        reference_outputs: Any = None,  # Sequence[SufficiencyOutput] | SufficiencyOutput | None
    ) -> list[Any]:  # list[alt.Chart]
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
        list[alt.Chart]
            List of Altair charts for each measure
        """
        import altair as alt
        import numpy as np
        import pandas as pd

        # Extrapolation parameters
        last_X = output.steps[-1]
        geomshape = (0.01 * last_X, last_X * 4, len(output.steps))
        projection = np.geomspace(*geomshape).astype(np.int64)

        # Wrap reference
        if reference_outputs is None:
            reference_outputs = []
        elif not isinstance(reference_outputs, (list, tuple)):
            reference_outputs = [reference_outputs]

        charts = []

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

                    # Prepare data
                    plot_data = []

                    # Actual measurements
                    for step, value in zip(output.steps, values):
                        plot_data.append({
                            "step": int(step),
                            "value": float(value),
                            "type": "Model Results",
                            "series": f"{name}_{class_name}"
                        })

                    # Projection curve
                    proj_values = project_steps(output.params[name][i], projection)
                    for step, value in zip(projection, proj_values):
                        plot_data.append({
                            "step": int(step),
                            "value": float(value),
                            "type": "Potential Model Results",
                            "series": f"{name}_{class_name}"
                        })

                    df = pd.DataFrame(plot_data)

                    # Create chart
                    line = alt.Chart(df[df["type"] == "Potential Model Results"]).mark_line().encode(
                        x=alt.X("step:Q", scale=alt.Scale(type="log"), title="Steps"),
                        y=alt.Y("value:Q", title=name),
                        color=alt.Color("type:N", legend=alt.Legend(title="Series")),
                        tooltip=["step:Q", alt.Tooltip("value:Q", format=".4f")]
                    )

                    points = alt.Chart(df[df["type"] == "Model Results"]).mark_point(size=100).encode(
                        x=alt.X("step:Q", scale=alt.Scale(type="log")),
                        y=alt.Y("value:Q"),
                        color=alt.Color("type:N"),
                        tooltip=["step:Q", alt.Tooltip("value:Q", format=".4f")]
                    )

                    chart = (line + points).properties(
                        width=500,
                        height=400,
                        title=f"{name} Sufficiency - Class {class_name}"
                    )

                    # Add asymptote if requested
                    if show_asymptote:
                        bound = 1 - output.params[name][i][2]
                        asymptote_df = pd.DataFrame({
                            "step": [projection[0], projection[-1]],
                            "value": [bound, bound],
                            "type": [f"Asymptote: {bound:.4g}", f"Asymptote: {bound:.4g}"]
                        })
                        asymptote = alt.Chart(asymptote_df).mark_line(strokeDash=[5, 5]).encode(
                            x="step:Q",
                            y="value:Q",
                            color=alt.Color("type:N")
                        )
                        chart = chart + asymptote

                    charts.append(chart)
            else:
                # Single-class plotting
                plot_data = []

                # Actual measurements
                for step, value in zip(output.steps, measures):
                    plot_data.append({
                        "step": int(step),
                        "value": float(value),
                        "type": "Model Results"
                    })

                # Projection curve
                proj_values = project_steps(output.params[name], projection)
                for step, value in zip(projection, proj_values):
                    plot_data.append({
                        "step": int(step),
                        "value": float(value),
                        "type": "Potential Model Results"
                    })

                df = pd.DataFrame(plot_data)

                # Create chart
                line = alt.Chart(df[df["type"] == "Potential Model Results"]).mark_line().encode(
                    x=alt.X("step:Q", scale=alt.Scale(type="log"), title="Steps"),
                    y=alt.Y("value:Q", title=name),
                    color=alt.Color("type:N", legend=alt.Legend(title="Series")),
                    tooltip=["step:Q", alt.Tooltip("value:Q", format=".4f")]
                )

                points = alt.Chart(df[df["type"] == "Model Results"]).mark_point(size=100).encode(
                    x=alt.X("step:Q", scale=alt.Scale(type="log")),
                    y=alt.Y("value:Q"),
                    color=alt.Color("type:N"),
                    tooltip=["step:Q", alt.Tooltip("value:Q", format=".4f")]
                )

                chart = (line + points).properties(
                    width=500,
                    height=400,
                    title=f"{name} Sufficiency"
                )

                # Add asymptote if requested
                if show_asymptote:
                    bound = 1 - output.params[name][2]
                    asymptote_df = pd.DataFrame({
                        "step": [projection[0], projection[-1]],
                        "value": [bound, bound],
                        "type": [f"Asymptote: {bound:.4g}", f"Asymptote: {bound:.4g}"]
                    })
                    asymptote = alt.Chart(asymptote_df).mark_line(strokeDash=[5, 5]).encode(
                        x="step:Q",
                        y="value:Q",
                        color=alt.Color("type:N")
                    )
                    chart = chart + asymptote

                charts.append(chart)

        return charts

    def _plot_base_stats(
        self,
        output: Any,  # BaseStatsOutput
        log: bool = True,
        channel_limit: int | None = None,
        channel_index: Any = None,  # int | Iterable[int] | None
    ) -> Any:  # alt.VConcatChart | alt.HConcatChart
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
        alt.VConcatChart or alt.HConcatChart
            Altair chart with histogram grid
        """
        import altair as alt
        import pandas as pd

        max_channels, ch_mask = output._get_channels(channel_limit, channel_index)
        factors = output.factors(exclude_constant=True)

        if not factors:
            # Return empty chart
            return alt.Chart(pd.DataFrame()).mark_point()

        charts = []

        if max_channels == 1:
            # Single channel histogram
            for metric_name, metric_values in factors.items():
                df = pd.DataFrame({
                    "value": metric_values.flatten(),
                    "metric": metric_name
                })

                chart = alt.Chart(df).mark_bar().encode(
                    x=alt.X("value:Q", bin=alt.Bin(maxbins=20), title="Values"),
                    y=alt.Y("count()", scale=alt.Scale(type="log" if log else "linear"), title="Counts"),
                    tooltip=["count()"]
                ).properties(
                    width=250,
                    height=200,
                    title=metric_name
                )
                charts.append(chart)
        else:
            # Multi-channel histogram
            channelwise_metrics = ["mean", "std", "var", "skew", "zeros", "brightness", "contrast", "darkness", "entropy"]

            for metric_name, metric_values in factors.items():
                if metric_name in channelwise_metrics:
                    # Reshape for channel-wise data
                    data = metric_values[ch_mask].reshape(-1, max_channels)

                    plot_data = []
                    for ch_idx in range(max_channels):
                        for val in data[:, ch_idx]:
                            plot_data.append({
                                "value": float(val),
                                "channel": f"Channel {ch_idx}",
                                "metric": metric_name
                            })

                    df = pd.DataFrame(plot_data)

                    chart = alt.Chart(df).mark_bar(opacity=0.7).encode(
                        x=alt.X("value:Q", bin=alt.Bin(maxbins=20), title="Values"),
                        y=alt.Y("count()", scale=alt.Scale(type="log" if log else "linear"), title="Counts"),
                        color=alt.Color("channel:N", legend=alt.Legend(title="Channel")),
                        tooltip=["channel:N", "count()"]
                    ).properties(
                        width=250,
                        height=200,
                        title=metric_name
                    )
                    charts.append(chart)
                else:
                    # Non-channelwise metric
                    df = pd.DataFrame({
                        "value": metric_values.flatten(),
                        "metric": metric_name
                    })

                    chart = alt.Chart(df).mark_bar().encode(
                        x=alt.X("value:Q", bin=alt.Bin(maxbins=20), title="Values"),
                        y=alt.Y("count()", scale=alt.Scale(type="log" if log else "linear"), title="Counts"),
                        tooltip=["count()"]
                    ).properties(
                        width=250,
                        height=200,
                        title=metric_name
                    )
                    charts.append(chart)

        # Arrange in grid (3 columns)
        rows = []
        for i in range(0, len(charts), 3):
            row_charts = charts[i:i+3]
            if len(row_charts) == 1:
                rows.append(row_charts[0])
            else:
                rows.append(alt.hconcat(*row_charts))

        if len(rows) == 1:
            return rows[0]
        else:
            return alt.vconcat(*rows)

    def _plot_drift_mvdc(
        self,
        output: Any,  # DriftMVDCOutput
    ) -> Any:  # alt.Chart
        """
        Render the roc_auc metric over the train/test data in relation to the threshold.

        Parameters
        ----------
        output : DriftMVDCOutput
            The drift MVDC output object to plot

        Returns
        -------
        alt.Chart
            Altair line chart with drift detection
        """
        import altair as alt
        import numpy as np
        import pandas as pd

        resdf = output.to_dataframe()

        if resdf.shape[0] < 3:
            # Not enough data to plot
            return alt.Chart(pd.DataFrame()).mark_point().properties(
                title="Insufficient data for drift detection plot"
            )

        # Prepare data for plotting
        plot_data = []

        for idx, row in resdf.iterrows():
            period = row["chunk"]["period"]
            value = row["domain_classifier_auroc"]["value"]
            upper_thr = row["domain_classifier_auroc"]["upper_threshold"]
            lower_thr = row["domain_classifier_auroc"]["lower_threshold"]
            alert = row["domain_classifier_auroc"]["alert"]

            plot_data.append({
                "index": idx,
                "value": float(value),
                "upper_threshold": float(upper_thr),
                "lower_threshold": float(lower_thr),
                "period": "train" if period == "reference" else "test",
                "alert": bool(alert)
            })

        df = pd.DataFrame(plot_data)

        # Create base chart
        base = alt.Chart(df).encode(
            x=alt.X("index:Q", title="Chunk Index")
        )

        # Threshold lines
        upper_line = base.mark_line(strokeDash=[5, 5], color="red").encode(
            y=alt.Y("upper_threshold:Q", title="ROC AUC")
        )

        lower_line = base.mark_line(strokeDash=[5, 5], color="red").encode(
            y="lower_threshold:Q"
        )

        # Train and test lines
        train_df = df[df["period"] == "train"]
        test_df = df[df["period"] == "test"]

        train_line = alt.Chart(train_df).mark_line(color="blue").encode(
            x="index:Q",
            y=alt.Y("value:Q", scale=alt.Scale(domain=[0, 1.1])),
            tooltip=["index:Q", alt.Tooltip("value:Q", format=".4f")]
        )

        test_line = alt.Chart(test_df).mark_line(color="green").encode(
            x="index:Q",
            y="value:Q",
            tooltip=["index:Q", alt.Tooltip("value:Q", format=".4f")]
        )

        # Drift markers
        drift_df = df[df["alert"]]
        drift_points = alt.Chart(drift_df).mark_point(
            shape="diamond",
            size=100,
            color="magenta",
            filled=True
        ).encode(
            x="index:Q",
            y="value:Q",
            tooltip=["index:Q", alt.Tooltip("value:Q", format=".4f")]
        )

        # Combine all layers
        chart = (upper_line + lower_line + train_line + test_line + drift_points).properties(
            width=600,
            height=400,
            title="Domain Classifier, Drift Detection"
        )

        return chart
