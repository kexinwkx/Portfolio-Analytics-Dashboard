import plotly.graph_objects as go
import pandas as pd


defplot_line(series: pd.Series,title: str,yaxis_title: str):
    fig = go.Figure()
if series is None or series.empty:
        fig.update_layout(title=f"{title} (No data)")
return fig

    fig.add_trace(
        go.Scatter(
            x=series.index,
            y=series.values,
            mode="lines"
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title=yaxis_title
    )

    return fig
