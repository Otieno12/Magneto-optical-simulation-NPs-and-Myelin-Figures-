import plotly.express as px
import pandas as pd
import numpy as np


df1 = df[df['filename'] == ''

if not df1.empty:
    df_plot = df1.head(100).copy()

    # Create histogram bins between 0 and 50 µm
    bins = np.linspace(0, 30, 20)
    df_plot['bin'] = pd.cut(df_plot['diameter_um'], bins=bins, include_lowest=True)

    # Compute per-bin stats
    summary = df_plot.groupby('bin')['diameter_um'].agg(['count', 'mean', 'std']).reset_index()
    summary['bin_center'] = summary['bin'].apply(lambda x: x.mid)
    summary['std'] = summary['std'].fillna(0)

    
    fig = px.bar(
        summary,
        x='bin_center',
        y='count',
        error_y='std',
        labels={'bin_center': 'Diameter (μm)', 'count': 'Count ± Std Dev'},
        title='Diameter Distribution at 0 T'
    )

    fig.update_layout(
        xaxis=dict(range=[0, 50], tickmode='linear', dtick=5),
        template='plotly_white'
    )

    fig.show()
else:
    print("No data found for the specified filename.")
