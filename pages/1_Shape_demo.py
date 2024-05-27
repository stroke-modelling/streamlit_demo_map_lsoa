"""
Testing page for scatter fill polygons with holes.
"""
import streamlit as st
import shapely
from shapely import Polygon, MultiPolygon
import pandas as pd
import plotly.graph_objs as go

from utilities.maps import convert_shapely_polys_into_xy

# Define some shapes.
shape_a = Polygon(
    ((0, 0), (0, 10), (10, 10), (10, 0)),
    (  # Holes:
        ((2, 2), (2, 4), (4, 4)),
        ((6, 6), (6, 8), (8, 6))
    ))
shape_b = Polygon(
    ((15, 15), (15, 25), (25, 25), (25, 15)),
    (  # Holes:
        ((20, 22), (22, 22), (22, 24)),
    ))
shape_c = MultiPolygon((
    shapely.geometry.polygon.orient(Polygon(
        ((15, 0), (15, 5), (20, 5), (20, 0)),
        (  # Holes:
            ((17, 3), (17, 4), (18, 4)),
            ((16, 2), (16.5, 3), (16.5, 2))
        ))),
    shapely.geometry.polygon.orient(Polygon(
        ((21, 5), (21, 10), (25, 10), (25, 5)),
        (  # Holes:
            ((22, 6), (22, 8), (24, 6)),
        )))
))

# Fix hole order:
shape_a = shapely.geometry.polygon.orient(shape_a)
shape_b = shapely.geometry.polygon.orient(shape_b)

# Place into DataFrame:
df = pd.DataFrame([[0, shape_a], [1, shape_b], [2, shape_c]], columns=['id', 'geometry'])

st.write(df)

# Convert to x, y:
x, y = convert_shapely_polys_into_xy(df)
df['x'] = x
df['y'] = y

# Plot:
# Add each row of the dataframe separately.
# Scatter the edges of the polygons and use "fill" to colour
# within the lines.
fig = go.Figure()

for i in df.index:
    fig.add_trace(go.Scatter(
        x=df.loc[i, 'x'],
        y=df.loc[i, 'y'],
        mode='lines',
        fill="toself",
        )
        )

st.plotly_chart(fig)
