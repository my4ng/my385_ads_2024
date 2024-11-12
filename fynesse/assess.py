from .config import *

from . import access

"""These are the types of import we might expect in this file
import pandas
import bokeh
import seaborn
import matplotlib.pyplot as plt
import sklearn.decomposition as decomposition
import sklearn.feature_extraction"""

"""Place commands in this file to assess the data you have downloaded. How are missing values encoded, how are outliers encoded? What do columns represent, makes rure they are correctly labeled. How is the data indexed. Crete visualisation routines to assess the data (e.g. in bokeh). Ensure that date formats are correct and correctly timezoned."""

import osmnx as ox
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np
import numpy.polynomial.polynomial as polynomial

def plot_full_addr_buildings(name: str, latitude: float, longitude: float, size: float, gdf: pd.DataFrame):
    bbox = ox.utils_geo.bbox_from_point((latitude, longitude), size * 500)
    left, bottom, right, top = bbox

    graph = ox.graph_from_bbox(bbox, truncate_by_edge=True)
    _, edges = ox.graph_to_gdfs(graph)

    _, ax = plt.subplots(figsize=(8,8))

    edges.plot(ax=ax, linewidth=1, edgecolor="dimgray")
    color = gdf['joined'].notnull().map(lambda b: 'C0' if b else 'silver')
    gdf.plot(ax=ax, color=color)

    ax.set_title(f"{name}: Full Address Buildings")
    ax.set_xlim([left, right])
    ax.set_ylim([bottom, top])
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.ticklabel_format(useOffset=False)

    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    plt.tight_layout()

def plot_pp_transactions(name: str, latitude: float, longitude: float, size: float, df: pd.DataFrame):
    bbox = ox.utils_geo.bbox_from_point((latitude, longitude), size * 500)
    left, bottom, right, top = bbox
    
    graph = ox.graph_from_bbox(bbox, truncate_by_edge=True)
    _, edges = ox.graph_to_gdfs(graph)

    _, ax = plt.subplots(figsize=(8,8))

    edges.plot(ax=ax, linewidth=1, edgecolor="dimgray")
    ax.scatter(df['longitude'], df['latitude'], marker='.', s=50, alpha=0.25)
    ax.set_title(f"{name}: Transactions")

    ax.set_xlim([left, right])
    ax.set_ylim([bottom, top])
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")    
    ax.ticklabel_format(useOffset=False)

    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    plt.tight_layout()

def join_addr_pp_transaction(full_addr_df: pd.DataFrame, transaction_df: pd.DataFrame) -> pd.DataFrame:
    transaction_df['joined'] = transaction_df['primary_addressable_object_name'] + ' ' + transaction_df['street']
    joined_df = full_addr_df.merge(transaction_df, on='joined', how='left')
    joined_df = joined_df[['date_of_transfer', 'price', 'joined', 'area', 'geometry']]
    return joined_df

def plot_joined_addr_pp_transaction(name: str, latitude: float, longitude: float, size: float, joined_df: pd.DataFrame):
    bbox = ox.utils_geo.bbox_from_point((latitude, longitude), size * 500)
    left, bottom, right, top = bbox
    
    graph = ox.graph_from_bbox(bbox, truncate_by_edge=True)
    _, edges = ox.graph_to_gdfs(graph)

    _, ax = plt.subplots(figsize=(8,8))

    edges.plot(ax=ax, linewidth=1, edgecolor="dimgray")
    column = joined_df['price'].notnull().astype(int) + joined_df['joined'].notnull().astype(int)
    colors = ['silver', 'C0', 'C3']
    color = column.apply(lambda i: colors[i])
    joined_df.plot(ax=ax, color=color)
    ax.set_title(f"{name}: Matched transaction buildings")

    ax.set_xlim([left, right])
    ax.set_ylim([bottom, top])
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")    
    ax.ticklabel_format(useOffset=False)

    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    plt.tight_layout()

def visualise_area_price_corr(name: str, joined_df: pd.DataFrame):
    _, ax = plt.subplots()

    joined_df = joined_df[joined_df['price'].notnull()]
    r, p = stats.pearsonr(joined_df['area'], joined_df['price'])
    print(f"Pearson's r: {r:.3f}")

    b, k = polynomial.polyfit(joined_df['area'], joined_df['price'], 1)
    a = np.linspace(0, 1000, 100)

    ax.scatter(joined_df['area'], joined_df['price'], s=10)

    ax.set_title(f"{name}: Area/price correlation")
    ax.plot(a, b + k * a, lw=1, color='k')
    ax.ticklabel_format(useOffset=False, style='plain')
    ax.set_xlim(0, 1000)
    ax.set_ylim(0)
    ax.set_xlabel("area")
    ax.set_ylabel("price")

    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    plt.tight_layout()

def data():
    """Load the data from access and ensure missing values are correctly encoded as well as indices correct, column names informative, date and times correctly formatted. Return a structured data structure such as a data frame."""
    df = access.data()
    raise NotImplementedError

def query(data):
    """Request user input for some aspect of the data."""
    raise NotImplementedError

def view(data):
    """Provide a view of the data that allows the user to verify some aspect of its quality."""
    raise NotImplementedError

def labelled(data):
    """Provide a labelled set of data ready for supervised learning."""
    raise NotImplementedError
