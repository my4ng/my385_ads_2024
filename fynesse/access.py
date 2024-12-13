from .config import *

"""These are the types of import we might expect in this file
import httplib2
import oauth2
import tables
import mongodb
import sqlite"""

import os
import subprocess
import math
import requests
import zipfile
import pymysql
from pymysql.constants import CLIENT
import csv
import sqlalchemy

import osmnx as ox
from osmnx._errors import InsufficientResponseError
import pandas as pd
from pandarallel import pandarallel
import numpy as np

"""Place commands in this file to access the data electronically. Don't remove any missing values, or deal with outliers. Make sure you have legalities correct, both intellectual property and personal data privacy rights. Beyond the legal side also think about the ethical issues around this data. """

def download_price_paid_data(data_dir_path, year_from, year_to):
    # Base URL where the dataset is stored
    base_url = "http://prod.publicdata.landregistry.gov.uk.s3-website-eu-west-1.amazonaws.com"
    """Download UK house price data for given year range"""
    # File name with placeholders
    file_name = "/pp-<year>-part<part>.csv"
    if not os.path.exists(data_dir_path):
        os.makedirs(data_dir_path)
    for year in range(year_from, (year_to+1)):
        print (f"Downloading data for year: {year}")
        for part in range(1,3):
            url = base_url + file_name.replace("<year>", str(year)).replace("<part>", str(part))
            response = requests.get(url)
            if response.status_code == 200:
                with open(data_dir_path + file_name.replace("<year>", str(year)).replace("<part>", str(part)), "wb") as file:
                    file.write(response.content)

def download_postcode_data(data_dir_path):
    url = "https://www.getthedata.com/downloads/open_postcode_geo.csv.zip"
    """Download UK postcode data"""
    if not os.path.exists(data_dir_path):
        os.makedirs(data_dir_path)
    response = requests.get(url)
    if response.status_code == 200:
        zip_path = data_dir_path + "/open_postcode_geo.csv.zip"
        with open(zip_path, "wb") as file:
            file.write(response.content)
        with zipfile.ZipFile(zip_path, 'r') as zip:
            zip.extract("open_postcode_geo.csv", data_dir_path)
        os.remove(zip_path)
# This file accesses the data

def create_connection(user, password, host, database, port=3306):
    """ Create a database connection to the MariaDB database
        specified by the host url and database name.
    :param user: username
    :param password: password
    :param host: host url
    :param database: database name
    :param port: port number
    :return: Connection object or None
    """
    conn = None
    try:
        conn = pymysql.connect(user=user,
                               passwd=password,
                               host=host,
                               port=port,
                               local_infile=1,
                               db=database,
                               client_flag=CLIENT.MULTI_STATEMENTS
                               )
        print(f"Connection established!")
    except Exception as e:
        print(f"Error connecting to the MariaDB Server: {e}")
    return conn

def create_engine(username, password, host, database="ads_2024", port=3306):
    db_url = f"mariadb+pymysql://{username}:{password}@{host}:{port}/{database}?local_infile=1"
    engine = sqlalchemy.create_engine(db_url)
    return engine

def housing_upload_join_data(data_dir_path, conn, year):
    with conn.cursor() as cur:
        start_date = str(year) + "-01-01"
        end_date = str(year) + "-12-31"

        print('Selecting data for year: ' + str(year))
        cur.execute(f"""
                    SELECT pp.*, po.country, po.latitude, po.longitude FROM (SELECT price, 
                    date_of_transfer, postcode, property_type, new_build_flag, tenure_type, 
                    primary_addressable_object_name, secondary_addressable_object_name, 
                    street, locality, town_city, district, county FROM pp_data 
                    WHERE date_of_transfer BETWEEN "{start_date}" AND "{end_date}") AS pp 
                    INNER JOIN postcode_data AS po ON pp.postcode = po.postcode
                    """)
        rows = cur.fetchall()

        csv_file_path = data_dir_path + '/output_file.csv'

        # Write the rows to the CSV file
        with open(csv_file_path, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            # Write the data rows
            csv_writer.writerows(rows)
        print('Storing data for year: ' + str(year))
        cur.execute(f"LOAD DATA LOCAL INFILE '" + csv_file_path + "' INTO TABLE `prices_coordinates_data` FIELDS TERMINATED BY ',' OPTIONALLY ENCLOSED by '\"' LINES STARTING BY '' TERMINATED BY '\n';")

        print('Data stored for year: ' + str(year))
        os.remove(csv_file_path)
    conn.commit()

def pois_near_coordinates_df(latitude: float, longitude: float, features: dict, distance_km: float = 1.0) -> pd.DataFrame:
    tags = {}
    for tag in features.values():
        for (key, value) in tag:
            if value is True:
                tags[key] = True
            else:
                val = []
                for v in value:
                    if isinstance(v, tuple):
                        val.append(v[0])
                    else:
                        val.append(v)
                if key in tags:
                    if tags[key] is True:
                        pass
                    else:
                        tags[key].extend(val)
                else:
                    tags[key] = val
    try:
        pois = ox.features_from_point((latitude, longitude), tags, distance_km * 500)
        pois = pd.DataFrame(pois)
    # handle no features
    except InsufficientResponseError:
        pois = pd.DataFrame([])
    return pois

def count_pois_near_coordinates(latitude: float, longitude: float, features: dict, distance_km: float = 1.0) -> dict:
    pois = pois_near_coordinates_df(latitude, longitude, features, distance_km)
    poi_counts = {}

    for name, feature in features.items():
        pred = pd.Series(False, index=pois.index)
        for tag in feature:
            if tag[0] in pois.columns:
                if tag[1] is True:
                    pred |= (pois[tag[0]].notna())
                else:
                    for value in tag[1]:
                        if isinstance(value, tuple):
                            if value[1] in pois.columns:
                                pred |= (pois[tag[0]] == value[0]) & (pois[value[1]].isin(value[2]))
                        else:
                            pred |= (pois[tag[0]] == value)
        poi_counts[name] = pred.to_numpy().sum(dtype=np.int32)
    return poi_counts

def full_addr_buildings_gdf(latitude: float, longitude: float, size: float) -> pd.DataFrame:
    bbox = ox.utils_geo.bbox_from_point((latitude, longitude), size * 500)
    gdf = ox.features_from_bbox(bbox, { 'building': True })
    gdf['joined'] = (gdf['addr:housenumber'].fillna(gdf['addr:housename']) + ' ' 
                            + gdf['addr:street']).fillna(gdf['name']).str.upper()
    gdf['area'] = gdf['geometry'].to_crs(epsg=27700).area
    return gdf

def pp_transactions_df(latitude: float, longitude: float, size: float, conn: pymysql.Connection, 
                       start_date: str, end_date: str | None = None) -> pd.DataFrame:
    bbox = ox.utils_geo.bbox_from_point((latitude, longitude), size * 500)
    left, bottom, right, top = bbox

    if end_date is None:
        query = f"""
        SELECT * FROM `prices_coordinates_data` 
        WHERE date_of_transfer >= '{start_date}'
        AND latitude BETWEEN {bottom} AND {top}
        AND longitude BETWEEN {left} AND {right}
        """
    else:
        query = f"""
        SELECT * FROM `prices_coordinates_data` 
        WHERE date_of_transfer BETWEEN '{start_date}' AND '{end_date}'
        AND latitude BETWEEN {bottom} AND {top}
        AND longitude BETWEEN {left} AND {right}
        """

    return pd.read_sql(query, conn)

def download_2021_census_oa_data(ts_no: str, data_dir_path: str="data"):
    url = f"https://www.nomisweb.co.uk/output/census/2021/census2021-{ts_no}.zip"
    name = f"census2021-{ts_no}-oa.csv"
    if not os.path.exists(data_dir_path):
        os.makedirs(data_dir_path)
    if os.path.isfile(data_dir_path + "/" + name):
        return
    response = requests.get(url)
    if response.status_code == 200:
        zip_path = data_dir_path + f"/census2021-{ts_no}.zip"
        with open(zip_path, "wb") as file:
            file.write(response.content)
        with zipfile.ZipFile(zip_path, 'r') as zip:
            zip.extract(name, data_dir_path)
        os.remove(zip_path)

def download_2011_census_oa_data(ks_no: str, data_dir_path: str="data"):
    url = f"https://www.nomisweb.co.uk/output/census/2011/{ks_no}_2011_oa.zip"
    name = f"{ks_no}data.csv"
    if not os.path.exists(data_dir_path):
        os.makedirs(data_dir_path)
    if os.path.isfile(data_dir_path + "/" + name):
        return
    response = requests.get(url)
    if response.status_code == 200:
        zip_path = data_dir_path + f"/{ks_no}_2011_oa.zip"
        with open(zip_path, "wb") as file:
            file.write(response.content)
        with zipfile.ZipFile(zip_path, 'r') as zip:
            with open(data_dir_path + "/" + name, 'wb') as f:
                f.write(zip.read("ks101ew_2011oa/" + name.upper()))
        os.remove(zip_path)

def download_oa_geo(data_dir_path: str="data"):
    url = "https://open-geography-portalx-ons.hub.arcgis.com/api/download/v1/items/6beafcfd9b9c4c9993a06b6b199d7e6d/csv?layers=0"
    path = data_dir_path + "/oa_geo.csv"
    if not os.path.exists(data_dir_path):
        os.makedirs(data_dir_path)
    if os.path.isfile(path):
        return
    response = requests.get(url)
    if response.status_code == 200:
        with open(path, "wb") as file:
            file.write(response.content)
    

def download_eng_pbf(data_dir_path: str="data"):
    url = "https://download.openstreetmap.fr/extracts/europe/united_kingdom/england-latest.osm.pbf"
    path = data_dir_path + "/england-latest.osm.pbf"
    if os.path.isfile(path):
        return
    # broken, replace great-britain with united-kingdom
    # fp = get_data("england", directory=data_dir_path)
    # print("Data was downloaded to:", fp)
    if not os.path.exists(data_dir_path):
        os.makedirs(data_dir_path)
    response = requests.get(url)
    if response.status_code == 200:
        with open(path, "wb") as file:
            file.write(response.content)

def features_from_point_local(latitude: float, longitude: float, tags: dict, size: float=1.0, data_dir_path: str="data"):
    path = data_dir_path + "/england-latest.osm.pbf"
    tmp_path = data_dir_path + "/temp" + str(np.random.randint(1 << 31)) +".osm"
    x = size / (111.320 * math.cos(latitude * math.pi / 180))
    y = size / 110.574
    minx = longitude - x / 2
    miny = latitude - y / 2
    maxx = longitude + x / 2
    maxy = latitude + y / 2
    subprocess.run(["osmium", 
                    "extract", 
                    f"--bbox={minx},{miny},{maxx},{maxy}", 
                    "-S", "relations=false",
                    path, 
                    f"--output={tmp_path}"])
    df = ox.features_from_xml(tmp_path, tags=tags)
    os.remove(tmp_path)
    return df

def get_pc_student_df(engine: sqlalchemy.Engine) -> pd.DataFrame:
    with engine.connect() as conn:
        query = """
        SELECT OA21CD, L15 / total AS pc_student, latitude, longitude FROM `nssec_geo_data`
        """
        return pd.read_sql(query, conn, index_col='OA21CD')

def get_subset_df(df: pd.DataFrame, col:str, n: int=100, q: list[int] | None=None, random_state=42) -> pd.DataFrame:
    if q is None:
        sample = df.sample(n, random_state=random_state)
    else:
        q = np.array(q)
        q = q / q.sum()
        qcut = pd.qcut(df[col], len(q), labels=False)
        sample = df.groupby(qcut, group_keys=False).apply(lambda g: g.sample(int(n * q[g.name]), random_state=random_state))
    return sample

def get_pois_df(df: pd.DataFrame, features: dict, dist: float=1) -> pd.DataFrame:
    def count_inner(row):
        return count_pois_near_coordinates(
            row['latitude'], 
            row['longitude'], 
            features,
            dist)
    pois_df = df.parallel_apply(count_inner, axis=1, result_type='expand')
    pois_df.columns = features.keys()
    return pois_df

def get_hh_size_df(engine: sqlalchemy.Engine) -> pd.DataFrame:
    with engine.connect() as conn:
        query = """
        SELECT *
        FROM `hh_size_geo_data`
        """
        df = pd.read_sql(query, conn, index_col='OA21CD')
        return df

def get_pop_change_df(engine: sqlalchemy.Engine) -> pd.DataFrame:
    with engine.connect() as conn:
        query = """
        SELECT * FROM `pop_data`
        """
        df = pd.read_sql(query, conn, index_col='OA')
        df.insert(0, 'pop_change', (df['pop_21'] - df['pop_11']) / df['pop_11'])
        df.drop(['pop_11', 'pop_21'], axis=1, inplace=True)
        return df

def get_grouped_subset_df(df: pd.DataFrame, col:str, n: int=5, random_state=42) -> pd.DataFrame:
    samples = df[col].sample(n, random_state=random_state)
    samples = pd.concat([df.loc[df[col] == s] for s in samples])
    print(samples)
    return samples