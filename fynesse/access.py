from .config import *

"""These are the types of import we might expect in this file
import httplib2
import oauth2
import tables
import mongodb
import sqlite"""

import os
import requests
import zipfile
import pymysql
import csv
import sqlalchemy

import osmnx as ox
import pandas as pd

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
                               )
        print(f"Connection established!")
    except Exception as e:
        print(f"Error connecting to the MariaDB Server: {e}")
    return conn

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
    """
    Count Points of Interest (POIs) near a given pair of coordinates within a specified distance.
    Args:
        latitude (float): Latitude of the location.
        longitude (float): Longitude of the location.
        features (dict): A dictionary of interested features and its matching tag-value list.
        distance_km (float): The distance around the location in kilometers. Default is 1 km.
    Returns:
        pd: A pandas DataFrame that holds the results of the
    """
    tags = { tag[0]: True for feature in features.values() for tag in feature }
    pois = ox.features_from_point((latitude, longitude), tags, distance_km * 1000.)
    return pd.DataFrame(pois)

def count_pois_near_coordinates(latitude: float, longitude: float, features: dict, distance_km: float = 1.0) -> dict:
    """
    Count Points of Interest (POIs) near a given pair of coordinates within a specified distance.
    Args:
        latitude (float): Latitude of the location.
        longitude (float): Longitude of the location.
        features (dict): A dictionary of interested features and its matching tag-value list.
        distance_km (float): The distance around the location in kilometers. Default is 1 km.
    Returns:
        dict: A dictionary where keys are the OSM tags and values are the counts of POIs for each tag.
    """
    pois = pois_near_coordinates_df(latitude, longitude, features, distance_km)
    poi_counts = {}

    for name, feature in features.items():
        pred = False
        for tag in feature:
            if tag[1] is True:
                pred |= (pois[tag[0]].notna())
            else:
                for value in tag[1]:
                    pred |= (pois[tag[0]] == value)
        poi_counts[name] = pred.sum()
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

def data():
    """Read the data from the web or local file, returning structured format such as a data frame"""
    raise NotImplementedError

