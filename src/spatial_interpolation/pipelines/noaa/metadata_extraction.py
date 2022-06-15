from lxml import html

import aiohttp
import asyncio
import requests
import time
import logging

import pandas as pd

from typing import Dict, Any, List, Union

logger = logging.getLogger("noaa.metadata_extraction")

def make_ndbc_metadata_df(
    buoy_locations_df: pd.DataFrame, 
    buoy_stdmet_index_df: pd.DataFrame = None
) -> pd.DataFrame:
    """
    Make a dataframe containing the information about all files available in the NOAA NDBC website
    with the corresponding information about the buoys associated with each file.

    Parameters:
    ------------
    buoy_locations_df: pd.DataFrame
        Dataframe containing the information about the buoys as ontained from `data_extraction.extract_buoy_locations`
    buoy_stdmet_index_df: pd.DataFrame
        Dataframe containing the information about the buoys as obtained from `data_extraction.get_buoy_stdmet_index_df`
        If not provided, the function will try to download the information from the NOAA website.
    """

    if buoy_stdmet_index_df is None:
        buoy_stdmet_index_df = get_buoy_stdmet_index_df()

    # Merge the dataframes
    ndbc_metadata_df = buoy_stdmet_index_df.join(
        buoy_locations_df.sort_values(by="year")
        .drop_duplicates(subset=["buoy_id"], keep="last")
        .drop(columns="year")
        .set_index("buoy_id"),
        on="buoy_id",
        how="inner",
    )

    return ndbc_metadata_df

def get_buoy_stdmet_index_df(stdmet_index_html:str=None) -> pd.DataFrame:
    """
    Get a dataframe listing all current historical stdmet files and their URLs
    from [NOAA's NDBC data index](https://www.ndbc.noaa.gov/data/stdmet/).

    Parameters
    ----------
    stdmet_index_html: str
        The html of the NDBC data index. If None, the html will be downloaded
    """
    if stdmet_index_html is None:
        #TODO: Make this retrieve the HTML will all files listed as obtained with wget
        # rather than only the first few files which requests returns
        stdmet_index_html = requests.get("https://www.ndbc.noaa.gov/data/stdmet/").text

    stdmet_index_df = (
        pd.read_html(stdmet_index_html)[0]
        .dropna(how="all", axis=1)
        .dropna(subset=["Last modified"], how="any", axis=0)
        .astype({"Last modified": "datetime64"})
        .rename(columns={"Name": "filename"})
        .assign(
            buoy_id=lambda df: df.filename.str.split("h").str[0],
            year=lambda df: df.filename.str.split(".txt")
            .str[0]
            .str.slice(-4)
            .astype(int),
        )
    )    

    return stdmet_index_df


# Buoy locations 
# ===============

def extract_buoy_locations(buoy_ids: List[str]) -> pd.DataFrame:
    """
    Extract the buoy locations from the NOAA website.
    and return a pandas dataframe.
    """
    start_time = time.time()
    buoys_info_list = get_buoys_info_async(buoy_ids)
    buoys_info_list = asyncio.get_event_loop().run_until_complete(buoys_info_list)
    end_time = time.time()

    print(f"Time taken: {end_time-start_time}")

    # Convert the buoy locations to a pandas dataframe
    buoys_info_dfs = [
        buoy_locations_dict_to_df(buoy_info) for buoy_info in buoys_info_list
    ]
    buoys_info_df = pd.concat(buoys_info_dfs, axis=0)
    # Write the buoy locations to a csv file
    return buoys_info_df

def buoy_locations_dict_to_df(
    buoy_locations_dict: Dict[str, Union[str, List[dict]]]
) -> pd.DataFrame:
    """
    Convert a dictionary of buoy locations to a pandas dataframe.
    """
    buoy_locations_df = (
        pd.DataFrame(buoy_locations_dict.get("locations"))
        .assign(
            buoy_id=buoy_locations_dict["buoy_id"],
            buoy_name=buoy_locations_dict.get("buoy_name"),
        )
        .dropna(how="all", axis="columns")
    )
    buoy_locations_df = buoy_locations_df[sorted(buoy_locations_df.columns)]
    return buoy_locations_df


async def get_buoys_info_async(buoy_ids: List[str]) -> List[dict]:
    """
    Get the location of a buoy from its buoy id.
    by querying the NCEI's (National Centers for Environmental Information)
    website about the NOAA Marine Environmental Buoy Database

    Source: https://www.ncei.noaa.gov/access/marine-environmental-buoy-database/#jump2map
    """

    base_url = "https://www.ncei.noaa.gov/access/marine-environmental-buoy-database/"
    logger.info(f"Getting buoy locations for {len(buoy_ids)} buoy ids from {base_url}")
    buoy_urls = {buoy_id: f"{base_url}/{buoy_id.lower()}.html" for buoy_id in buoy_ids}
    buoys_info_list = []

    async with aiohttp.ClientSession() as session:
        tasks = [
            asyncio.ensure_future(get_async(session, url)) for url in buoy_urls.values()
        ]

        txt_responses = await asyncio.gather(*tasks)

    for buoy_id, resp_txt in zip(buoy_urls.keys(), txt_responses):
        buoy_dict = extract_buoy_info_from_html(buoy_id, resp_txt)
        buoys_info_list.append(buoy_dict)

    return buoys_info_list


async def get_async(session, url) -> Dict[str, Any]:
    async with session.get(url) as resp:
        resp_text = await resp.text()
        return resp_text

def extract_buoy_info_from_html(buoy_id: str, buoy_html: str) -> dict:
    """
    Parse the location of a buoy from the html of the buoy's page.
    """

    # Get the name of the buoy
    resp_h3_tags = html.fromstring(buoy_html).xpath("//h3")
    buoy_name = (
        resp_h3_tags[0].text.split("-")[-1] if len(resp_h3_tags) > 0 else buoy_id
    )
    buoy_name = buoy_name.split("-")[-1].replace("\n", " ").strip()

    # Get latitude and longitude per year of the buoy
    stations_info_dfs: list = pd.read_html(buoy_html)
    if len(stations_info_dfs) == 0:
        locations_per_year = []
    else:
        stations_info_df = stations_info_dfs[0]
        stations_info_df = stations_info_df[stations_info_df.columns[0:2]]
        stations_info_df.columns = stations_info_df.columns.str.lower().str.strip()
        stations_info_df = stations_info_df.assign(
            latitude=lambda df: df.iloc[:, 0].str.split("|").str.get(0).astype(float),
            longitude=lambda df: df.iloc[:, 0].str.split("|").str.get(-1).astype(float),
        )
        locations_per_year = stations_info_df.iloc[:, 1:].to_dict(orient="records")

    buoy_locations_dict = {
        "buoy_id": buoy_id,
        "buoy_name": buoy_name,
        "locations": locations_per_year,
    }

    return buoy_locations_dict




if __name__ == "__main__":
    import time, json

    # Get the relevant buoy ids
    buoys_dict = json.load(open("references/south-atlantic-buoys.json"))
    buoy_ids = [id_ for val in buoys_dict.values() for id_ in val]

    # Get the buoy locations asynchronously
    start_time = time.time()
    buoys_info_list = get_buoys_info_async(buoy_ids)
    buoys_info_list = asyncio.get_event_loop().run_until_complete(buoys_info_list)
    end_time = time.time()

    print(f"Time taken: {end_time-start_time}")

    # Convert the buoy locations to a pandas dataframe
    buoys_info_dfs = [
        buoy_locations_dict_to_df(buoy_info) for buoy_info in buoys_info_list
    ]
    buoys_info_df = pd.concat(buoys_info_dfs, axis=0)
    # Write the buoy locations to a csv file
    buoys_info_df.to_csv("references/buoy_locations.csv", index=False)
