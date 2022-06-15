from ml_collections import config_dict
# from pandas import IndexSlice as idx
import pandas as pd
from shapely import geometry

eval_sets = config_dict.ConfigDict()

eval_sets.time_range = pd.date_range("2011-01-01", "2022-01-01", freq="1H").to_list()
eval_sets.entire_area = geometry.box(*(-98.251934,12.282308,-45,35.55)) 
area_1 = geometry.box(*(-97.806644,18.930645,-80,30.366655))
area_2 = geometry.box(*(-82.836914,25.799891,-68.115234,35.55))
# areas are defined as bbox in the form of (min_lon, min_lat, max_lon, max_lat)

## EVALUATION SET 1
set1 = config_dict.ConfigDict()
set1.name = "set1"
set1.area = area_1
set1.partial = dict(a={},b={},c={})

# buoy locations that will be completely omitted from the training set in this evaluation set
set1.locations_full = ["42040"]
set1.eval = pd.MultiIndex.from_product([set1.locations_full, eval_sets.time_range]).to_list()
# specic buoy locations that will be partially omitted from the training set at specified time periods
set1.partial.a.locations = ["42019"]
set1.partial.a.time = dict(start="2021-03-01 01:00", end="2021-03-20 01:00")
patial_time_1 = pd.date_range(set1.partial.a.time.start, set1.partial.a.time.end, freq="1H")
set1.eval = set1.eval + pd.MultiIndex.from_product([set1.partial.a.locations,patial_time_1]).to_list()

set1.partial.b.locations = ["42001"]
set1.partial.b.time = dict(start="2016-11-25 00:00", end="2017-01-01 01:00")
patial_time_2 = pd.date_range(set1.partial.b.time.start, set1.partial.b.time.end, freq="1H")
set1.eval = set1.eval + pd.MultiIndex.from_product([set1.partial.b.locations,patial_time_2]).to_list()

set1.partial.c.locations = ["42019"]
set1.partial.c.time = dict(start="2020-05-05 01:00", end="2020-05-30 23:00")
patial_time_4 = pd.date_range(set1.partial.c.time.start, set1.partial.c.time.end, freq="1H")
set1.eval = set1.eval + pd.MultiIndex.from_product([set1.partial.c.locations,patial_time_4]).to_list()



## EVALUATION SET 2
set2 = config_dict.ConfigDict()
set2.name = "set2"
set2.area = area_1.union(area_2)
set2.partial = dict(a={},b={},c={},d={})

# buoy locations that will be completely omitted from the training set in this evaluation set
set2.locations_full = ["42040", "41036"]
set2.eval = pd.MultiIndex.from_product([set2.locations_full, eval_sets.time_range]).to_list()
# specic buoy locations that will be partially omitted from the training set at specified time periods
set2.partial.a.locations = ["41001","41004"]
set2.partial.a.time = dict(start="2020-10-10 01:00", end="2020-10-30 01:00")
patial_time_1 = pd.date_range(set2.partial.a.time.start, set2.partial.a.time.end, freq="1H")
set2.eval = set2.eval + pd.MultiIndex.from_product([set2.partial.a.locations,patial_time_1]).to_list()

set2.partial.b.locations = ["42001","41012"]
set2.partial.b.time = dict(start="2016-07-01 01:00", end="2016-07-15 01:00")
patial_time_2 = pd.date_range(set2.partial.b.time.start, set2.partial.b.time.end, freq="1H")
set2.eval = set2.eval + pd.MultiIndex.from_product([set2.partial.b.locations,patial_time_2]).to_list()

set2.partial.c.locations = ["42001","42002", "42003"]
set2.partial.c.time = dict(start="2011-03-01 01:00", end="2011-03-10 23:00")
patial_time_3 = pd.date_range(set2.partial.c.time.start, set2.partial.c.time.end, freq="1H")
set2.eval = set2.eval + pd.MultiIndex.from_product([set2.partial.c.locations,patial_time_3]).to_list()

set2.partial.d.locations = ["41047","42056","42035","41035","42012"]
set2.partial.d.time = dict(start="2015-06-10 01:00", end="2015-06-13 23:00")
patial_time_4 = pd.date_range(set2.partial.d.time.start, set2.partial.d.time.end, freq="1H")
set2.eval = set2.eval + pd.MultiIndex.from_product([set2.partial.d.locations,patial_time_4]).to_list()

## EVALUATION SET 3
set3 = config_dict.ConfigDict()
set3.name = "set3"
set3.area = eval_sets.entire_area
set3.partial = dict(**set2.partial.to_dict().copy(), e={}, f={})
set3.eval = set2.eval

# this set includes the same full/partial locations and time periods as set 1 but
# in the entire area of the dataset and includes some additional eval locations and time periods
set3.locations_full = set2.locations_full + ["41041", "42065"]
set3.eval + pd.MultiIndex.from_product([["41041"], eval_sets.time_range]).to_list()

set3.partial.e.locations = ["41046","42058"]
set3.partial.e.time = dict(start="2018-09-10 01:00", end="2018-09-30 23:00")
partial_time_1 = pd.date_range(set3.partial.e.time.start, set3.partial.e.time.end, freq="1H")
set3.eval = set3.eval + pd.MultiIndex.from_product([set3.partial.e.locations,partial_time_1]).to_list()

# Hurricane Dorian hit the caribbean and coast of Florida on August 2019
set3.partial.f.locations = ["41043","41047","41025","41010", "41001", "42065"]
set3.partial.f.time = dict(start="2019-10-20 01:00", end="2019-09-12 23:00")
partial_time_2 = pd.date_range(set3.partial.f.time.start, set3.partial.f.time.end, freq="1H")
set3.eval = set3.eval + pd.MultiIndex.from_product([set3.partial.f.locations,partial_time_2]).to_list()

## set 4
# set4 = set1.copy_and_resolve_references()
# set4.name = "set4"
# set4.partial.a.locations = ["42019"]
# set4.partial.a.time = dict(start="2020-05-05 01:00", end="2020-05-30 23:00")
# patial_time_1 = pd.date_range(set4.partial.a.time.start, set4.partial.a.time.end, freq="1H")
# set4.eval = set4.eval + pd.MultiIndex.from_product([set4.partial.a.locations,patial_time_1]).to_list()


eval_sets.ndbc = dict(
    set1=set1.to_dict(),
    set2=set2.to_dict(),
    set3=set3.to_dict(),
    # set4=set4.to_dict()
)

def get_eval_sets(as_conf=False):   
    return eval_sets