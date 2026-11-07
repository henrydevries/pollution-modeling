import numpy as np
import pandas as pd

def partmissing(sensorseries):
    mask = ~np.isnan(sensorseries)
    if not mask.any(): return 100
    start_idx = np.argmax(mask)
    end_idx = len(sensorseries) - np.argmax(mask[::-1]) - 1
    series = sensorseries[start_idx:end_idx+1]
    return np.isnan(series).sum()/len(series) * 100

def partoutlier(sensorseries):
    mask = ~np.isnan(sensorseries)
    if not mask.any(): return None
    start_idx = np.argmax(mask)
    end_idx = len(sensorseries) - np.argmax(mask[::-1]) - 1
    series = sensorseries[start_idx:end_idx+1]

    nonmissing = np.sum(~np.isnan(series))
    outlier_count = np.sum(series > 500)
    return outlier_count / nonmissing * 100

def get_start_percentage(sensorseries):
    mask = ~np.isnan(sensorseries)
    start_idx = np.argmax(mask)
    return (start_idx / len(sensorseries)) * 100

def longest_high_run(sensorseries, threshold=500):
    mask = ~np.isnan(sensorseries)
    if not mask.any(): return None
    start_idx = np.argmax(mask)
    end_idx = len(sensorseries) - np.argmax(mask[::-1]) - 1
    series = sensorseries[start_idx:end_idx+1]

    high = series > threshold

    max_run = 0
    current = 0
    for x in high:
        if x:
            current += 1
            max_run = max(max_run, current)
        else:
            current = 0
    return max_run

def dataloader():
    raw_pollution = ...
    sensors = ...
    
    # Step 1: throw away completely useless sensors
    # Keep only viable sensors
    missing_rates = raw_pollution.apply(partmissing, axis=1)
    sensors['missing_rate'] = sensors.index.map(missing_rates)
    sensors = sensors[sensors['missing_rate'] < 50]
    raw_pollution = raw_pollution.loc[sensors.index]

    # Removing extreme outlier cases
    outlier_rates = raw_pollution.apply(partoutlier, axis=1)
    sensors['outlier_rate'] = sensors.index.map(outlier_rates)
    sensors = sensors[sensors['outlier_rate'] < 25]
    raw_pollution = raw_pollution.loc[sensors.index]

    # Removing consistent outlier sensors
    runs = raw_pollution.apply(longest_high_run, axis=1)
    sensors["max_high_run"] = sensors.index.map(runs)
    sensors = sensors[sensors.max_high_run < 14]
    raw_pollution = raw_pollution.loc[sensors.index]

    # Step 2: retrieving data that is viable for evaluating, but not usefull for training, not used in practice
    start_percentages = raw_pollution.apply(get_start_percentage, axis=1)
    sensors['start_at_percentage'] = raw_pollution.index.map(start_percentages)
    sensors_extra = sensors[sensors['start_at_percentage'] >= 50]
    sensors = sensors[sensors['start_at_percentage'] < 50]
    pollution_extra = raw_pollution.loc[sensors_extra.index]
    pollution = raw_pollution.loc[sensors.index]

    # Step 3: collecting all data
    dates = pd.to_datetime(raw_pollution.columns)
    longitude_extra = sensors_extra["longitude"].values
    latitude_extra = sensors_extra["latitude"].values
    pollution_extra = pollution_extra.values.T
    longitude = sensors["longitude"].values
    latitude = sensors['latitude'].values
    pollution = pollution.values.T
    pollution[pollution == 0] = np.nan

    return dates, longitude, latitude, pollution, longitude_extra, latitude_extra, pollution_extra