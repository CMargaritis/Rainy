from datetime import datetime
import requests
import openmeteo_requests
import requests_cache
from retry_requests import retry
import pandas as pd
import numpy as np  # Import numpy for wind calculations
import matplotlib.pyplot as plt
from nicegui import ui, app, native, Client
from scipy.interpolate import RectBivariateSpline
from fastapi import Request

import matplotlib
matplotlib.use("svg")

app.add_static_files("media","media")

# --- Configuration ---
# Setup the Open-Meteo API client with caching and retry on error
cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)
PRESSURE_LEVELS = [1000, 975, 950, 925, 900, 850, 800, 700, 600, 500, 400, 300, 250, 200, 150, 100, 70, 50, 30]
PRESSURE_SPARSE = [1000, 900, 800, 700, 600, 500, 400, 300, 200, 100, 30]
PRESSURE_LEVELS_EXPLAINED = [
    "1000hPa   110m",
    " 975hPa   320m",
    " 950hPa   500m",
    " 925hPa   800m",
    " 900hPa  1000m",
    " 850hPa  1500m",
    " 800hPa  1900m",
    " 700hPa    3km",
    " 600hPa  4.2km",
    " 500hPa  5.6km",
    " 400hPa  7.2km",
    " 300hPa  9.2km",
    " 250hPa 10.4km",
    " 200hPa 11.8km",
    " 150hPa 13.5km",
    " 100hPa 15.8km",
    "  70hPa 17.7km",
    "  50hPa 19.3km",
    "  30hPa 22.0km",
    
]

PRESSURE_SPARSE_EXPLAINED = [
    "1000hPa   110m",
    " 900hPa  1000m",
    " 800hPa  1900m",
    " 700hPa    3km",
    " 600hPa  4.2km",
    " 500hPa  5.6km",
    " 400hPa  7.2km",
    " 300hPa  9.2km",
    " 200hPa 11.8km",
    " 100hPa 15.8km",
    "  30hPa 22.0km",
    
]


TIME_AXIS = ["00:00", "01:00", "02:00", "03:00", "04:00", "05:00", "06:00", "07:00", "08:00", "09:00", "10:00", "11:00", "12:00", "13:00", "14:00", "15:00", "16:00", "17:00", "18:00", "19:00", "20:00", "21:00", "22:00", "23:00",]

weather_models = ["best_match", "ecmwf_ifs", "ecmwf_ifs025", "ecmwf_aifs025_single", "cma_grapes_global", "bom_access_global", "kma_seamless", "kma_ldps", "kma_gdps", "meteofrance_seamless", "meteofrance_arpege_world", "meteofrance_arpege_europe", "meteofrance_arome_france", "meteofrance_arome_france_hd", "knmi_seamless", "knmi_harmonie_arome_europe", "knmi_harmonie_arome_netherlands", "dmi_seamless", "dmi_harmonie_arome_europe", "gfs_seamless", "gfs_global", "gfs_hrrr", "ncep_nbm_conus", "ncep_nam_conus", "gfs_graphcast025", "icon_seamless", "icon_global", "icon_eu", "icon_d2", "italia_meteo_arpae_icon_2i", "ukmo_seamless", "ukmo_global_deterministic_10km", "ukmo_uk_deterministic_2km", "jma_seamless", "jma_msm", "jma_gsm", "gem_seamless", "gem_global", "gem_regional", "gem_hrdps_continental", "metno_seamless", "metno_nordic", "meteoswiss_icon_seamless", "meteoswiss_icon_ch1", "meteoswiss_icon_ch2"]
    
spline_tension = 0.4


# --- UI Layout and Styling ---

async def page():
    # print(ip = client.environ['asgi.scope']['client'][0])
    # ui.query('body').style('background-color: #f0f0f0;')
    # ui.label('Hello from PyInstaller')

    # ui.query('body').style('background-color: #f0f0f0;')
    
    
    def set_cookie(key,value,expire=365*24*3600):
        js = f'document.cookie = "{key}={value}; max-age={expire}; path=/; SameSite=Lax;"'
        ui.run_javascript(js)
        
    async def get_cookie(key):
        """Asynchronously gets a cookie value from the browser."""
        js_code = f"""
                    (function() {{
                        const value = document.cookie
                            .split('; ')
                            .find(row => row.startsWith('{key}='))
                            ?.split('=')[1];
                        return value;
                    }})();
                    """
        cookie_value = await ui.run_javascript(js_code)
        # The result can be None if the cookie isn't found
        return cookie_value
        
   
    with ui.card().classes('w-full'):
        # ui.label('Rainy').classes('text-2xl font-bold text-center mb-4')
        ui.image('media/logo.png').classes('w-1/4 mx-auto')
        
        # #get current city
        # url = f'http://ip-api.com/json/'
        # r = requests.get(url)
        # results = r.json()

        city         = await get_cookie('city' )  #app.storage.user.get('city',None)
        stored_model = await get_cookie('model')  #app.storage.user.get('model',weather_models[0])

        if stored_model is None: stored_model = weather_models[0]
        
        with ui.row().classes('w-full items-end'):
            location_input = ui.input(label='Location', placeholder='e.g., Delft',value=city).classes('flex-grow')
            
            model_select = ui.select(
                weather_models,
                value=stored_model,
                label='Weather Model'
            ).classes('w-48')
            
            with ui.input('Date', value=datetime.now().strftime('%Y-%m-%d')) as date_input:
                with ui.menu().props('no-parent-event') as menu:
                    with ui.date(value=datetime.now().strftime('%Y-%m-%d')).bind_value(date_input):
                        with ui.row().classes('justify-end'):
                            ui.button('Close', on_click=menu.close).props('flat')
            with date_input.add_slot('append'):
                ui.icon('edit_calendar').on('click', menu.open).classes('cursor-pointer')
            
            # date_input = ui.date(value=datetime.now().strftime('%Y-%m-%d'), on_change=lambda e: date_input.set_value(e.value)).props('flat')

        # print(date.value)
        # print(date_input.value)
        generate_button = ui.button('Reload Charts', on_click=lambda: generate_charts())

        plot_container = ui.row().classes('w-full justify-center')

    # --- Functions ---


    def interpolate_native(numpy_array):
        cleaned_array = numpy_array.copy()
        num_rows, num_cols = cleaned_array.shape
        # Pass 1: Interpolate down the columns (axis=0)
        for c in range(num_cols):
            col = cleaned_array[:, c]
            if np.all(np.isnan(col)):  # Skip if the whole column is NaN
                continue

            nan_indices = np.where(np.isnan(col))[0]
            non_nan_indices = np.where(~np.isnan(col))[0]
            non_nan_values = col[non_nan_indices]

            # Use np.interp for 1D linear interpolation
            interpolated_values = np.interp(nan_indices, non_nan_indices, non_nan_values)
            col[nan_indices] = interpolated_values
            cleaned_array[:, c] = col

        # Pass 2: Interpolate across the rows (axis=1)
        # This catches NaNs in rows where the entire column might have been NaN
        for r in range(num_rows):
            row = cleaned_array[r, :]
            if np.all(np.isnan(row)): # Skip if the whole row is NaN
                continue

            nan_indices = np.where(np.isnan(row))[0]
            non_nan_indices = np.where(~np.isnan(row))[0]
            non_nan_values = row[non_nan_indices]

            interpolated_values = np.interp(nan_indices, non_nan_indices, non_nan_values)
            row[nan_indices] = interpolated_values
            cleaned_array[r, :] = row

        # Final fallback: fill any remaining NaNs with 0
        cleaned_array[np.isnan(cleaned_array)] = 0
        return cleaned_array

    def numpy_to_echart_heatmap_data_with_interpolation_native(
        numpy_array: np.ndarray,
        x_labels: list = None,
        y_labels: list = None
    ) -> list:
        """
        Cleans a 2D NumPy array by interpolating NaN values using only NumPy, then
        converts it to the [x, y, value] format for ECharts heatmaps.

        The interpolation is a two-pass process:
        1. Pass 1: Interpolates linearly down each column (axis=0).
        2. Pass 2: Interpolates linearly across each row (axis=1) to fill NaNs
        that might remain (e.g., if a whole column was NaN).
        3. Final Fallback: Any remaining NaNs (e.g., if the whole array was NaN)
        are filled with 0.

        Args:
            numpy_array: A 2D NumPy array that may contain np.nan values.
            x_labels: Optional list of labels for the x-axis. Must match the
                    number of columns in the array.
            y_labels: Optional list of labels for the y-axis. Must match the
                    number of rows in the array.

        Returns:
            A list of lists in the format [x, y, value] for ECharts.
            Returns an empty list if the input is not a 2D array or is empty.
        """

        num_rows, num_cols = numpy_array.shape
        cleaned_array = interpolate_native(numpy_array)
        # --- End of Interpolation Step ---

        # Validate labels if provided
        if x_labels is not None and len(x_labels) != num_cols:
            raise ValueError(f"Length of x_labels ({len(x_labels)}) must match number of columns ({num_cols}).")
        if y_labels is not None and len(y_labels) != num_rows:
            raise ValueError(f"Length of y_labels ({len(y_labels)}) must match number of rows ({num_rows}).")

        echart_data = []
        for r_idx in range(num_rows):
            for c_idx in range(num_cols):
                value = cleaned_array[r_idx, c_idx]

                x_coord = x_labels[c_idx] if x_labels else c_idx
                y_coord = y_labels[r_idx] if y_labels else r_idx

                echart_data.append([x_coord, y_coord, value])

        return echart_data

    def resample_2d_array(original_array, original_y, original_x, new_y, new_x):
        """
        Resamples a 2D NumPy array to a new grid, handling monotonic axes.

        This function checks if axes are decreasing and reverses them and the data
        accordingly before performing bivariate spline interpolation. It assumes
        the input axes are monotonic (either strictly increasing or strictly decreasing).

        Args:
            original_array (np.ndarray): The input 2D NumPy array (data).
            original_y (np.ndarray): 1D array of original y-axis coordinates (monotonic).
            original_x (np.ndarray): 1D array of original x-axis coordinates (monotonic).
            new_y (np.ndarray): 1D array of the desired new y-axis coordinates.
            new_x (np.ndarray): 1D array of the desired new x-axis coordinates.

        Returns:
            np.ndarray: The resampled 2D NumPy array with shape (len(new_y), len(new_x)).
        """
        # Make copies to avoid modifying the user's original data
        x_processed = np.copy(original_x)
        y_processed = np.copy(original_y)
        data_processed = np.copy(original_array)

        # --- Pre-processing: Ensure axes are increasing for the interpolator ---

        # 1. Handle the X-axis
        # If the first element is greater than the last, the axis is decreasing.
        if x_processed[0] > x_processed[-1]:
            x_processed = x_processed[::-1]       # Reverse the x-axis
            data_processed = data_processed[:, ::-1] # Reverse the data columns

        # 2. Handle the Y-axis
        if y_processed[0] > y_processed[-1]:
            y_processed = y_processed[::-1]       # Reverse the y-axis
            data_processed = data_processed[::-1, :] # Reverse the data rows

        # --- Interpolation ---

        # Now that the axes and data are guaranteed to be in the correct order,
        # we can create the interpolator.
        interpolator = RectBivariateSpline(y_processed, x_processed, data_processed)

        # Evaluate the interpolator on the new grid.
        # We sort the new axes to ensure they are also monotonically increasing.
        resampled_array = interpolator(np.sort(new_y), np.sort(new_x))

        return resampled_array


    def geocode_location(location_name):
        """Geocode a location name using the free Open-Meteo Geocoding API."""
        url = "https://geocoding-api.open-meteo.com/v1/search"
        params = {"name": location_name, "count": 1}
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            results = response.json()
            if "results" in results and len(results["results"]) > 0:
                location = results["results"][0]
                return location["latitude"], location["longitude"], location.get("name", "")
        except requests.exceptions.RequestException as e:
            ui.notify(f"Geocoding request failed: {e}", color='negative')
        except Exception as e:
            ui.notify(f"An error occurred during geocoding: {e}", color='negative')
        return None, None, None

    def get_weather_data(latitude, longitude, date, model):
        """Fetch weather data from Open-Meteo API, including cloud cover and snowfall."""
        url = "https://api.open-meteo.com/v1/forecast"
        hourly_params = ["temperature_2m", "relative_humidity_2m", "apparent_temperature", "rain", "showers", "snowfall", "wind_speed_10m", "wind_direction_10m", "wind_gusts_10m", "surface_pressure", "precipitation", "precipitation_probability", "cloud_cover_low", "cloud_cover", "cloud_cover_mid", "cloud_cover_high", "visibility",]
        for p in PRESSURE_LEVELS:
            hourly_params.extend([
                f"temperature_{p}hPa",
                f"wind_speed_{p}hPa",
                f"wind_direction_{p}hPa",
                f"cloud_cover_{p}hPa"
            ])
            
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": date,
            "end_date": date,
            "hourly": hourly_params,
            "models": model,
            "timezone": "auto",
            "wind_speed_unit": "ms",
        }
        try:
            responses = openmeteo.weather_api(url, params=params)
            response = responses[0]

            hourly = response.Hourly()
            hourly_data = {
                "date": pd.to_datetime(hourly.Time(), unit="s", utc=True),
                "temperature_2m":            hourly.Variables( 0).ValuesAsNumpy(),
                "relative_humidity_2m":      hourly.Variables( 1).ValuesAsNumpy(),
                "apparent_temperature":      hourly.Variables( 2).ValuesAsNumpy(),
                "rain":                      hourly.Variables( 3).ValuesAsNumpy(),
                "showers":                   hourly.Variables( 4).ValuesAsNumpy(),
                "snowfall":                  hourly.Variables( 5).ValuesAsNumpy()*10,
                "wind_speed_10m":            hourly.Variables( 6).ValuesAsNumpy(),
                "wind_direction_10m":        hourly.Variables( 7).ValuesAsNumpy(),
                "wind_gusts_10m":            hourly.Variables( 8).ValuesAsNumpy(),
                "surface_pressure":          hourly.Variables( 9).ValuesAsNumpy(),
                "precipitation":             hourly.Variables(10).ValuesAsNumpy(),
                "precipitation_probability": hourly.Variables(11).ValuesAsNumpy(),
                "cloud_cover_low":           hourly.Variables(12).ValuesAsNumpy(),
                "cloud_cover":               hourly.Variables(13).ValuesAsNumpy(),
                "cloud_cover_mid":           hourly.Variables(14).ValuesAsNumpy(),
                "cloud_cover_high":          hourly.Variables(15).ValuesAsNumpy(),
                "visibility":                hourly.Variables(16).ValuesAsNumpy(),
                
            }
            
            pl_data = {
                "temperature"    : np.zeros((len(PRESSURE_LEVELS),24)),
                "wind_speed"     : np.zeros((len(PRESSURE_LEVELS),24)),
                "wind_direction" : np.zeros((len(PRESSURE_LEVELS),24)),
                "cloud_cover"    : np.zeros((len(PRESSURE_LEVELS),24)),
            }
            
            j = 0
            for i in range(17,len(hourly_params),4):
                pl_data["temperature"   ][j,:] = np.round(hourly.Variables(i  ).ValuesAsNumpy(),0)
                pl_data["wind_speed"    ][j,:] = np.round(hourly.Variables(i+1).ValuesAsNumpy(),0)
                pl_data["wind_direction"][j,:] = np.round(hourly.Variables(i+2).ValuesAsNumpy(),0)
                pl_data["cloud_cover"   ][j,:] = np.round(hourly.Variables(i+3).ValuesAsNumpy(),0)
                j += 1
            
            set_cookie('model',model) #app.storage.user['model'] = model
            return pd.DataFrame(data=hourly_data), pl_data
        
        except Exception as e:
            ui.notify(f"Failed to get weather data: {e}", color='negative')
            return None, None

    def create_temperature_plots(weather_data, pl_data):
        ui.echart({
            "title": {"text": 'Temperature at Surface'},
            "tooltip": {
                "trigger": 'axis',
                
            },
            "legend": {
                "data": ['Actual', 'Apparent'],
            },
            "xAxis": {
                "type": 'category',
                "data": TIME_AXIS,
                "boundaryGap": False,
            },
            "yAxis": {
                "type": 'value',
                "name": 'Temperature',
                "axisLabel": {"formatter": '{value} °C'},
            },
            "series": [
                {
                    "name": 'Actual',
                    "type": 'line',
                    "data": [round(x,1) for x in weather_data["temperature_2m"]],
                    'smooth': spline_tension,
                },
                {
                    "name": 'Apparent',
                    "type": 'line',
                    "data": [round(x,1) for x in weather_data["apparent_temperature"]],
                    'smooth': spline_tension,
                }
            ]
        })
        
        ui.echart({
            "tooltip": {
                "trigger": 'axis',
                
            },
            "xAxis": {
                "type": 'category',
                "data": TIME_AXIS,
                "boundaryGap": False,
            },
            "yAxis": {
                "type": 'value',
                "name": 'Relative Humidity',
                'min': 0,
                'max': 100,
                "axisLabel": {"formatter": '{value} %'},
            },
            "series": [
                {
                    "name": 'Relative Humidity',
                    "type": 'line',
                    "data": list(weather_data["relative_humidity_2m"]),
                    'smooth': spline_tension,
                },
            ]
        })
        
        
    def create_wind(weather_data, pl_data):
        ui.echart({
            "title": {"text": 'Wind at Surface'},
            "tooltip": {
                "trigger": 'axis',
                
            },
            "legend": {
                "data": ['Wind Speed', 'Wind Gusts'],
            },
            "xAxis": {
                "type": 'category',
                "data": TIME_AXIS,
                "boundaryGap": False,
            },
            "yAxis": [
                {
                "type": 'value',
                'min' : 0,
                'max' : max(30, np.max(weather_data["wind_speed_10m"]), np.max(weather_data["wind_gusts_10m"])),
                "name": 'Wind Speed',
                "axisLabel": {"formatter": '{value} m/s'},
                },],
            
            
            "series": [
                {
                    "name": 'Wind Speed',
                    "type": 'line',
                    "yAxisIndex": 0,
                    "data": [round(x,1) for x in weather_data["wind_speed_10m"]],
                    'smooth': spline_tension,
                    "markArea": {
                        "silent": True,
                        "data": [
                            [
                                {
                                    "yAxis": 0,
                                    "itemStyle": {
                                        "color": 'rgba(0, 175, 255, 0.2)'
                                    }
                                },
                                {
                                    "yAxis": 10
                                }
                            ],
                            [
                                {
                                    "yAxis": 10,
                                    "itemStyle": {
                                        "color": 'rgba(255, 255, 0, 0.2)'
                                    }
                                },
                                {
                                    "yAxis": 20
                                }
                            ],
                            [
                                {
                                    "yAxis": 20,
                                    "itemStyle": {
                                        "color": 'rgba(255, 123, 40, 0.2)'
                                    }
                                },
                                {
                                    # This will extend to the top of the chart
                                    "yAxis": max(30, np.max(weather_data["wind_speed_10m"]), np.max(weather_data["wind_gusts_10m"]))
                                }
                            ]
                        ]
                    }
                },
                {
                    "name": 'Wind Gusts',
                    "type": 'line',
                    "yAxisIndex": 0,
                    "data": [round(x,1) for x in weather_data["wind_gusts_10m"]],
                    'smooth': spline_tension,
                },
            ]
        })
        
        ui.echart({
            "tooltip": {
                "trigger": 'axis',
                
            },
            "xAxis": {
                "type": 'category',
                "data": TIME_AXIS,
                "boundaryGap": False,
            },
            "yAxis": [
                {
                "type": 'value',
                "name": 'Wind Direction',
                
                'interval' : 90,
                "min": 0,
                "max": 360,
                "axisLabel": {"formatter": '{value}°'},
                }
                ],
            
            
            "series": [
                {
                    "name": 'Wind Direction',
                    "type": 'line',
                    "data": list(weather_data["wind_direction_10m"]),
                    'smooth': spline_tension,
                }
            ]
        })
        

    def create_precipitation(weather_data, pl_data):
        ui.echart({
            "title": {"text": 'Precipitation'},
            "tooltip": {
                "trigger": 'axis',
                
            },
            "xAxis": {
                "type": 'category',
                "data": TIME_AXIS,
                "boundaryGap": False,
            },
            "yAxis": {
                "type": 'value',
                'min': 0,
                'max': 100,
                "name": 'Precipitation Probability',
                "axisLabel": {"formatter": '{value} %'},
            },
            "series": [
                {
                    "name": 'Precipitation Probability',
                    "type": 'line',
                    "data": list(weather_data["precipitation_probability"]),
                    'smooth': spline_tension,
                },
            ]
        })
                
        ui.echart({
            "tooltip": {
                "trigger": 'axis',
            },
            "legend": {
                "data": ["Rain", "Showers", "Snowfall", "Total"],
            },
            "xAxis": {
                "type": 'category',
                "data": TIME_AXIS,
                "boundaryGap": False,
            },
            "yAxis": {
                "type": 'value',
                'min': 0,
                'max': max(10, np.max(weather_data["rain"] + weather_data["showers"] + weather_data["snowfall"])),
                "name": 'Precipitation Type',
                "axisLabel": {"formatter": '{value} mm/h'},
            },
            "series": [
                {
                    "name": 'Rain',
                    "type": 'line',
                    'stack': 'Total',
                    'areaStyle': {},
                    "data": [round(x, 1) for x in weather_data["rain"]],
                    'smooth': spline_tension,
                    "lineStyle": {
                        "width": 0,
                    },
                    "symbol": 'none',
                    # Add markArea to the first series
                    "markArea": {
                        "silent": True,
                        "data": [
                            [
                                {
                                    "yAxis": 0,
                                    "itemStyle": {
                                        "color": 'rgba(0, 175, 255, 0.2)'
                                    }
                                },
                                {
                                    "yAxis": 3
                                }
                            ],
                            [
                                {
                                    "yAxis": 3,
                                    "itemStyle": {
                                        "color": 'rgba(255, 255, 0, 0.2)'
                                    }
                                },
                                {
                                    "yAxis": 6
                                }
                            ],
                            [
                                {
                                    "yAxis": 6,
                                    "itemStyle": {
                                        "color": 'rgba(255, 123, 40, 0.2)'
                                    }
                                },
                                {
                                    # This will extend to the top of the chart
                                    "yAxis": max(10, np.max(weather_data["rain"] + weather_data["showers"] + weather_data["snowfall"]))
                                }
                            ]
                        ]
                    }
                },
                {
                    "name": 'Showers',
                    "type": 'line',
                    'stack': 'Total',
                    'areaStyle': {},
                    "data": [round(x, 1) for x in weather_data["showers"]],
                    'smooth': spline_tension,
                    "lineStyle": {
                        "width": 0,
                    },
                    "symbol": 'none',
                },
                {
                    "name": 'Snowfall',
                    "type": 'line',
                    'stack': 'Total',
                    'areaStyle': {},
                    "lineStyle": {
                        "width": 0,
                    },
                    "data": [round(x, 1) for x in weather_data["snowfall"]],
                    'smooth': spline_tension,
                    "symbol": 'none',
                },
                {
                    "name": 'Total',
                    "type": 'line',
                    "data": [round(x, 1) for x in weather_data["rain"] + weather_data["showers"] + weather_data["snowfall"]],
                    'smooth': spline_tension,
                },
            ],
        })
        
    def create_clouds(weather_data, pl_data):
        
        
        
        
        ui.echart({
            "title": {"text": 'Clouds'},
            "tooltip": {
                "trigger": 'axis',
                
            },
            "legend": {
                "data": ["Low", "Mid", "High", "Total"],
            },
            "xAxis": {
                "type": 'category',
                "data": TIME_AXIS,
                "boundaryGap": False,
            },
            "yAxis": {
                "type": 'value',
                'min': 0,
                'max': 100,
                "name": 'Cloud Cover Type',
                "axisLabel": {"formatter": '{value} %'},
            },
            "series": [
                {
                    "name": 'Low',
                    "type": 'line',
                    "data": list(weather_data["cloud_cover_low"]),
                    'smooth': spline_tension,
                },
                {
                    "name": 'Mid',
                    "type": 'line',
                    "data": list(weather_data["cloud_cover_mid"]),
                    'smooth': spline_tension,
                },
                {
                    "name": 'High',
                    "type": 'line',
                    "data": list(weather_data["cloud_cover_high"]),
                    'smooth': spline_tension,
                },
                {
                    "name": 'Total',
                    "type": 'line',
                    "data": list(weather_data["cloud_cover"]),
                    'smooth': spline_tension,
                },
            ]
        })
        
        ui.echart({
            "tooltip": {
                "trigger": 'axis',
                
            },
            "xAxis": {
                "type": 'category',
                "data": TIME_AXIS,
                "boundaryGap": False,
            },
            "yAxis": {
                "type": 'value',
                "name": 'Visibility',
                "axisLabel": {"formatter": '{value} m'},
            },
            "series": [
                {
                    "name": 'Visibility',
                    "type": 'line',
                    "data": list(weather_data["visibility"]),
                    'smooth': spline_tension,
                },
            ]
        })
        
        
        # clouds_high_res, (new_width, new_height) = upsample_and_interpolate(pl_data["cloud_cover"],10)
        ui.echart({
            "tooltip": {
                'position': 'top',
                'axisPointer': {
                    'type': 'cross',
                    'crossStyle': {
                        'color': '#999'
                    }
                }
                
            },
            "xAxis": {
                "type": 'category',
                "data": TIME_AXIS,
            },
            "yAxis": {
                "type": 'category',
                "data": PRESSURE_LEVELS_EXPLAINED,
                "name": 'Cloud Cover %',
                'axisLabel': {
                    'fontFamily': 'monospace' 
                }
                
            },
            'visualMap': {
                'min': 0,
                'max': 100,
                'calculable': 'true',
                'realtime': 'false',
                'inRange': {
                'color': ['#FFFFFF', '#333333']
                },
                'orient': 'horizontal',
                'left': 'center',
                # 'bottom': '0%',
            },
            'series': {
                'type' : 'heatmap',
                'data' : numpy_to_echart_heatmap_data_with_interpolation_native(pl_data["cloud_cover"])
            }
            
        })
    
    
    def create_airchart(weather_date, pl_data):
        def wind_speed_dir_to_uv(speed_ms, direction_deg):
            """
            Converts wind speed and direction into U and V components in knots.
            """
            M_S_TO_KNOTS = 1.94384
            speed_knots = speed_ms * M_S_TO_KNOTS
            direction_rad = np.radians(direction_deg)
            u = -speed_knots * np.sin(direction_rad)
            v = -speed_knots * np.cos(direction_rad)
            return u, v

        # --- 1. Generate Sample Data (24-hour period) ---
        pressure_levels = PRESSURE_SPARSE
        time_steps = np.arange(0, 26, 2)

        X, Y = np.meshgrid(time_steps, pressure_levels)

        T               = resample_2d_array(interpolate_native(pl_data["temperature"]   ),PRESSURE_LEVELS,np.arange(0,24,1),PRESSURE_SPARSE,time_steps)
        wind_speed      = resample_2d_array(interpolate_native(pl_data["wind_speed"]    ),PRESSURE_LEVELS,np.arange(0,24,1),PRESSURE_SPARSE,time_steps)
        wind_direction  = resample_2d_array(interpolate_native(pl_data["wind_direction"]),PRESSURE_LEVELS,np.arange(0,24,1),PRESSURE_SPARSE,time_steps)
        

        # --- 2. Process Wind Data ---
        u, v = wind_speed_dir_to_uv(wind_speed, wind_direction)

        # --- 3. Create the Plot ---
        with ui.matplotlib(figsize=(12,8)).figure as fig:
            ax = fig.gca()
            

            # --- Plot 1: Temperature as a filled colormap ---
            contour_levels = np.arange(np.min(T,), np.max(T), 5)

            temp_contour = ax.contourf(X, Y, T, levels=contour_levels, cmap='cool', extend='both')
            cbar = fig.colorbar(temp_contour, ax=ax, orientation='vertical', pad=0.02)
            cbar.set_label('Temperature (°C)', fontsize=12)

            # --- Plot 2: Wind Barbs ---
            ax.barbs(X, Y, u, v, length=7, pivot='middle', color='black')

            # --- 4. Configure the Axes ---
            # ax.invert_yaxis()
            # ax.set_yscale('log')
            ax.set_yticks(pressure_levels[::-1])
            ax.set_yticklabels(PRESSURE_SPARSE_EXPLAINED)
            ax.set_ylim(0,1050)


            ax.set_xticks(time_steps)
            ax.set_xticklabels([f"{i:02d}:00" for i in time_steps])
            ax.set_xlim(-1, 25)

            # --- 5. Add Titles and Labels ---
            ax.set_title('24-Hour Cross-Section of Wind and Temperature', fontsize=16, pad=15)
            ax.grid(True, linestyle='--', alpha=0.5)

            fig.tight_layout()
            


    def generate_charts():
        """Main function to orchestrate geocoding, data fetching, and chart generation."""
        location = location_input.value
        date = date_input.value
        model = model_select.value

        if not location:
            ui.notify("Please enter a location.", color='warning')
            return

        lat, lon, found_name = geocode_location(location)
        if lat is None or lon is None:
            ui.notify(f"Could not find coordinates for '{location}'. Please be more specific.", color='negative')
            return
        else:
            #app.storage.user['city'] = location
            set_cookie('city',location)

        ui.notify(f"Found location: {found_name}. Fetching weather data...", color='info')
        weather_data, pl_data = get_weather_data(lat, lon, date, model)

        if weather_data is not None:
            # Clear previous plots before adding new ones
            plot_container.clear()
            with plot_container:
                try:
                    create_temperature_plots(weather_data, pl_data)
                except Exception as e:
                    pass
                
                try:
                    create_wind(weather_data,pl_data)
                except Exception as e:
                    pass
                try:
                    create_precipitation(weather_data,pl_data)
                except Exception as e:
                    pass
                try:
                    create_clouds(weather_data,pl_data)
                except Exception as e:
                    pass
                try:
                    create_airchart(weather_data,pl_data)
                except Exception as e:
                    pass
                
                
            ui.notify("Charts generated successfully!", color='positive')
        else:
            ui.notify("No weather data available for the selected criteria.", color='warning')


    generate_charts()

# --- Run the App ---
ui.run(
    page,
    title="Rainy",
    port=8080, 
    reload=False, 
    storage_secret="z%3T5Kjwhu&zVQK**Uq%Hhd5C2LKG93F7u7BhYXU",
    favicon="media/favicon.png"
    )