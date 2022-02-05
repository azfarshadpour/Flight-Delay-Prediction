# Midterm_project

Project scope: We utilized historical flight data in supervised machine learning (ML) algorithms to create predictions of flight delays.

The tables we analyzed are as follows: 

flights.csv: The departure and arrival information about flights in US in years 2018 and 2019.

fuel_consumption.csv: The fuel consumption of different airlines from years 2015-2019 aggregated per month.

passengers.csv: The passenger totals on different routes from years 2015-2019 aggregated per month.

flights_test.csv: The departure and arrival information about flights in US in January 2020. This table is used for evaluation. As part of our submission requirements, we have predicted delays on flights from first 7 days of 2020 (1st of January - 7th of January). A sample submission is included in file sample_submission.csv.

On the aforementioned data, we have performed data exploration analysis techniques, which were further used in ML modelling. From the statistical models selected, we have formulated data insights, which enable our final algorithm to make accurate and precise predictions in regard to airline flight delays.

## Data Exploration

In this repository, you will find the Jupyter Notebook that contains the data exploration and cleaning for the flights.csv used to model and predict airline delays.

We started off by creating a pandas dataframe out of the CSV file, then we sampled 200,000 rows from the total flights.csv, to make the data more manageable for local computation.

We cleaned all the null values and removed features that had a sister (duplicate) column. 

Below is the glossary of all the features/categories available

Glossary:

- fl_date: Flight Date (yyyy-mm-dd)
- mkt_unique_carrier: Unique Marketing Carrier Code. When the same code has been used by multiple carriers, a numeric suffix is used for earlier users, for example, PA, PA(1), PA(2). Use this field for analysis across a range of years.
- branded_code_share: Reporting Carrier Operated or Branded Code Share Partners
- mkt_carrier: Code assigned by IATA and commonly used to identify a carrier. As the same code may have been assigned to different carriers over time, the code is not always unique. For analysis, use the Unique Carrier Code.
- mkt_carrier_fl_num: Flight Number
- op_unique_carrier: Unique Scheduled Operating Carrier Code. When the same code has been used by multiple carriers, a numeric suffix is used for earlier users, for example, PA, PA(1), PA(2). Use this field for analysis across a range of years.
- tail_num: Tail Number
- op_carrier_fl_num: Flight Number
- origin_airport_id: Origin Airport, Airport ID. An identification number assigned by US DOT to identify a unique airport. Use this field - for airport analysis across a range of years because an airport can change its airport code and airport codes can be reused.
- origin: Origin Airport
- origin_city_name: Origin Airport, City Name
- dest_airport_id: Destination Airport, Airport ID. An identification number assigned by US DOT to identify a unique airport. Use this - field for airport analysis across a range of years because an airport can change its airport code and airport codes can be reused.
- dest: Destination Airport
- dest_city_name: Destination Airport, City Name
- crs_dep_time: CRS Departure Time (local time: hhmm)
- dep_time: Actual Departure Time (local time: hhmm)
- dep_delay: Difference in minutes between scheduled and actual departure time. Early departures show negative numbers.
- taxi_out: Taxi Out Time, in Minutes
- wheels_off: Wheels Off Time (local time: hhmm)
- wheels_on: Wheels On Time (local time: hhmm)
- taxi_in: Taxi In Time, in Minutes
- crs_arr_time: CRS Arrival Time (local time: hhmm)
- arr_time: Actual Arrival Time (local time: hhmm)
- arr_delay: Difference in minutes between scheduled and actual arrival time. Early arrivals show negative numbers.
- cancelled: Cancelled Flight Indicator (1=Yes)
- cancellation_code: Specifies The Reason For Cancellation
- diverted: Diverted Flight Indicator (1=Yes)
- dup: Duplicate flag marked Y if the flight is swapped based on Form-3A data
- crs_elapsed_time: CRS Elapsed Time of Flight, in Minutes
- actual_elapsed_time: Elapsed Time of Flight, in Minutes
- air_time: Flight Time, in Minutes
- flights: Number of Flights
- distance: Distance between airports (miles)
- carrier_delay: Carrier Delay, in Minutes
- weather_delay: Weather Delay, in Minutes
- nas_delay: National Air System Delay, in Minutes
- security_delay: Security Delay, in Minutes
- late_aircraft_delay: Late Aircraft Delay, in Minutes
- first_dep_time: First Gate Departure Time at Origin Airport
- total_add_gtime: Total Ground Time Away from Gate for Gate Return or Cancelled Flight
- longest_add_gtime: Longest Time Away from Gate for Gate Return or Cancelled Flight

DEP_DELAY (Departure Delay) was removed from the dataset as its inclusion would result in biased models.It is understood that if a plane is leaving late then its chances of arriving late to its destination will increase. Normally when your flight leaves late, the airlines pushes for the flights to have shorter elapsed (flight) times to compensate for the delay. (This can be found in our EDA notebook.)

In addition, there were other features we didnt use due to it not being availble in the test dataset, as well as it potentially could not being in future data. EDA notebook will highlight features that were removed for the reasons previously stated.

# Modelling

Our next step after cleaning the data through a thorough EDA process done in two stages, is modelling.

This dataset consists of 28 features, out of which there are a series of them (listed above) that can affect the predictive model in a positive way in terms of predictions and therefore accuracy. However, when you use them, you are making the assumption that you are most probably already sitting in the plane, or in the best case scenario, your flight status on the departure boards has been changed to: "delayed".

Because we were not sure which ML algorithm will be the best for this type Regression problem we tested the following:

1. Random Forests
2. Linear Regression
3. Logistic Regression
4. Lasso Regression
5. XGBoost

We already know that this dataset if severely imbalanced which will force us to handle the data so ML algorithms can interpret.

## Machine Learning

For the ML the workflow was pretty straight forward by starting defining the target, which was the ARR_delay, and then dropping from the dataframe to define X (features). With this done, We split the data with a 30% and 70% for the test and training set respectively and used a typical random_State of 13.

We found the XGBoost gave best results while handling the variables of the data. 

(add in precision, accuracy and another score to evaluate out ML algorithm)

## Summary & Recommendations

- After EDA was completed, we found there are many ways a flight can be delayed; based on previous departure times, luggage delays, or unexpected extreme weather. However it is important to remember that these conclusions are based on a 2 year data analysis and this could well be a good year for some airlines and bad for others for any particular reason, which may change the results of the analysis.

- It is quite hard to create a ML model for flight delay prediction before you even know that the flight is delayed on the departure board. Neural Networks respond a lot better under these conditions with an average difference in accuracy, precision and recall. Maybe an even more thorough feature analysis could also raise these metrics , so it might be worth investing the time to do so.

- There are a series of variables (features) that were not included on this project due to a shortage of data and we believe after our research that they are key to predicting a flight delay accurately. Some of these are the weather, mechanical issues, and security issues. Then inside some of these there are sub-categories that also play a key role such as humidity, wind, precipitation, etc, and should be accounted for. Much of this data is available in APIs but needs to be scraped from different websites and it will require quite a lot more work to add it to the existing dataset, but will certainly translate into more realistic and therefore more accurate predictions.

# Way Forward

- Add more additional features that can be used in EDA of the data.

- Do EDA with 10 year data set, and not just a 2 year period, would grant more insight into the airlines performance and overall trending.

- Re-run the ML model with the best metrics again but with more objectives in regard to selecting how particular airlines do in a respective city.

# Notebook BreakDown 

- README.md

- exploratory_analysis notebook

- modeling notebook

- submission.csv 
