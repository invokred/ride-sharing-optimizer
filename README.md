# Ride Sharing Travel Time Optimizer
The following project focuses on solving the NP-Hard *Travelling Salesman Problem*, hereforth mentioned as TSP. 
*The main motivation leading to developing this repository was to solve a real-life problem under the subject Operations Research. 
PS: Taking Genetic Algorithms as my additional was one of the best decisions of my student career to expand my repertoire of optimization techniques*

Ride-Sharing is a perfect example of TSP with only one arithmatical constraint -> For n trips TSP has to solve for 2n points (pickup locations + dropoff locations) For simplicity, only pickup locations have been considered here. 
To cite an example, consider a bus of a football team boarding the entire team (total n staff) from n locations. To answer where the bus would go in the end; it comes back to the initial most point. 
*Note: Entire focus here is an attempt to solve TSP in a meta-heuristic approach instead of the regualar exact algorithms (dynamic problem, branch-and-bound)*

## Dataset
The dataset used consists of *1.5 million records* of Taxi Travel data freely available opensource by [New York City Taxi and Limousine Commision](https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page) which can be found [here](https://www.kaggle.com/c/nyc-taxi-trip-duration). The problem statement was procured from [Kaggle](https://www.kaggle.com/c/nyc-taxi-trip-duration/data). The dataset consists of trip records of 3 types of taxis - a) Yellow b) Green c) For-hire-vehicles. Although they haven't been distinguished for generalizing the result thus eliminating Type-II error due to lack of external validity. Basically, this means that the solution thus obtained can be used for any cabs' rides not only restricting to NYC Cabs.

### Data Fields
* id
* vendor_id	
* pickup_datetime
* dropoff_datetime
* passenger_count
* pickup_longitude
* pickup_latitude
* dropoff_longitude
* dropoff_latitude
* store_and_fwd_flag
* trip_duration

## Requirements
* python3
* matplotlib
* numpy
* pandas
* xgboost
* geopy

## Verbosity
All files are developed using *Jupyter Notebook* for better debugging and increased utility. 

*XGBoost Regressor* has used to estimate travel times between a pickup and dropoff location. (I haven't added generated xgb_model.sav. You may generate your own by executing xgboost.ipynb after cloning)
*Ant Colony Optimization algorithm* has been used to find the optimal path with least time. (*XGBoost regressor* is used to build cost matrix)

### User Inputs
filename: final.ipynb
```
#user inputs
location_count = 15
ant_count = 10
g = 100
alpha = 1.0
beta = 10.0
rho = 0.5
q = 10.0
verbose = True
```

### Final Output
![](https://raw.githubusercontent.com/invokred/ride-sharing-optimizer/main/Figures/final-path.png)
