from pyspark.sql import SparkSession
from pyspark.sql.functions import *

# Create Spark Session
spark = SparkSession.builder \
    .appName("AirlinePerformanceAnalytics") \
    .getOrCreate()

# Load datasets
flights = spark.read.csv(
    "dataset/flights.csv",
    header=True,
    inferSchema=True
)

airlines = spark.read.csv(
    "dataset/airlines.csv",
    header=True,
    inferSchema=True
)

airports = spark.read.csv(
    "dataset/airports.csv",
    header=True,
    inferSchema=True
)

# Remove duplicate rows
flights = flights.dropDuplicates()

# Remove rows only if important columns are null
flights = flights.dropna(
    subset=[
        "AIRLINE",
        "ORIGIN_AIRPORT",
        "DESTINATION_AIRPORT"
    ]
)

print("Datasets loaded successfully")

print("Total Flights:", flights.count())

flights.show(5)


# Total Flights
total_flights = flights.count()

# Total Airlines
total_airlines = flights.select(
    "AIRLINE"
).distinct().count()

# Delayed Flights
delayed_flights = flights.filter(
    flights["ARRIVAL_DELAY"] > 15
).count()

# Cancelled Flights
cancelled_flights = flights.filter(
    flights["CANCELLED"] == 1
).count()

print("Total Flights:", total_flights)
print("Total Airlines:", total_airlines)
print("Delayed Flights:", delayed_flights)
print("Cancelled Flights:", cancelled_flights)

airline_performance = flights.groupBy(
    "AIRLINE"
).agg(
    avg("ARRIVAL_DELAY").alias("AvgArrivalDelay"),
    avg("DEPARTURE_DELAY").alias("AvgDepartureDelay"),
    count("*").alias("TotalFlights")
).orderBy(
    "AvgArrivalDelay"
)

print("Airline Performance")
airline_performance.show()


busy_airports = flights.groupBy(
    "ORIGIN_AIRPORT"
).count().orderBy(
    "count",
    ascending=False
)

print("Busy Airports")
busy_airports.show(10)

routes = flights.withColumn(
    "ROUTE",
    concat_ws(
        " -> ",
        flights.ORIGIN_AIRPORT,
        flights.DESTINATION_AIRPORT
    )
)

popular_routes = routes.groupBy(
    "ROUTE"
).count().orderBy(
    "count",
    ascending=False
)

print("Popular Routes")
popular_routes.show(10)


cancellations = flights.filter(
    flights["CANCELLED"] == 1
)

cancel_analysis = cancellations.groupBy(
    "AIRLINE"
).count().orderBy(
    "count",
    ascending=False
)

print("Cancellation Analysis")
cancel_analysis.show()


from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans

# Select features
cluster_data = flights.select(
    "ARRIVAL_DELAY",
    "DEPARTURE_DELAY",
    "DISTANCE",
    "AIR_TIME"
)

# Remove nulls again
cluster_data = cluster_data.dropna()

# Convert to feature vector
assembler = VectorAssembler(
    inputCols=[
        "ARRIVAL_DELAY",
        "DEPARTURE_DELAY",
        "DISTANCE",
        "AIR_TIME"
    ],
    outputCol="features"
)

final_data = assembler.transform(cluster_data)

# KMeans model
kmeans = KMeans(
    k=5,
    seed=1
)

model = kmeans.fit(final_data)

predictions = model.transform(final_data)

print("Clustering Complete")

predictions.show(5)