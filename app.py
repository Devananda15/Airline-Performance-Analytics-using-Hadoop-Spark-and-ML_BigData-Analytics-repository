
import streamlit as st
import pandas as pd
import plotly.express as px
import pydeck as pdk

from pyspark.sql import SparkSession
from pyspark.sql.functions import *

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans

# ------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------

st.set_page_config(
    page_title="Airline Performance Dashboard",
    layout="wide"
)

# ------------------------------------------------
# LOAD CSS
# ------------------------------------------------

with open("style.css") as f:
    st.markdown(
        f"<style>{f.read()}</style>",
        unsafe_allow_html=True
    )

# ------------------------------------------------
# SPARK SESSION
# ------------------------------------------------

spark = SparkSession.builder \
    .appName("AirlineDashboard") \
    .master("local[*]") \
    .config("spark.driver.memory", "4g") \
    .config("spark.executor.memory", "4g") \
    .config("spark.driver.host", "127.0.0.1") \
    .getOrCreate()

# ------------------------------------------------
# LOAD DATA FROM HDFS
# ------------------------------------------------

flights = spark.read.csv(
    "hdfs://localhost:9000/airline_data/flights.csv",
    header=True,
    inferSchema=True
)

airlines = spark.read.csv(
    "hdfs://localhost:9000/airline_data/airlines.csv",
    header=True,
    inferSchema=True
)

airports = spark.read.csv(
    "hdfs://localhost:9000/airline_data/airports.csv",
    header=True,
    inferSchema=True
)

# ------------------------------------------------
# DATA CLEANING
# ------------------------------------------------

flights = flights.dropDuplicates()

flights = flights.dropna(
    subset=[
        "AIRLINE",
        "ORIGIN_AIRPORT",
        "DESTINATION_AIRPORT"
    ]
)

# ------------------------------------------------
# SIDEBAR FILTERS
# ------------------------------------------------

st.sidebar.title("Dashboard Filters")

airline_list = ["All"] + sorted(
    [row["AIRLINE"] for row in flights.select("AIRLINE").distinct().collect()]
)

selected_airline = st.sidebar.selectbox(
    "Select Airline",
    airline_list
)

if selected_airline != "All":
    flights = flights.filter(
        flights["AIRLINE"] == selected_airline
    )

month_list = ["All"] + [str(i) for i in range(1, 13)]

selected_month = st.sidebar.selectbox(
    "Select Month",
    month_list
)

if selected_month != "All":
    flights = flights.filter(
        flights["MONTH"] == int(selected_month)
    )

# ------------------------------------------------
# KPI VALUES
# ------------------------------------------------

total_flights = flights.count()

total_airlines = flights.select(
    "AIRLINE"
).distinct().count()

delayed_flights = flights.filter(
    flights["ARRIVAL_DELAY"] > 15
).count()

cancelled_flights = flights.filter(
    flights["CANCELLED"] == 1
).count()

avg_delay = flights.select(
    avg("ARRIVAL_DELAY")
).collect()[0][0]

# ------------------------------------------------
# HEADER
# ------------------------------------------------

st.title("✈ Airline Performance Analytics Dashboard")

st.markdown(
"""
### Big Data Analytics using Hadoop HDFS • Spark • Streamlit • KMeans
"""
)

# ------------------------------------------------
# KPI CARDS
# ------------------------------------------------

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric(
    "Total Flights",
    f"{total_flights:,}"
)

col2.metric(
    "Airlines",
    total_airlines
)

col3.metric(
    "Delayed Flights",
    f"{delayed_flights:,}"
)

col4.metric(
    "Cancelled Flights",
    f"{cancelled_flights:,}"
)

col5.metric(
    "Average Delay",
    f"{avg_delay:.2f} min"
)

st.divider()

# ------------------------------------------------
# TABS
# ------------------------------------------------

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Airline Analytics",
    "Airport Analytics",
    "Route Analytics",
    "Delay Trends",
    "ML Clustering",
    "Flight Records"
])

# =================================================
# TAB 1 — AIRLINE ANALYTICS
# =================================================

with tab1:

    st.subheader("✈ Airline Delay Analysis")

    col1, col2 = st.columns(2)

    # ---------------- AIRLINE DELAYS ----------------

    airline_delay = flights.groupBy(
        "AIRLINE"
    ).agg(
        avg("ARRIVAL_DELAY").alias("AvgDelay")
    ).orderBy(
        "AvgDelay",
        ascending=False
    ).limit(10)

    airline_pd = airline_delay.toPandas()

    fig1 = px.bar(
        airline_pd,
        x="AIRLINE",
        y="AvgDelay",
        color="AvgDelay",
        title="Average Delay by Airline"
    )

    col1.plotly_chart(
        fig1,
        use_container_width=True
    )

    # ---------------- CANCELLATION ANALYSIS ----------------

    cancelled_count = flights.filter(
        flights["CANCELLED"] == 1
    ).count()

    non_cancelled_count = flights.filter(
        flights["CANCELLED"] == 0
    ).count()

    cancel_pd = pd.DataFrame({
        "Status": ["Cancelled", "Completed"],
        "Count": [cancelled_count, non_cancelled_count]
    })

    fig2 = px.pie(
        cancel_pd,
        names="Status",
        values="Count",
        title="Cancelled vs Completed Flights",
        hole=0.4
    )

    col2.plotly_chart(
        fig2,
        use_container_width=True
    )

# =================================================
# TAB 2 — AIRPORT ANALYTICS
# =================================================

with tab2:

    st.subheader("🛫 Airport Traffic Analytics")

    col1, col2 = st.columns(2)

    # ---------------- BUSY AIRPORTS ----------------

    busy_airports = flights.groupBy(
        "ORIGIN_AIRPORT"
    ).count().orderBy(
        "count",
        ascending=False
    ).limit(10)

    airport_pd = busy_airports.toPandas()

    fig3 = px.bar(
        airport_pd,
        x="ORIGIN_AIRPORT",
        y="count",
        color="count",
        title="Top Busy Airports"
    )

    col1.plotly_chart(
        fig3,
        use_container_width=True
    )

    # ---------------- DELAY DISTRIBUTION ----------------

    delay_sample = flights.select(
        "ARRIVAL_DELAY"
    ).dropna().limit(5000)

    delay_pd = delay_sample.toPandas()

    fig4 = px.histogram(
        delay_pd,
        x="ARRIVAL_DELAY",
        nbins=50,
        title="Delay Distribution"
    )

    col2.plotly_chart(
        fig4,
        use_container_width=True
    )

# =================================================
# TAB 3 — ROUTE ANALYTICS
# =================================================

with tab3:

    st.subheader("🛣 Route Analytics")

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
    ).limit(10)

    route_pd = popular_routes.toPandas()

    fig5 = px.bar(
        route_pd,
        x="ROUTE",
        y="count",
        color="count",
        title="Most Popular Routes"
    )

    st.plotly_chart(
        fig5,
        use_container_width=True
    )

    # ---------------- ROUTE MAP ----------------

    st.subheader("🗺 Flight Route Map")

    airport_coords = pd.DataFrame({
        "airport": ["ATL", "LAX", "ORD", "DFW", "DEN"],
        "lat": [33.6407, 33.9416, 41.9742, 32.8998, 39.8561],
        "lon": [-84.4277, -118.4085, -87.9073, -97.0403, -104.6737]
    })

    st.pydeck_chart(
        pdk.Deck(
            map_style="mapbox://styles/mapbox/light-v9",
            initial_view_state=pdk.ViewState(
                latitude=39.5,
                longitude=-98.35,
                zoom=3,
                pitch=50,
            ),
            layers=[
                pdk.Layer(
                    "ScatterplotLayer",
                    data=airport_coords,
                    get_position='[lon, lat]',
                    get_radius=50000,
                    pickable=True,
                ),
            ],
        )
    )

# =================================================
# TAB 4 — DELAY TRENDS
# =================================================

with tab4:

    st.subheader("📈 Delay Trend Analysis")

    col1, col2 = st.columns(2)

    # ---------------- MONTHLY DELAYS ----------------

    monthly_delay = flights.groupBy(
        "MONTH"
    ).agg(
        avg("ARRIVAL_DELAY").alias("AvgDelay")
    ).orderBy("MONTH")

    monthly_pd = monthly_delay.toPandas()

    fig6 = px.line(
        monthly_pd,
        x="MONTH",
        y="AvgDelay",
        markers=True,
        title="Monthly Average Delay"
    )

    col1.plotly_chart(
        fig6,
        use_container_width=True
    )

    # ---------------- DAY-WISE DELAYS ----------------

    weekday_delay = flights.groupBy(
        "DAY_OF_WEEK"
    ).agg(
        avg("ARRIVAL_DELAY").alias("AvgDelay")
    ).orderBy("DAY_OF_WEEK")

    weekday_pd = weekday_delay.toPandas()

    fig7 = px.area(
        weekday_pd,
        x="DAY_OF_WEEK",
        y="AvgDelay",
        title="Day-wise Delay Trends"
    )

    col2.plotly_chart(
        fig7,
        use_container_width=True
    )

# =================================================
# TAB 5 — ML CLUSTERING
# =================================================

with tab5:

    st.subheader("🤖 Flight Clustering Analysis")

    cluster_data = flights.select(
        "ARRIVAL_DELAY",
        "DEPARTURE_DELAY",
        "DISTANCE",
        "AIR_TIME"
    ).dropna()

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

    kmeans = KMeans(
        k=5,
        seed=1
    )

    model = kmeans.fit(final_data)

    predictions = model.transform(final_data)

    cluster_pd = predictions.select(
        "ARRIVAL_DELAY",
        "DISTANCE",
        "prediction"
    ).limit(5000).toPandas()

    fig8 = px.scatter(
        cluster_pd,
        x="DISTANCE",
        y="ARRIVAL_DELAY",
        color="prediction",
        title="Flight Clusters"
    )

    st.plotly_chart(
        fig8,
        use_container_width=True
    )

    cluster_counts = predictions.groupBy(
        "prediction"
    ).count()

    cluster_count_pd = cluster_counts.toPandas()

    fig9 = px.bar(
        cluster_count_pd,
        x="prediction",
        y="count",
        color="prediction",
        title="Cluster Distribution"
    )

    st.plotly_chart(
        fig9,
        use_container_width=True
    )

    st.info(
        """
        Cluster Insights:
        • Some clusters represent short-distance on-time flights
        • Some clusters represent highly delayed long-distance flights
        • Clustering helps identify operational patterns
        """
    )

# =================================================
# TAB 6 — FLIGHT RECORDS
# =================================================

with tab6:

    st.subheader("📄 Flight Dataset Records")

    st.dataframe(
        flights.limit(100).toPandas()
    )

# ------------------------------------------------
# FOOTER
# ------------------------------------------------

st.markdown("---")

st.markdown(
"""
<center>
Built using Hadoop HDFS • Spark • Streamlit • Plotly • KMeans
</center>
""",
unsafe_allow_html=True
)