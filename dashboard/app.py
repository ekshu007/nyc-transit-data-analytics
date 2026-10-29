from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ------------------------------------------------------------
# Page setup and paths
# ------------------------------------------------------------
st.set_page_config(
    page_title="NYC Subway Analytics",
    page_icon="🚇",
    layout="wide",
    initial_sidebar_state="expanded",
)

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "data" / "subway_ml_dataset.parquet"
REPORTS_DIR = PROJECT_DIR / "reports"

st.title("🚇 NYC Subway Analytics")
st.markdown(
    "Explore subway segment travel times, service patterns, and model performance."
)
st.caption(
    "Travel time represents an individual station-to-station segment, "
    "not a complete journey or a direct measure of delays."
)


# ------------------------------------------------------------
# Data loading
# ------------------------------------------------------------
@st.cache_data
def load_data(path_string: str, modified_time: float) -> pd.DataFrame:
    data = pd.read_parquet(path_string)
    data["service_date"] = pd.to_datetime(data["service_date"])
    data["mean_minutes"] = data["mean_seconds"] / 60.0
    data["month_number"] = data["service_date"].dt.month
    data["month_name"] = data["service_date"].dt.strftime("%B")
    data["weekday_name"] = data["service_date"].dt.day_name()
    return data


@st.cache_data
def load_csv(path_string: str, modified_time: float) -> pd.DataFrame:
    return pd.read_csv(path_string)


if not DATA_PATH.exists():
    st.error(f"Dataset not found: {DATA_PATH}")
    st.info(
        "Expected file: data/subway_ml_dataset.parquet. "
        "Run the dataset-preparation cell in your ML notebook first."
    )
    st.stop()

df = load_data(str(DATA_PATH), DATA_PATH.stat().st_mtime)

# ------------------------------------------------------------
# Sidebar filters
# ------------------------------------------------------------
st.sidebar.title("Dashboard Filters")

min_date = df["service_date"].min().date()
max_date = df["service_date"].max().date()

date_range = st.sidebar.date_input(
    "Service date range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)

route_options = sorted(df["route_ids"].dropna().unique().tolist())
selected_routes = st.sidebar.multiselect(
    "Route labels",
    options=route_options,
    default=[],
    help="Leave empty to include every route label.",
)

filtered_df = df.copy()

if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
    start_date, end_date = date_range
    filtered_df = filtered_df[
        filtered_df["service_date"].between(
            pd.Timestamp(start_date),
            pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1),
        )
    ]

if selected_routes:
    filtered_df = filtered_df[filtered_df["route_ids"].isin(selected_routes)]

if filtered_df.empty:
    st.warning("No observations match these filters. Change the date or route selection.")
    st.stop()

st.sidebar.markdown("---")
st.sidebar.caption("Charts update based on the selected date range and route labels.")

# ------------------------------------------------------------
# KPI overview
# ------------------------------------------------------------
st.subheader("Network Overview")

unique_segments = filtered_df[
    ["a_station", "b_station", "route_ids"]
].drop_duplicates().shape[0]

k1, k2, k3, k4 = st.columns(4)
k1.metric("Segment observations", f"{len(filtered_df):,}")
k2.metric("Unique segments", f"{unique_segments:,}")
k3.metric("Average segment time", f"{filtered_df['mean_minutes'].mean():.2f} min")
k4.metric("Median segment time", f"{filtered_df['mean_minutes'].median():.2f} min")

k5, k6, k7 = st.columns(3)
k5.metric("Service days", f"{filtered_df['service_date'].nunique():,}")
k6.metric("Route labels", f"{filtered_df['route_ids'].nunique():,}")
k7.metric(
    "Selected date coverage",
    f"{filtered_df['service_date'].min():%d %b %Y} – "
    f"{filtered_df['service_date'].max():%d %b %Y}",
)

st.markdown("---")

overview_tab, segment_tab, route_tab, ml_tab, data_tab = st.tabs(
    [
        "📊 Trends & Overview",
        "🚉 Segment Analysis",
        "🚇 Route Analysis",
        "🤖 ML Evaluation",
        "📁 Data Explorer",
    ]
)

# ------------------------------------------------------------
# Tab 1: Trends
# ------------------------------------------------------------
with overview_tab:
    st.subheader("Monthly Travel-Time Trends")

    monthly = (
        filtered_df.groupby(["month_number", "month_name"], as_index=False)
        .agg(
            average_minutes=("mean_minutes", "mean"),
            median_minutes=("mean_minutes", "median"),
            observations=("mean_seconds", "size"),
            service_days=("service_date", "nunique"),
        )
        .sort_values("month_number")
    )

    monthly_long = monthly.melt(
        id_vars=["month_number", "month_name"],
        value_vars=["average_minutes", "median_minutes"],
        var_name="Metric",
        value_name="Travel Time (minutes)",
    )
    monthly_long["Metric"] = monthly_long["Metric"].map(
        {"average_minutes": "Average", "median_minutes": "Median"}
    )

    month_order = [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December",
    ]
    fig_monthly = px.line(
        monthly_long,
        x="month_name",
        y="Travel Time (minutes)",
        color="Metric",
        markers=True,
        category_orders={"month_name": month_order},
        title="Average and Median Segment Travel Time by Month",
    )
    fig_monthly.update_layout(xaxis_title="Month")
    st.plotly_chart(fig_monthly, use_container_width=True)

    st.caption(
        "Monthly averages may change because of segment coverage and the mix of "
        "observed segments, not necessarily because service worsened."
    )

    left, right = st.columns(2)
    with left:
        st.subheader("Travel-Time Distribution")
        fig_dist = px.histogram(
            filtered_df,
            x="mean_minutes",
            nbins=60,
            title="Distribution of Segment Travel Times",
            labels={"mean_minutes": "Travel time (minutes)"},
        )
        st.plotly_chart(fig_dist, use_container_width=True)

    with right:
        st.subheader("Weekday Comparison")
        weekday_order = [
            "Monday", "Tuesday", "Wednesday", "Thursday",
            "Friday", "Saturday", "Sunday",
        ]
        weekday = (
            filtered_df.groupby("weekday_name", as_index=False)
            .agg(
                average_minutes=("mean_minutes", "mean"),
                median_minutes=("mean_minutes", "median"),
                observations=("mean_seconds", "size"),
            )
        )
        weekday["weekday_name"] = pd.Categorical(
            weekday["weekday_name"], categories=weekday_order, ordered=True
        )
        weekday = weekday.sort_values("weekday_name")
        fig_weekday = px.bar(
            weekday,
            x="weekday_name",
            y="average_minutes",
            title="Average Segment Travel Time by Weekday",
            labels={
                "weekday_name": "Weekday",
                "average_minutes": "Average travel time (minutes)",
            },
        )
        st.plotly_chart(fig_weekday, use_container_width=True)

    st.subheader("Monthly Data Coverage")
    st.dataframe(monthly, use_container_width=True, hide_index=True)

# ------------------------------------------------------------
# Tab 2: Segment analysis
# ------------------------------------------------------------
with segment_tab:
    st.subheader("Slowest Station-to-Station Segments")
    minimum_records = st.slider(
        "Minimum observations per segment",
        min_value=10,
        max_value=200,
        value=50,
        step=10,
    )

    segment_stats = (
        filtered_df.groupby(["a_station", "b_station", "route_ids"], as_index=False)
        .agg(
            record_count=("mean_seconds", "size"),
            average_minutes=("mean_minutes", "mean"),
            median_minutes=("mean_minutes", "median"),
            std_seconds=("mean_seconds", "std"),
            observed_days=("service_date", "nunique"),
        )
    )
    segment_stats = segment_stats[
        segment_stats["record_count"] >= minimum_records
    ].copy()
    segment_stats["variability_pct"] = (
        100 * segment_stats["std_seconds"]
        / (segment_stats["average_minutes"] * 60).replace(0, float("nan"))
    )

    if segment_stats.empty:
        st.warning("No segments meet this minimum-observation threshold.")
    else:
        slowest = segment_stats.nlargest(15, "average_minutes")
        fig_slowest = px.bar(
            slowest.sort_values("average_minutes"),
            x="average_minutes",
            y="a_station",
            color="route_ids",
            orientation="h",
            hover_data=["b_station", "record_count", "median_minutes"],
            title="15 Slowest Segments by Average Travel Time",
            labels={
                "a_station": "Starting station",
                "average_minutes": "Average travel time (minutes)",
                "route_ids": "Route",
            },
        )
        st.plotly_chart(fig_slowest, use_container_width=True)
        st.dataframe(
            slowest[
                [
                    "a_station", "b_station", "route_ids", "record_count",
                    "observed_days", "average_minutes", "median_minutes",
                ]
            ].round(2),
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("---")
    st.subheader("Segments with the Highest Travel-Time Variability")

    if not segment_stats.empty:
        variable = (
            segment_stats.dropna(subset=["variability_pct"])
            .nlargest(15, "variability_pct")
        )
        fig_variable = px.bar(
            variable.sort_values("variability_pct"),
            x="variability_pct",
            y="a_station",
            color="route_ids",
            orientation="h",
            hover_data=["b_station", "record_count", "average_minutes"],
            title="15 Most Variable Segments",
            labels={
                "a_station": "Starting station",
                "variability_pct": "Coefficient of variation (%)",
                "route_ids": "Route",
            },
        )
        st.plotly_chart(fig_variable, use_container_width=True)
        st.dataframe(
            variable[
                [
                    "a_station", "b_station", "route_ids", "record_count",
                    "average_minutes", "std_seconds", "variability_pct",
                ]
            ].round(2),
            use_container_width=True,
            hide_index=True,
        )

    st.caption(
        "Variability is standard deviation divided by the mean. High variability "
        "does not by itself prove a delay or service problem."
    )

# ------------------------------------------------------------
# Tab 3: Route analysis
# ------------------------------------------------------------
with route_tab:
    st.subheader("Compare Route Labels")
    route_stats = (
        filtered_df.groupby("route_ids", as_index=False)
        .agg(
            observations=("mean_seconds", "size"),
            average_minutes=("mean_minutes", "mean"),
            median_minutes=("mean_minutes", "median"),
            observed_days=("service_date", "nunique"),
        )
        .sort_values("average_minutes", ascending=False)
    )

    unique_segment_counts = (
        filtered_df.groupby("route_ids")
        .apply(
            lambda group: group[["a_station", "b_station"]].drop_duplicates().shape[0],
            include_groups=False,
        )
    )
    route_stats["unique_segments"] = route_stats["route_ids"].map(unique_segment_counts)

    fig_route = px.bar(
        route_stats.head(20).sort_values("average_minutes"),
        x="average_minutes",
        y="route_ids",
        orientation="h",
        hover_data=["observations", "unique_segments", "median_minutes"],
        title="20 Route Labels with the Highest Average Segment Time",
        labels={
            "route_ids": "Route label",
            "average_minutes": "Average segment travel time (minutes)",
        },
    )
    st.plotly_chart(fig_route, use_container_width=True)
    st.dataframe(route_stats.round(2), use_container_width=True, hide_index=True)
    st.caption(
        "Combined labels such as 'A C' are treated as labels in the dataset. "
        "These are segment averages, not end-to-end journey times."
    )

# ------------------------------------------------------------
# Tab 4: ML evaluation
# ------------------------------------------------------------
with ml_tab:
    st.subheader("Model Performance Comparison")
    comparison_path = REPORTS_DIR / "model_comparison.csv"
    predictions_path = REPORTS_DIR / "test_predictions.csv"

    if comparison_path.exists():
        comparison = load_csv(
            str(comparison_path), comparison_path.stat().st_mtime
        )
        metric = st.radio(
            "Evaluation metric",
            ["MAE (seconds)", "RMSE (seconds)"],
            horizontal=True,
        )
        comparison = comparison.sort_values(metric)
        fig_models = px.bar(
            comparison,
            x="Model",
            y=metric,
            color="Model",
            title=f"Model Comparison — {metric}",
            text=metric,
        )
        fig_models.update_traces(texttemplate="%{text:.2f}", textposition="outside")
        fig_models.update_layout(showlegend=False)
        st.plotly_chart(fig_models, use_container_width=True)
        st.dataframe(comparison.round(2), use_container_width=True, hide_index=True)

        st.download_button(
            "Download model comparison CSV",
            data=comparison.to_csv(index=False).encode("utf-8"),
            file_name="model_comparison.csv",
            mime="text/csv",
        )
        st.info(
            "The historical segment median uses only training-period history for "
            "known segments. It is a historical benchmark, not a trained ML model. "
            "Random Forest is the trained machine-learning model."
        )
    else:
        st.warning("Model comparison file not found in reports/.")

    if predictions_path.exists():
        st.markdown("---")
        st.subheader("Actual vs Predicted Travel Time")
        predictions = load_csv(
            str(predictions_path), predictions_path.stat().st_mtime
        )
        required = {
            "actual_seconds",
            "random_forest_prediction",
            "segment_median_prediction",
        }

        if required.issubset(predictions.columns):
            sample_size = min(3000, len(predictions))
            plot_data = predictions.sample(n=sample_size, random_state=42)
            max_value = max(
                plot_data["actual_seconds"].max(),
                plot_data["random_forest_prediction"].max(),
            ) / 60

            fig_scatter = go.Figure()
            fig_scatter.add_trace(
                go.Scattergl(
                    x=plot_data["actual_seconds"] / 60,
                    y=plot_data["random_forest_prediction"] / 60,
                    mode="markers",
                    name="Random Forest",
                    marker={"size": 5, "opacity": 0.45},
                )
            )
            fig_scatter.add_trace(
                go.Scatter(
                    x=[0, max_value],
                    y=[0, max_value],
                    mode="lines",
                    name="Perfect prediction",
                    line={"dash": "dash"},
                )
            )
            fig_scatter.update_layout(
                title="Random Forest: Actual vs Predicted",
                xaxis_title="Actual segment time (minutes)",
                yaxis_title="Predicted segment time (minutes)",
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

            predictions["rf_absolute_error_seconds"] = (
                predictions["actual_seconds"]
                - predictions["random_forest_prediction"]
            ).abs()
            predictions["segment_absolute_error_seconds"] = (
                predictions["actual_seconds"]
                - predictions["segment_median_prediction"]
            ).abs()

            error_summary = pd.DataFrame(
                {
                    "Model": ["Random Forest", "Historical Segment Median"],
                    "MAE (seconds)": [
                        predictions["rf_absolute_error_seconds"].mean(),
                        predictions["segment_absolute_error_seconds"].mean(),
                    ],
                }
            )
            fig_error = px.bar(
                error_summary,
                x="Model",
                y="MAE (seconds)",
                title="Absolute Error Comparison on Saved Test Predictions",
                text="MAE (seconds)",
            )
            st.plotly_chart(fig_error, use_container_width=True)

            st.download_button(
                "Download test predictions CSV",
                data=predictions.to_csv(index=False).encode("utf-8"),
                file_name="test_predictions.csv",
                mime="text/csv",
            )
        else:
            st.warning(
                "Prediction CSV is missing expected columns. Check "
                "reports/test_predictions.csv."
            )
    else:
        st.warning("Test prediction file not found in reports/.")

# ------------------------------------------------------------
# Tab 5: Data explorer and data quality
# ------------------------------------------------------------
with data_tab:
    st.subheader("Explore Filtered Observations")
    preview_rows = st.slider(
        "Rows to preview", min_value=10, max_value=200, value=50, step=10
    )
    st.dataframe(
        filtered_df.head(preview_rows),
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        "Download filtered observations CSV",
        data=filtered_df.to_csv(index=False).encode("utf-8"),
        file_name="filtered_subway_observations.csv",
        mime="text/csv",
    )

    st.subheader("Data Quality Summary")
    quality = pd.DataFrame(
        {
            "Column": filtered_df.columns,
            "Missing Values": [
                filtered_df[column].isna().sum()
                for column in filtered_df.columns
            ],
            "Missing (%)": [
                round(100 * filtered_df[column].isna().mean(), 2)
                for column in filtered_df.columns
            ],
        }
    )
    st.dataframe(quality, use_container_width=True, hide_index=True)

# ------------------------------------------------------------
# Footer
# ------------------------------------------------------------
st.markdown("---")
st.caption(
    "NYC Subway Analytics | Python • Pandas • DuckDB • "
    "scikit-learn • Plotly • Streamlit"
)
