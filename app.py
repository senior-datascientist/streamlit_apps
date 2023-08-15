import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp, entropy
import plotly.express as px
import plotly.graph_objects as go

# Generate random data for demonstration
def generate_random_data(samples=1000, features=5):
    return pd.DataFrame(np.random.randn(samples, features), columns=[f"Feature {i}" for i in range(1, features+1)])

def detect_data_drift(data1, data2):
    p_values = []
    for col in data1.columns:
        p_value = ks_2samp(data1[col], data2[col]).pvalue
        p_values.append(p_value)
    return p_values

# Generate initial baseline data
baseline_data = generate_random_data()

# Generate new data for monitoring
new_data = generate_random_data()

# Create tabs in the sidebar
tabs = st.sidebar.tabs(label='Navigation', children=[
    st.sidebar.tab(label='Data Drift', children=[
        # Content for Data Drift tab
        st.title("Data Drift Monitoring App"),
        
        st.subheader("Baseline Data"),
        st.write(baseline_data),

        st.subheader("New Data"),
        st.write(new_data),

        st.subheader("Data Drift Detection"),

        # Detect data drift using Kolmogorov-Smirnov test
        p_values = detect_data_drift(baseline_data, new_data)
        threshold = 0.05

        for i, p_value in enumerate(p_values):
            st.write(f"Feature {i+1}:")
            if p_value < threshold:
                st.write("No significant drift detected.")
            else:
                st.write("Significant drift detected!")
    ]),
    st.sidebar.tab(label='Histograms', children=[
        # Content for Histograms tab
        st.title("Histograms"),
        
        # Visualization: Feature Drift using Plotly
        st.subheader("Feature Drift Visualization using Plotly"),

        for col in baseline_data.columns:
            df_concat = pd.concat([baseline_data[[col]], new_data[[col]]], keys=['Baseline', 'Incoming'], names=['Source'])

            fig = px.histogram(
                df_concat.reset_index(),  # Reset index to avoid issues with multi-index
                x=col, color='Source', nbins=30, opacity=0.7, barmode='overlay', title=f"{col} Distribution Comparison"
            )
            st.plotly_chart(fig)
    ]),
    st.sidebar.tab(label='Metrics', children=[
        # Content for Metrics tab
        st.title("Metrics"),
        
        # Visualization: Kolmogorov-Smirnov Distance and Jensen-Shannon Divergence
        st.subheader("Data Drift Metrics"),

        st.write("Kolmogorov-Smirnov Distance:"),
        st.write("Feature-wise p-values:", p_values),

        js_divergences = []
        for col in baseline_data.columns:
            p = np.concatenate([baseline_data[col], new_data[col]])
            q = np.concatenate([new_data[col], baseline_data[col]])
            js_divergence = 0.5 * (entropy(p, 0.5 * (p + q)) + entropy(q, 0.5 * (p + q)))
            js_divergences.append(js_divergence)
        st.write("Jensen-Shannon Divergence:", js_divergences)
    ])
])

# Display the selected tab's content
tabs

# Visualization: Line Chart for Data Drift Over Time
st.subheader("Data Drift Over Time")

time_steps = list(range(10))  # Convert range to list
data_drift_scores = np.random.rand(len(time_steps))

fig_line = go.Figure(data=go.Scatter(x=time_steps, y=data_drift_scores, mode='lines+markers'))
fig_line.update_layout(title="Data Drift Over Time", xaxis_title="Time", yaxis_title="Data Drift Score")
st.plotly_chart(fig_line)
