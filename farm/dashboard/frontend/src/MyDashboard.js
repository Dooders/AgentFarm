import { Chart } from "chart.js";
import React, { useEffect } from "react";
import { withStreamlitConnection } from "streamlit-component-lib";

const MyDashboard = ({ args }) => {
  useEffect(() => {
    // Render Resource and Health Trends
    const ctx1 = document.getElementById("resource-health-chart");
    new Chart(ctx1, {
      type: "line",
      data: {
        labels: args.stepNumbers, // Step numbers from Streamlit
        datasets: [
          {
            label: "Resources",
            data: args.resourceLevels, // Resource levels from Streamlit
            borderColor: "blue",
            fill: false,
          },
          {
            label: "Health",
            data: args.healthLevels, // Health levels from Streamlit
            borderColor: "green",
            fill: false,
          },
        ],
      },
    });

    // Render Action Distribution
    const ctx2 = document.getElementById("action-distribution-chart");
    new Chart(ctx2, {
      type: "bar",
      data: {
        labels: args.actionTypes, // Action types from Streamlit
        datasets: [
          {
            label: "Action Frequency",
            data: args.actionFrequencies, // Action frequencies from Streamlit
            backgroundColor: "orange",
          },
        ],
      },
    });
  }, [args]);

  return (
    <div>
      <h3>Resource and Health Trends</h3>
      <canvas id="resource-health-chart"></canvas>
      <h3>Action Distribution</h3>
      <canvas id="action-distribution-chart"></canvas>
    </div>
  );
};

export default withStreamlitConnection(MyDashboard);
