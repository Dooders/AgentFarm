fetch("/static/family_tree.json")
  .then((response) => response.json())
  .then((data) => {
    const HUD_WIDTH = 200;
    const HUD_HEIGHT = 200;
    let isPushing = false;
    let cy = null;
    let updateRootPosition = null;
    let lastPan = { x: 0, y: 0 };
    let currentColorBy = "offspring";

    function getNodeColor(node, colorBy) {
      if (!cy) return "#ffffff";

      const data = node.data();
      let intensity = 0;

      try {
        switch (colorBy) {
          case "offspring":
            const maxOffspring = Math.max(
              ...cy.nodes().map((n) => n.data("offspring_count") || 0)
            );
            intensity =
              maxOffspring > 0 ? (data.offspring_count || 0) / maxOffspring : 0;
            break;

          case "resources":
            const maxResources = Math.max(
              ...cy.nodes().map((n) => n.data("resources_consumed") || 0)
            );
            intensity =
              maxResources > 0
                ? (data.resources_consumed || 0) / maxResources
                : 0;
            break;

          case "age":
            const maxAge = Math.max(
              ...cy.nodes().map((n) => n.data("birth_time") || 0)
            );
            intensity = maxAge > 0 ? (data.birth_time || 0) / maxAge : 0;
            break;
        }
      } catch (e) {
        console.warn("Error calculating node color:", e);
        return "#ffffff";
      }

      const rgb_value = Math.max(
        0,
        Math.min(255, Math.round(255 * (1 - intensity * 0.3)))
      );
      return intensity === 0
        ? "#ffffff"
        : `#${rgb_value.toString(16).padStart(2, "0")}${rgb_value
            .toString(16)
            .padStart(2, "0")}ff`;
    }

    function updateNodeColors(colorBy) {
      if (!cy) return;
      currentColorBy = colorBy;
      cy.nodes().forEach((node) => {
        node.style("background-color", getNodeColor(node, colorBy));
      });
    }

    // Initialize Cytoscape
    function initCytoscape() {
      cy = cytoscape({
        container: document.getElementById("cy"),
        elements: {
          nodes: data.nodes.map((node) => ({
            data: { ...node },
            grabbable: false,
          })),
          edges: data.edges.map((edge) => ({
            data: {
              source: edge.source,
              target: edge.target,
              label: `Step ${edge.step}`,
            },
          })),
        },
        userPanningEnabled: true,
        userZoomingEnabled: true,
        boxSelectionEnabled: false,
        autounselectify: true,
        wheelSensitivity: 0.2,
        minZoom: 0.01,
        maxZoom: 3,
        style: [
          {
            selector: "node",
            style: {
              label: function (ele) {
                return ele.data("label") + "\n\n" + ele.data("details");
              },
              "text-valign": "center",
              "text-halign": "center",
              "text-wrap": "wrap",
              "background-color": "#ffffff",
              "text-valign": "center",
              "text-halign": "center",
              "text-wrap": "wrap",
              shape: "rectangle",
              width: "300px",
              height: "180px",
              padding: "20px",
              "font-size": "20px",
              "text-max-width": "280px",
              "border-width": "2px",
              "border-color": "#666",
              "border-style": "solid",
              "text-margin-y": "8px",
              "text-background-color": "white",
              "text-background-opacity": "0.9",
              "text-background-padding": "8px",
              "text-wrap": "wrap",
              "text-transform": "none",
              "text-opacity": 0,
              "text-opacity-depends-on-zoom": true,
              "text-opacity-min": 0.15,
              "text-opacity-max": 0.4,
            },
          },
          {
            selector: "edge",
            style: {
              "curve-style": "unbundled-bezier",
              "control-point-distances": [50],
              "control-point-weights": [0.5],
              "target-arrow-shape": "triangle",
              label: "data(label)",
              "font-size": "12px",
              "text-rotation": "autorotate",
              "text-margin-y": "-10px",
              "line-color": "#fff",
              "target-arrow-color": "#fff",
              width: 2,
              color: "#fff",
            },
          },
        ],
        panningEnabled: true,
        userPanningEnabled: true,
        minZoom: 0.01,
        maxZoom: 3,
      });

      // Apply colors after cy is initialized
      setTimeout(() => updateNodeColors(currentColorBy), 0);

      // Add zoom-based label visibility
      cy.on('zoom', _.throttle(() => {
        const zoom = cy.zoom();
        const minZoom = 0.15;
        const maxZoom = 0.4;
        
        cy.nodes().forEach(node => {
          if (zoom < minZoom) {
            node.style('text-opacity', 0);
          } else if (zoom > maxZoom) {
            node.style('text-opacity', 1);
          } else {
            // Linear interpolation between min and max zoom
            const opacity = (zoom - minZoom) / (maxZoom - minZoom);
            node.style('text-opacity', opacity);
          }
        });
      }, 100));

      return cy;
    }

    // Initialize the root position update function
    function initUpdateRootPosition() {
      updateRootPosition = _.throttle(() => {
        const rootNode = cy.nodes()[0];
        if (rootNode) {
          const renderedPosition = rootNode.renderedPosition();
          const zoom = cy.zoom();
          const modelX = Math.round(renderedPosition.x / zoom);
          const modelY = Math.round(renderedPosition.y / zoom);
          document.getElementById(
            "root-position"
          ).textContent = `Root: (${modelX}, ${modelY})`;
        }
      }, 16);
      return updateRootPosition;
    }

    // Initialize the center graph function
    function initCenterGraph() {
      return () => {
        const bb = cy.elements().boundingBox();
        const availableWidth = cy.width() - HUD_WIDTH;
        const availableHeight = cy.height() - HUD_HEIGHT;

        const centerX = HUD_WIDTH + availableWidth / 2;
        const centerY = HUD_HEIGHT + availableHeight / 2;

        const widthRatio = availableWidth / bb.w;
        const heightRatio = availableHeight / bb.h;
        const newZoom = Math.min(widthRatio, heightRatio) * 0.8;

        cy.zoom(newZoom);
        cy.center();

        const currentPan = cy.pan();
        cy.pan({
          x: currentPan.x + HUD_WIDTH / 2,
          y: currentPan.y + HUD_HEIGHT / 2,
        });

        setTimeout(updateRootPosition, 50);
      };
    }

    // Initialize everything in sequence
    cy = initCytoscape();
    updateRootPosition = initUpdateRootPosition();
    const centerGraph = initCenterGraph();

    // Set up event handlers
    cy.on(
      "pan",
      _.throttle(() => {
        const zoom = cy.zoom();
        const pan = cy.pan();
        const panDelta = {
          x: pan.x - lastPan.x,
          y: pan.y - lastPan.y,
        };

        // Check if we're in the HUD zone
        const inHudZone = {
          x: -pan.x / zoom < HUD_WIDTH,
          y: -pan.y / zoom < HUD_HEIGHT,
        };

        // If actively pushing against HUD (large pan delta), allow breakthrough
        if (Math.abs(panDelta.x) > 20 || Math.abs(panDelta.y) > 20) {
          isPushing = true;
          lastPan = pan;
          return; // Allow the pan
        }

        // If not pushing and in HUD zone, apply soft resistance
        if (!isPushing && (inHudZone.x || inHudZone.y)) {
          const newPan = { ...pan };
          if (inHudZone.x) {
            newPan.x = -HUD_WIDTH * zoom;
          }
          if (inHudZone.y) {
            newPan.y = -HUD_HEIGHT * zoom;
          }
          cy.pan(newPan);
        }
        lastPan = pan;
      }, 16)
    );

    cy.on("panend", () => {
      isPushing = false;
    });

    // Run initial layout
    cy.layout({
      name: "dagre",
      rankDir: "LR",
      nodeSep: 50,
      rankSep: 200,
      padding: 50,
      animate: false,
      spacingFactor: 1.0,
    }).run();

    // Initial centering
    centerGraph();

    // Set up UI handlers
    document
      .getElementById("fit-button")
      .addEventListener("click", centerGraph);

    const zoomDisplay = document.getElementById("zoom-level");
    cy.on(
      "zoom",
      _.throttle(() => {
        zoomDisplay.textContent = `Zoom: ${cy.zoom().toFixed(2)}`;
      }, 100)
    );

    // Add viewport change handlers
    cy.on("render", updateRootPosition);
    cy.on("pan", updateRootPosition);
    cy.on("zoom", updateRootPosition);
    cy.on("position", updateRootPosition);
    cy.on("layoutstop", updateRootPosition);
    cy.on("viewport", updateRootPosition);

    // Make updateRootPosition available globally
    cy.updateRootPosition = updateRootPosition;

    // Add node selection handlers for HUD overlay
    const hudOverlay = document.getElementById("hud-overlay");
    let selectedNode = null;

    // When a node is tapped, display agent details in the HUD overlay
    cy.on("tap", "node", function (event) {
      const node = event.target;
      const data = node.data();

      // Update selected node
      if (selectedNode) {
        selectedNode.removeClass("selected");
      }
      selectedNode = node;
      node.addClass("selected");

      // Build details HTML with structure matching the CSS
      let detailsHtml = `<h3>Agent Details</h3>`;

      const stats = [
        ["ID", data.id],
        ["Type", data.agent_type],
        ["Age", data.age],
        ["Birth Time", data.birth_time],
        ["Death Time", data.death_time || "Still alive"],
        ["Offspring", data.offspring_count],
        ["Total Resources", data.resources_consumed?.toFixed(2) || "0.00"],
        ["Avg Resources", data.avg_resources?.toFixed(2) || "0.00"],
        ["Generation", data.generation],
      ];

      stats.forEach(([label, value]) => {
        if (value !== undefined) {
          detailsHtml += `<p><strong>${label}:</strong> ${value}</p>`;
        }
      });

      // Update HUD content and ensure it's visible
      hudOverlay.innerHTML = detailsHtml;
      hudOverlay.style.display = "block";
      hudOverlay.style.opacity = "1";

      // Prevent event from bubbling to background
      event.stopPropagation();
    });

    // Clear the HUD overlay when tapping on the background
    cy.on("tap", function (event) {
      if (event.target === cy) {
        hudOverlay.style.opacity = "0";
        // Add a delay before hiding to allow fade out
        setTimeout(() => {
          hudOverlay.style.display = "none";
        }, 300);
        if (selectedNode) {
          selectedNode.removeClass("selected");
          selectedNode = null;
        }
      }
    });

    // Add CSS transition for smooth opacity changes
    hudOverlay.style.transition = "opacity 0.3s ease-in-out";

    // Add style for selected nodes
    cy.style()
      .selector("node.selected")
      .style({
        "border-color": "#4af",
        "border-width": "3px",
        "border-style": "solid",
      })
      .update();

    // Add this after all the other event handlers
    document.getElementById("color-by").addEventListener("change", (event) => {
      updateNodeColors(event.target.value);
    });
  });
