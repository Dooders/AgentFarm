fetch('/static/family_tree.json')
    .then(response => response.json())
    .then(data => {
        const HUD_WIDTH = 200;
        const HUD_HEIGHT = 200;
        let isPushing = false;
        let cy = null;  // Declare cy at the top level
        let updateRootPosition = null;  // Declare updateRootPosition reference
        let lastPan = { x: 0, y: 0 };  // Initialize lastPan here

        // Initialize Cytoscape
        function initCytoscape() {
            cy = cytoscape({
                container: document.getElementById('cy'),
                elements: {
                    nodes: data.nodes.map(node => ({
                        data: { ...node },
                        grabbable: false
                    })),
                    edges: data.edges.map(edge => ({
                        data: { source: edge.source, target: edge.target, label: `Step ${edge.step}` }
                    }))
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
                        selector: 'node',
                        style: {
                            'label': function(ele) {
                                return ele.data('label') + '\n\n' + ele.data('details');
                            },
                            'text-valign': 'center',
                            'text-halign': 'center',
                            'text-wrap': 'wrap',
                            'background-color': function(ele) {
                                const zoom = ele.cy().zoom();
                                const baseColor = ele.data('color');
                                if (baseColor === '#ffffff') return baseColor; // Handle white case
                                
                                // Parse the original RGB values (excluding alpha)
                                const r = parseInt(baseColor.slice(1,3), 16);
                                const g = parseInt(baseColor.slice(3,5), 16);
                                const b = parseInt(baseColor.slice(5,7), 16);
                                
                                // More aggressive darkening based on zoom
                                const darknessFactor = Math.max(0.2, Math.min(1, Math.pow(zoom, 2)));
                                
                                // Adjust RGB values, keeping blue at 255
                                const newR = Math.floor(r * darknessFactor);
                                const newG = Math.floor(g * darknessFactor);
                                
                                // Return color in rgb() format instead of hex
                                return `rgb(${newR}, ${newG}, 255)`;
                            },
                            'text-valign': 'center',
                            'text-halign': 'center',
                            'text-wrap': 'wrap',
                            'shape': 'rectangle',
                            'width': '300px',
                            'height': '180px',
                            'padding': '20px',
                            'font-size': '20px',
                            'text-max-width': '280px',
                            'border-width': '2px',
                            'border-color': '#666',
                            'border-style': 'solid',
                            'text-margin-y': '8px',
                            'text-background-color': 'white',
                            'text-background-opacity': '0.9',
                            'text-background-padding': '8px',
                            'text-wrap': 'wrap',
                            'text-transform': 'none'
                        }
                    },
                    {
                        selector: 'edge',
                        style: {
                            'curve-style': 'unbundled-bezier',
                            'control-point-distances': [50],
                            'control-point-weights': [0.5],
                            'target-arrow-shape': 'triangle',
                            'label': 'data(label)',
                            'font-size': '12px',
                            'text-rotation': 'autorotate',
                            'text-margin-y': '-10px',
                            'line-color': '#fff',
                            'target-arrow-color': '#fff',
                            'width': 2,
                            'color': '#fff'
                        }
                    }
                ],
                panningEnabled: true,
                userPanningEnabled: true,
                minZoom: 0.01,
                maxZoom: 3,
            });
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
                    document.getElementById('root-position').textContent =
                        `Root: (${modelX}, ${modelY})`;
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
                
                const centerX = HUD_WIDTH + (availableWidth / 2);
                const centerY = HUD_HEIGHT + (availableHeight / 2);
                
                const widthRatio = availableWidth / bb.w;
                const heightRatio = availableHeight / bb.h;
                const newZoom = Math.min(widthRatio, heightRatio) * 0.8;
                
                cy.zoom(newZoom);
                cy.center();
                
                const currentPan = cy.pan();
                cy.pan({
                    x: currentPan.x + (HUD_WIDTH / 2),
                    y: currentPan.y + (HUD_HEIGHT / 2)
                });

                setTimeout(updateRootPosition, 50);
            };
        }

        // Initialize everything in sequence
        cy = initCytoscape();
        updateRootPosition = initUpdateRootPosition();
        const centerGraph = initCenterGraph();

        // Set up event handlers
        cy.on('pan', _.throttle(() => {
            const zoom = cy.zoom();
            const pan = cy.pan();
            const panDelta = {
                x: pan.x - lastPan.x,
                y: pan.y - lastPan.y
            };
            
            // Check if we're in the HUD zone
            const inHudZone = {
                x: -pan.x / zoom < HUD_WIDTH,
                y: -pan.y / zoom < HUD_HEIGHT
            };

            // If actively pushing against HUD (large pan delta), allow breakthrough
            if (Math.abs(panDelta.x) > 20 || Math.abs(panDelta.y) > 20) {
                isPushing = true;
                lastPan = pan;
                return;  // Allow the pan
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
        }, 16));

        cy.on('panend', () => {
            isPushing = false;
        });

        // Run initial layout
        cy.layout({
            name: 'dagre',
            rankDir: 'LR',
            nodeSep: 50,
            rankSep: 200,
            padding: 50,
            animate: false,
            spacingFactor: 1.0
        }).run();

        // Initial centering
        centerGraph();

        // Set up UI handlers
        document.getElementById('fit-button').addEventListener('click', centerGraph);

        const zoomDisplay = document.getElementById('zoom-level');
        cy.on('zoom', _.throttle(() => {
            zoomDisplay.textContent = `Zoom: ${cy.zoom().toFixed(2)}`;
        }, 100));

        // Add viewport change handlers
        cy.on('render', updateRootPosition);
        cy.on('pan', updateRootPosition);
        cy.on('zoom', updateRootPosition);
        cy.on('position', updateRootPosition);
        cy.on('layoutstop', updateRootPosition);
        cy.on('viewport', updateRootPosition);

        // Make updateRootPosition available globally
        cy.updateRootPosition = updateRootPosition;

        // Add node selection handlers for HUD overlay
        const hudOverlay = document.getElementById('hud-overlay');
        let selectedNode = null;

        // When a node is tapped, display agent details in the HUD overlay
        cy.on('tap', 'node', function(event) {
            const node = event.target;
            const data = node.data();
            
            // Update selected node
            if (selectedNode) {
                selectedNode.removeClass('selected');
            }
            selectedNode = node;
            node.addClass('selected');
            
            // Build details HTML with structure matching the CSS
            let detailsHtml = `<h3>Agent ${data.id}</h3>`;
            
            const stats = [
                ['Generation', data.generation],
                ['Type', data.agent_type],
                ['Birth Time', data.birth_time],
                ['Offspring', data.offspring_count],
                ['Resources', data.resources_consumed?.toFixed(1)]
            ];
            
            stats.forEach(([label, value]) => {
                if (value !== undefined) {
                    detailsHtml += `<p><strong>${label}:</strong><span>${value}</span></p>`;
                }
            });
            
            // Update HUD content and ensure it's visible
            hudOverlay.innerHTML = detailsHtml;
            hudOverlay.style.display = 'block';
            hudOverlay.style.opacity = '1';
            
            // Prevent event from bubbling to background
            event.stopPropagation();
        });

        // Clear the HUD overlay when tapping on the background
        cy.on('tap', function(event) {
            if (event.target === cy) {
                hudOverlay.style.opacity = '0';
                // Add a delay before hiding to allow fade out
                setTimeout(() => {
                    hudOverlay.style.display = 'none';
                }, 300);
                if (selectedNode) {
                    selectedNode.removeClass('selected');
                    selectedNode = null;
                }
            }
        });

        // Add CSS transition for smooth opacity changes
        hudOverlay.style.transition = 'opacity 0.3s ease-in-out';

        // Add style for selected nodes
        cy.style()
          .selector('node.selected')
          .style({
              'border-color': '#4af',
              'border-width': '3px',
              'border-style': 'solid'
          })
          .update();
    }); 