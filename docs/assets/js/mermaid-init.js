(() => {
  const mermaidBlocks = document.querySelectorAll("pre > code.language-mermaid");
  if (!mermaidBlocks.length || typeof window.mermaid === "undefined") {
    return;
  }

  window.mermaid.initialize({
    startOnLoad: false,
    securityLevel: "strict",
    flowchart: {
      htmlLabels: true,
      useMaxWidth: true,
      wrappingWidth: 220,
    },
  });

  const containers = [];
  mermaidBlocks.forEach((codeBlock, index) => {
    const pre = codeBlock.parentElement;
    if (!pre) {
      return;
    }

    const container = document.createElement("div");
    container.className = "mermaid";
    container.textContent = codeBlock.textContent || "";
    container.id = `mermaid-diagram-${index}`;
    pre.replaceWith(container);
    containers.push(container);
  });

  const modal = createModal();
  document.body.appendChild(modal.root);

  Promise.resolve(window.mermaid.run({ nodes: containers })).then(() => {
    containers.forEach((container) => enhanceDiagram(container, modal));
  });

  function enhanceDiagram(container, modal) {
    const svg = container.querySelector("svg");
    if (!svg) {
      return;
    }

    const figure = document.createElement("figure");
    figure.className = "mermaid-figure";

    const toolbar = document.createElement("div");
    toolbar.className = "mermaid-figure__toolbar";

    const expandBtn = makeButton("expand", "Expand diagram", "Expand");
    expandBtn.innerHTML =
      '<svg viewBox="0 0 24 24" width="16" height="16" aria-hidden="true" focusable="false">' +
      '<path fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" ' +
      'd="M4 9V4h5M20 9V4h-5M4 15v5h5M20 15v5h-5"/></svg>';
    toolbar.appendChild(expandBtn);

    const parent = container.parentElement;
    if (!parent) {
      return;
    }
    parent.insertBefore(figure, container);
    figure.appendChild(container);
    figure.appendChild(toolbar);

    const open = (origin) => modal.open(svg, origin);
    expandBtn.addEventListener("click", (event) => {
      event.stopPropagation();
      open(expandBtn);
    });

    container.addEventListener("click", (event) => {
      if (event.target.closest("a")) {
        return;
      }
      const selection = window.getSelection && window.getSelection();
      if (selection && selection.toString().length > 0) {
        return;
      }
      open(expandBtn);
    });

    container.setAttribute("role", "button");
    container.setAttribute("tabindex", "0");
    container.setAttribute("aria-label", "Open diagram in zoomable viewer");
    container.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        open(expandBtn);
      }
    });
  }

  function makeButton(variant, label, fallbackText) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = `mermaid-toolbar__btn mermaid-toolbar__btn--${variant}`;
    btn.setAttribute("aria-label", label);
    btn.title = label;
    btn.textContent = fallbackText;
    return btn;
  }

  function createModal() {
    const root = document.createElement("div");
    root.className = "mermaid-modal";
    root.setAttribute("role", "dialog");
    root.setAttribute("aria-modal", "true");
    root.setAttribute("aria-label", "Diagram viewer");
    root.hidden = true;

    const backdrop = document.createElement("div");
    backdrop.className = "mermaid-modal__backdrop";

    const dialog = document.createElement("div");
    dialog.className = "mermaid-modal__dialog";

    const toolbar = document.createElement("div");
    toolbar.className = "mermaid-modal__toolbar";

    const zoomIn = makeButton("zoom-in", "Zoom in", "+");
    const zoomOut = makeButton("zoom-out", "Zoom out", "\u2212");
    const reset = makeButton("reset", "Reset view", "Reset");
    const close = makeButton("close", "Close diagram viewer", "\u2715");

    const hint = document.createElement("span");
    hint.className = "mermaid-modal__hint";
    hint.textContent = "Drag to pan \u00B7 scroll to zoom \u00B7 Esc to close";

    toolbar.appendChild(zoomIn);
    toolbar.appendChild(zoomOut);
    toolbar.appendChild(reset);
    toolbar.appendChild(hint);
    toolbar.appendChild(close);

    const stage = document.createElement("div");
    stage.className = "mermaid-modal__stage";

    dialog.appendChild(toolbar);
    dialog.appendChild(stage);
    root.appendChild(backdrop);
    root.appendChild(dialog);

    let panZoom = null;
    let lastFocus = null;

    const onKey = (event) => {
      if (event.key === "Escape") {
        closeModal();
      }
    };

    const closeModal = () => {
      root.hidden = true;
      document.body.classList.remove("mermaid-modal-open");
      if (panZoom) {
        try {
          panZoom.destroy();
        } catch (e) {
          // ignore
        }
        panZoom = null;
      }
      stage.innerHTML = "";
      document.removeEventListener("keydown", onKey);
      if (lastFocus) {
        try {
          lastFocus.focus();
        } catch (e) {
          // ignore
        }
        lastFocus = null;
      }
    };

    const openModal = (sourceSvg, opener) => {
      lastFocus = opener || document.activeElement;
      stage.innerHTML = "";

      const clone = sourceSvg.cloneNode(true);
      clone.removeAttribute("id");
      clone.removeAttribute("style");
      clone.setAttribute("width", "100%");
      clone.setAttribute("height", "100%");
      clone.style.maxWidth = "100%";
      clone.style.maxHeight = "100%";
      clone.style.display = "block";
      stage.appendChild(clone);

      root.hidden = false;
      document.body.classList.add("mermaid-modal-open");
      document.addEventListener("keydown", onKey);

      requestAnimationFrame(() => {
        if (typeof window.svgPanZoom === "function") {
          panZoom = window.svgPanZoom(clone, {
            controlIconsEnabled: false,
            fit: true,
            center: true,
            minZoom: 0.2,
            maxZoom: 25,
            zoomScaleSensitivity: 0.4,
            contain: false,
          });
        }
        try {
          close.focus();
        } catch (e) {
          // ignore
        }
      });
    };

    backdrop.addEventListener("click", closeModal);
    close.addEventListener("click", closeModal);
    zoomIn.addEventListener("click", () => panZoom && panZoom.zoomIn());
    zoomOut.addEventListener("click", () => panZoom && panZoom.zoomOut());
    reset.addEventListener("click", () => {
      if (!panZoom) {
        return;
      }
      panZoom.resetZoom();
      panZoom.center();
      panZoom.fit();
    });

    return { root, open: openModal, close: closeModal };
  }
})();
