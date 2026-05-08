(() => {
  const mermaidBlocks = document.querySelectorAll("pre > code.language-mermaid");
  if (!mermaidBlocks.length || typeof window.mermaid === "undefined") {
    return;
  }

  window.mermaid.initialize({
    startOnLoad: false,
    securityLevel: "loose",
  });

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
  });

  window.mermaid.run();
})();
