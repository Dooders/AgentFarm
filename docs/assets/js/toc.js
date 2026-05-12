(() => {
  const tocSidebar = document.querySelector("[data-toc-sidebar]");
  const tocList = document.querySelector("[data-toc-list]");
  const contentRoot = document.querySelector("[data-toc-content]");
  const tocPanel = tocSidebar ? tocSidebar.querySelector(".toc-sidebar__panel") : null;

  if (!tocSidebar || !tocList || !contentRoot) {
    return;
  }

  const headingSelector = "h2, h3, h4";
  const headings = Array.from(contentRoot.querySelectorAll(headingSelector));

  if (headings.length === 0) {
    return;
  }

  const slugCounts = new Map();
  const usedIds = new Set(
    Array.from(contentRoot.querySelectorAll("[id]"))
      .map((element) => element.id)
      .filter(Boolean)
  );
  const slugify = (value) =>
    value
      .toLowerCase()
      .trim()
      .replace(/[^a-z0-9\s-]/g, "")
      .replace(/\s+/g, "-")
      .replace(/-+/g, "-");

  for (const heading of headings) {
    if (heading.id) {
      continue;
    }

    const baseSlug = slugify(heading.textContent || "");
    if (!baseSlug) {
      continue;
    }

    const duplicateCount = slugCounts.get(baseSlug) || 0;
    slugCounts.set(baseSlug, duplicateCount + 1);

    let nextId = duplicateCount === 0 ? baseSlug : `${baseSlug}-${duplicateCount + 1}`;
    let suffix = duplicateCount + 2;
    while (usedIds.has(nextId)) {
      nextId = `${baseSlug}-${suffix}`;
      suffix += 1;
    }

    heading.id = nextId;
    usedIds.add(nextId);
  }

  const activeLinks = new Map();

  for (const heading of headings) {
    const id = heading.getAttribute("id");
    if (!id) {
      continue;
    }

    const item = document.createElement("li");
    const level = Number(heading.tagName.slice(1));
    item.className = "toc-sidebar__item";
    item.setAttribute("data-level", String(level));

    const link = document.createElement("a");
    link.className = "toc-sidebar__link";
    link.textContent = heading.textContent || id;
    link.href = `#${id}`;

    item.appendChild(link);
    tocList.appendChild(item);
    activeLinks.set(id, link);
  }

  if (activeLinks.size === 0) {
    return;
  }

  tocSidebar.hidden = false;
  if (tocPanel) {
    const desktopQuery = window.matchMedia("(min-width: 1040px)");
    let panelTouchedByUser = false;

    const applyDefaultPanelState = () => {
      if (panelTouchedByUser) {
        return;
      }
      tocPanel.open = desktopQuery.matches;
    };

    applyDefaultPanelState();
    tocPanel.addEventListener("toggle", () => {
      panelTouchedByUser = true;
    });
    desktopQuery.addEventListener("change", applyDefaultPanelState);
  }

  const updateActiveLink = () => {
    let activeId = "";
    for (const heading of headings) {
      if (heading.getBoundingClientRect().top - 120 <= 0) {
        activeId = heading.id;
      } else {
        break;
      }
    }

    if (!activeId) {
      const hash = window.location.hash.replace(/^#/, "");
      if (hash && activeLinks.has(hash)) {
        activeId = hash;
      } else {
        activeId = headings[0].id;
      }
    }

    for (const link of activeLinks.values()) {
      link.classList.remove("is-active");
    }

    const activeLink = activeLinks.get(activeId);
    if (activeLink) {
      activeLink.classList.add("is-active");
    }
  };

  updateActiveLink();
  window.addEventListener("scroll", updateActiveLink, { passive: true });

  window.addEventListener("hashchange", () => {
    const hash = window.location.hash.replace(/^#/, "");
    if (!hash || !activeLinks.has(hash)) {
      return;
    }
    for (const link of activeLinks.values()) {
      link.classList.remove("is-active");
    }
    activeLinks.get(hash).classList.add("is-active");
  });

  for (const link of activeLinks.values()) {
    link.addEventListener("click", (event) => {
      const id = link.getAttribute("href").replace(/^#/, "");
      const target = document.getElementById(id);
      if (!target) {
        return;
      }

      event.preventDefault();
      target.scrollIntoView({ behavior: "smooth", block: "start" });
      window.history.replaceState(null, "", `#${id}`);
    });
  }
})();
