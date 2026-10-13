// Plays the Long-Toolkit CLI header animation (exported from the Go CLI frames)
// inside any element carrying `data-ascii-header="<path to frames JSON>"`.
(() => {
  const BOUND_ATTR = "data-ascii-header-bound";
  const cache = new Map();

  function loadFrames(url) {
    if (!cache.has(url)) {
      cache.set(
        url,
        fetch(url).then((response) => {
          if (!response.ok) {
            throw new Error(`Cannot load ${url}`);
          }
          return response.json();
        })
      );
    }
    return cache.get(url);
  }

  function escapeHtml(text) {
    return text
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");
  }

  function expandRuns(runs, width) {
    const cells = new Array(width).fill("");
    for (const [start, length, key] of runs || []) {
      for (let x = start; x < start + length && x < width; x += 1) {
        cells[x] = key;
      }
    }
    return cells;
  }

  function renderFrame(frame, width) {
    return frame.rows
      .map((row, y) => {
        const padded = row.padEnd(width, " ");
        const chars = Array.from(padded);
        const fg = expandRuns(frame.fg[y], chars.length);
        const bg = expandRuns(frame.bg[y], chars.length);
        let html = "";
        let start = 0;
        for (let x = 1; x <= chars.length; x += 1) {
          if (x < chars.length && fg[x] === fg[start] && bg[x] === bg[start]) {
            continue;
          }
          // Full blocks are painted as backgrounds so rows join without gaps.
          const text = escapeHtml(chars.slice(start, x).join("")).replace(
            /█+/g,
            (blocks) => `<span class="ascii-header__block">${blocks}</span>`
          );
          const styles = [];
          if (fg[start]) styles.push(`color:var(--ascii-${fg[start]})`);
          if (bg[start]) styles.push(`background:var(--ascii-${bg[start]})`);
          html += styles.length ? `<span style="${styles.join(";")}">${text}</span>` : text;
          start = x;
        }
        return html;
      })
      .join("\n");
  }

  function applyPalette(host, data) {
    const scheme = document.body.getAttribute("data-md-color-scheme") === "default" ? "light" : "dark";
    const palette = data.palettes[scheme] || data.palettes.dark;
    for (const [key, value] of Object.entries(palette)) {
      host.style.setProperty(`--ascii-${key}`, value);
    }
  }

  function fitFont(host, screen, data) {
    const width = host.clientWidth;
    if (!width) return;
    // Monospace glyphs are roughly 0.6em wide.
    const size = Math.max(4, Math.min(14, width / (data.width * 0.6)));
    screen.style.fontSize = `${size}px`;
  }

  function mount(host) {
    if (host.hasAttribute(BOUND_ATTR)) return;
    host.setAttribute(BOUND_ATTR, "true");
    const url = host.getAttribute("data-ascii-header");
    const screen = document.createElement("pre");
    screen.className = "ascii-header__screen";
    screen.setAttribute("aria-hidden", "true");
    host.appendChild(screen);

    loadFrames(url)
      .then((data) => {
        const rendered = data.frames.map((frame) => renderFrame(frame, data.width));
        applyPalette(host, data);
        fitFont(host, screen, data);

        const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
        let index = 0;
        screen.innerHTML = rendered[index];
        if (!reduceMotion) {
          const tick = () => {
            if (!host.isConnected) return;
            index = (index + 1) % rendered.length;
            screen.innerHTML = rendered[index];
            window.setTimeout(tick, data.frames[index].duration);
          };
          window.setTimeout(tick, data.frames[0].duration);
        }

        new MutationObserver(() => applyPalette(host, data)).observe(document.body, {
          attributes: true,
          attributeFilter: ["data-md-color-scheme"],
        });
        if (typeof ResizeObserver !== "undefined") {
          new ResizeObserver(() => fitFont(host, screen, data)).observe(host);
        }
      })
      .catch(() => {
        host.classList.add("ascii-header--unavailable");
      });
  }

  function init(root = document) {
    root.querySelectorAll("[data-ascii-header]").forEach(mount);
  }

  if (typeof window.document$ !== "undefined" && typeof window.document$.subscribe === "function") {
    window.document$.subscribe(() => init(document));
  } else if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", () => init(document));
  } else {
    init(document);
  }
})();
