"""
Observation rendering utilities for AgentObservation.

Provides minimalist, high-detail rendering of multichannel observation grids
and an optional interactive HTML viewer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import io
import json
import math

import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    # Colormap support for continuous channels
    import matplotlib.cm as mpl_cm
    import matplotlib.colors as mpl_colors
except Exception:  # pragma: no cover - optional dependency at runtime
    mpl_cm = None
    mpl_colors = None


RgbTuple = Tuple[int, int, int]


def _hex_to_rgb(color_hex: str) -> RgbTuple:
    color_hex = color_hex.lstrip("#")
    if len(color_hex) == 3:
        color_hex = "".join([c * 2 for c in color_hex])
    r = int(color_hex[0:2], 16)
    g = int(color_hex[2:4], 16)
    b = int(color_hex[4:6], 16)
    return (r, g, b)


def _ensure_numpy01(array_like: Union[np.ndarray, "torch.Tensor"]) -> np.ndarray:
    try:
        import torch
        if isinstance(array_like, torch.Tensor):
            if array_like.is_cuda:
                array_like = array_like.detach().cpu()
            array_like = array_like.detach().to(dtype=torch.float32)
            npy = array_like.numpy()
        else:
            npy = np.asarray(array_like, dtype=np.float32)
    except Exception:
        # Fallback without torch import
        npy = np.asarray(array_like, dtype=np.float32)
    # Normalize/clamp to [0,1]
    npy = np.nan_to_num(npy, nan=0.0, posinf=1.0, neginf=0.0)
    npy = np.clip(npy, 0.0, 1.0)
    return npy


@dataclass

    def __post_init__(self):
        if not (0.0 <= self.alpha <= 1.0):
            raise ValueError(f"ChannelStyle.alpha must be in [0, 1], got {self.alpha}")

def get_default_palette(channel_names: List[str]) -> Dict[str, ChannelStyle]:
    # Colorblind-friendly inspired palette
    default: Dict[str, ChannelStyle] = {
        "SELF_HP": ChannelStyle("#00e5ff", 0.95, None),
        "ALLIES_HP": ChannelStyle("#00c853", 0.85, None),
        "ENEMIES_HP": ChannelStyle("#ff1744", 0.85, None),
        "RESOURCES": ChannelStyle("#2ecc71", 0.60, None),
        "OBSTACLES": ChannelStyle("#9e9e9e", 0.65, None),
        "TERRAIN_COST": ChannelStyle(None, 0.90, "magma"),
        "VISIBILITY": ChannelStyle("#ffffff", 0.30, None),
        "KNOWN_EMPTY": ChannelStyle("#90a4ae", 0.15, None),
        "DAMAGE_HEAT": ChannelStyle("#ff9100", 0.75, None),
        "TRAILS": ChannelStyle("#00b8d4", 0.55, None),
        "ALLY_SIGNAL": ChannelStyle("#ffd600", 0.45, None),
        "GOAL": ChannelStyle("#d500f9", 0.95, None),
        "LANDMARKS": ChannelStyle("#7e57c2", 0.80, None),
    }
    # Add fallback for any unknown/custom channels
    fallback_colors = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    ]
    idx = 0
    for name in channel_names:
        if name not in default:
            color = fallback_colors[idx % len(fallback_colors)]
            default[name] = ChannelStyle(color, 0.6, None)
            idx += 1
    return default


def _render_channels_overlay(
    gridC01: np.ndarray,  # (C,H,W) in [0,1]
    channel_names: List[str],
    palette: Dict[str, ChannelStyle],
    background: str,
) -> np.ndarray:
    C, H, W = gridC01.shape
    base = np.zeros((H, W, 3), dtype=np.float32)
    bg_rgb = np.array(_hex_to_rgb(background), dtype=np.float32) / 255.0
    base[:] = bg_rgb

    for idx, name in enumerate(channel_names):
        layer = gridC01[idx]  # (H,W)
        style = palette.get(name, ChannelStyle("#ffffff", 0.5, None))

        if style.cmap and mpl_cm is not None:
            cmap = mpl_cm.get_cmap(style.cmap)
            rgba = cmap(layer)  # (H,W,4), already 0..1
            alpha = np.clip(rgba[..., 3] * style.alpha, 0.0, 1.0)
            color_rgb = rgba[..., :3]
        else:
            color_rgb = np.array(_hex_to_rgb(style.color or "#ffffff"), dtype=np.float32) / 255.0
            # Use value as intensity for alpha scaling to emphasize higher values
            alpha = np.clip(layer * style.alpha, 0.0, 1.0)
            # Also modulate color intensity by value to keep minimalist look
            color_rgb = color_rgb[None, None, :] * layer[..., None]

        # Alpha blend: out = src * a + dst * (1-a)
        base = color_rgb * alpha[..., None] + base * (1.0 - alpha[..., None])

    return np.clip(base, 0.0, 1.0)


def _nearest_scale(img_rgb01: np.ndarray, size: int) -> Image.Image:
    H, W = img_rgb01.shape[:2]
    # Decide target keeping square, use longest side = size
    scale = max(1, int(math.ceil(size / max(H, W))))
    target = (W * scale, H * scale)
    pil = Image.fromarray((img_rgb01 * 255.0).astype(np.uint8), mode="RGB")
    return pil.resize(target, resample=Image.NEAREST)


def _draw_grid_and_center(
    image: Image.Image, draw_grid: bool, draw_center: bool, cell_size: int
) -> None:
    if not (draw_grid or draw_center):
        return
    draw = ImageDraw.Draw(image)
    W, H = image.size
    if draw_grid and cell_size >= 8:
        # subtle grid lines
        grid_color = (255, 255, 255, 32)
        for x in range(0, W, cell_size):
            draw.line([(x, 0), (x, H)], fill=grid_color)
        for y in range(0, H, cell_size):
            draw.line([(0, y), (W, y)], fill=grid_color)
    if draw_center:
        cx = (W // (2 * cell_size)) * cell_size + cell_size // 2
        cy = (H // (2 * cell_size)) * cell_size + cell_size // 2
        cross_color = (255, 255, 255, 128)
        draw.line([(0, cy), (W, cy)], fill=cross_color)
        draw.line([(cx, 0), (cx, H)], fill=cross_color)


def _render_gallery(
    gridC01: np.ndarray,
    channel_names: List[str],
    palette: Dict[str, ChannelStyle],
    background: str,
    size: int,
) -> Image.Image:
    C, H, W = gridC01.shape
    cols = int(math.ceil(math.sqrt(C)))
    rows = int(math.ceil(C / cols))
    # Render each as overlay of a single channel for consistency
    tiles: List[Image.Image] = []
    for idx, name in enumerate(channel_names):
        single = gridC01[idx:idx+1, :, :]
        img01 = _render_channels_overlay(single, [name], palette, background)
        tile = _nearest_scale(img01, size // max(cols, rows))
        # label
        draw = ImageDraw.Draw(tile)
        label_bg = (0, 0, 0, 128)
        text = name
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None
        draw.rectangle([(2, 2), (2 + 6 * len(text), 16)], fill=label_bg)
        draw.text((4, 4), text, fill=(255, 255, 255), font=font)
        tiles.append(tile)

    tile_w, tile_h = tiles[0].size
    grid_img = Image.new("RGB", (tile_w * cols, tile_h * rows), color=_hex_to_rgb(background))
    for i, tile in enumerate(tiles):
        r = i // cols
        c = i % cols
        grid_img.paste(tile, (c * tile_w, r * tile_h))
    return grid_img


class ObservationRenderer:
    @staticmethod
    def render(
        observation: Union[np.ndarray, "torch.Tensor"],
        channel_names: List[str],
        mode: str = "overlay",
        size: int = 256,
        palette: Optional[Dict[str, ChannelStyle]] = None,
        grid: bool = False,
        legend: bool = False,  # reserved for future use
        background: str = "#111213",
        draw_center: bool = True,
        return_type: str = "pil",
    ) -> Union[Image.Image, np.ndarray]:
        gridC01 = _ensure_numpy01(observation)
        if gridC01.ndim != 3:
            raise ValueError("observation must have shape (C,H,W)")
        C = gridC01.shape[0]
        if len(channel_names) != C:
            raise ValueError("channel_names length must match observation channels")

        style_palette = palette or get_default_palette(channel_names)

        if mode == "overlay":
            img01 = _render_channels_overlay(gridC01, channel_names, style_palette, background)
            img = _nearest_scale(img01, size)
            cell_size = max(1, img.size[0] // gridC01.shape[2])
            _draw_grid_and_center(img, grid, draw_center, cell_size)
        elif mode == "gallery":
            img = _render_gallery(gridC01, channel_names, style_palette, background, size)
        else:
            raise ValueError("mode must be 'overlay' or 'gallery'")

        if return_type == "pil":
            return img
        if return_type == "numpy":
            return np.asarray(img)
        if return_type == "bytes":
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return buf.getvalue()
        raise ValueError("return_type must be 'pil', 'numpy', or 'bytes'")

    @staticmethod
    def to_interactive_json(
        observation: Union[np.ndarray, "torch.Tensor"],
        channel_names: List[str],
        meta: Optional[Dict[str, Union[int, float, str]]] = None,
    ) -> Dict:
        gridC01 = _ensure_numpy01(observation)
        C, H, W = gridC01.shape
        payload = {
            "shape": [int(C), int(H), int(W)],
            "channels": list(channel_names),
            "grid": gridC01.tolist(),
            "meta": meta or {},
        }
        return payload

    @staticmethod
    def render_interactive_html(
        observation: Union[np.ndarray, "torch.Tensor"],
        channel_names: List[str],
        outfile: Optional[str] = None,
        title: str = "Observation Viewer",
        background: str = "#0b0c0f",
        palette: Optional[Dict[str, ChannelStyle]] = None,
        initial_scale: int = 16,
    ) -> str:
        data = ObservationRenderer.to_interactive_json(observation, channel_names)
        palette_dict = {
            name: {"color": cs.color, "alpha": cs.alpha, "cmap": cs.cmap}
            for name, cs in (palette or get_default_palette(channel_names)).items()
        }
        html = _build_interactive_html(data, title, background, palette_dict, initial_scale)
        if outfile:
            with open(outfile, "w", encoding="utf-8") as f:
                f.write(html)
        return html


def _build_interactive_html(
    data: Dict,
    title: str,
    background: str,
    palette: Dict[str, Dict],
    initial_scale: int,
) -> str:
    # Minimal, shadcn-inspired styling with a clean dark theme.
    json_data = json.dumps(data)
    json_palette = json.dumps(palette)
    title_placeholder = title
    background_placeholder = background
    initial_scale_placeholder = str(int(initial_scale))
    html = """<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>__TITLE__</title>
  <style>
    :root {{
      --bg: __BACKGROUND__;
      --fg: #e5e7eb;
      --muted: #a1a1aa;
      --card: #111217;
      --accent: #3b82f6;
      --ring: #60a5fa;
      --border: #1f2937;
    }}
    html, body {{ height: 100%; margin: 0; background: var(--bg); color: var(--fg); font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, Noto Sans, Apple Color Emoji, Segoe UI Emoji; }}
    .container {{ max-width: 1200px; margin: 0 auto; padding: 16px; }}
    .card {{ background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 12px; box-shadow: 0 1px 2px rgba(0,0,0,0.2); }}
    .row {{ display: flex; gap: 12px; align-items: stretch; }}
    .col {{ flex: 1; min-width: 0; }}
    .panel {{ display: flex; gap: 8px; align-items: center; flex-wrap: wrap; margin-bottom: 8px; }}
    .select, .slider {{ appearance: none; background: var(--bg); color: var(--fg); border: 1px solid var(--border); border-radius: 8px; padding: 8px; }}
    .btn {{ background: #0f172a; color: var(--fg); border: 1px solid var(--border); border-radius: 8px; padding: 8px 10px; cursor: pointer; }}
    .btn:focus {{ outline: 2px solid var(--ring); outline-offset: 2px; }}
    .pill {{ padding: 4px 8px; border: 1px solid var(--border); border-radius: 999px; cursor: pointer; font-size: 12px; }}
    .pill.active {{ background: var(--accent); border-color: var(--accent); color: white; }}
    .legend {{ display: flex; gap: 8px; flex-wrap: wrap; margin-top: 8px; }}
    .legend-item {{ display: inline-flex; gap: 6px; align-items: center; font-size: 12px; color: var(--muted); }}
    .swatch {{ width: 12px; height: 12px; border-radius: 3px; border: 1px solid var(--border); }}
    #canvas-wrap {{ position: relative; }}
    #tooltip {{ position: absolute; pointer-events: none; background: rgba(17,18,23,0.95); color: var(--fg); border: 1px solid var(--border); border-radius: 8px; padding: 6px 8px; font-size: 12px; display: none; min-width: 160px; }}
    .kbd {{ font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, \"Liberation Mono\", monospace; background: #0b1220; border: 1px solid var(--border); border-radius: 6px; padding: 1px 4px; font-size: 11px; color: var(--muted); }}
  </style>
</head>
<body>
  <div class=\"container\">
    <div class=\"row\" style=\"gap: 16px\"> 
      <div class=\"col\">
        <div class=\"card\">
          <div class=\"panel\">
            <select id=\"channelSelect\" class=\"select\"></select>
            <button id=\"prevBtn\" class=\"btn\">Prev (<span class=\"kbd\">[</span>)</button>
            <button id=\"nextBtn\" class=\"btn\">Next (<span class=\"kbd\">]</span>)</button>
            <label class=\"pill\"><input type=\"checkbox\" id=\"overlayToggle\" checked /> Overlay</label>
            <label class=\"pill\"><input type=\"checkbox\" id=\"gridToggle\" /> Grid</label>
            <label class=\"pill\"><input type=\"checkbox\" id=\"centerToggle\" checked /> Center</label>
            <label style=\"margin-left:8px\">Scale <input type=\"range\" id=\"scale\" class=\"slider\" min=\"4\" max=\"48\" step=\"1\" value=\"__INITIAL_SCALE__\" /></label>
          </div>
          <div id=\"canvas-wrap\" class=\"card\" style=\"padding:8px\">
            <canvas id=\"canvas\"></canvas>
            <div id=\"tooltip\"></div>
          </div>
          <div class=\"legend\" id=\"legend\"></div>
        </div>
      </div>
    </div>
  </div>
  <script id=\"obs-data\" type=\"application/json\">__DATA__</script>
  <script id=\"palette\" type=\"application/json\">__PALETTE__</script>
  <script>
    const data = JSON.parse(document.getElementById('obs-data').textContent);
    const palette = JSON.parse(document.getElementById('palette').textContent);
    const C = data.shape[0], H = data.shape[1], W = data.shape[2];
    const channels = data.channels;
    const grid = data.grid; // C x H x W
    let scale = parseInt(document.getElementById('scale').value, 10) || 16;
    let overlay = true;
    let showGrid = false;
    let showCenter = true;
    let currentIndex = 0;

    const canvas = document.getElementById('canvas');
    const ctx = canvas.getContext('2d');
    const tooltip = document.getElementById('tooltip');

    function setCanvasSize() {
      canvas.width = W * scale;
      canvas.height = H * scale;
    }

    function hexToRgb(hex) {
      hex = hex.replace('#','');
      if (hex.length === 3) hex = hex.split('').map(c => c + c).join('');
      const num = parseInt(hex, 16);
      return { r: (num >> 16) & 255, g: (num >> 8) & 255, b: num & 255 };
    }

    function drawGrid() {
      if (!showGrid || scale < 8) return;
      ctx.save();
      ctx.strokeStyle = 'rgba(255,255,255,0.12)';
      ctx.lineWidth = 1;
      for (let x = 0; x <= W; x++) {
        ctx.beginPath(); ctx.moveTo(x*scale + 0.5, 0); ctx.lineTo(x*scale + 0.5, H*scale); ctx.stroke();
      }
      for (let y = 0; y <= H; y++) {
        ctx.beginPath(); ctx.moveTo(0, y*scale + 0.5); ctx.lineTo(W*scale, y*scale + 0.5); ctx.stroke();
      }
      ctx.restore();
    }

    function drawCenter() {
      if (!showCenter) return;
      ctx.save();
      ctx.strokeStyle = 'rgba(255,255,255,0.5)';
      ctx.lineWidth = 1;
      const cx = Math.floor(W/2)*scale + Math.floor(scale/2) + 0.5;
      const cy = Math.floor(H/2)*scale + Math.floor(scale/2) + 0.5;
      ctx.beginPath(); ctx.moveTo(0, cy); ctx.lineTo(W*scale, cy); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(cx, 0); ctx.lineTo(cx, H*scale); ctx.stroke();
      ctx.restore();
    }

    function lerp(a,b,t){ return a + (b-a)*t; }

    function drawSingle(idx) {
      const imgData = ctx.createImageData(W*scale, H*scale);
      const channelName = channels[idx];
      const pal = palette[channelName] || { color: '#ffffff', alpha: 0.6 };
      const base = hexToRgb('__BACKGROUND__');
      // Pre-fill background
      for (let i=0;i<imgData.data.length;i+=4){
        imgData.data[i] = base.r; imgData.data[i+1] = base.g; imgData.data[i+2] = base.b; imgData.data[i+3] = 255;
      }
      const layer = grid[idx];
      const color = hexToRgb(pal.color || '#ffffff');
      for (let y=0;y<H;y++){
        for (let x=0;x<W;x++){
          const v = Math.max(0, Math.min(1, layer[y][x]));
          const a = Math.max(0, Math.min(1, v * (pal.alpha ?? 0.6)));
          const r = Math.round(lerp(base.r, color.r * v, a));
          const g = Math.round(lerp(base.g, color.g * v, a));
          const b = Math.round(lerp(base.b, color.b * v, a));
          for (let sy=0; sy<scale; sy++){
            for (let sx=0; sx<scale; sx++){
              const px = (y*scale + sy) * (W*scale) + (x*scale + sx);
              const off = px*4;
              imgData.data[off] = r; imgData.data[off+1] = g; imgData.data[off+2] = b; imgData.data[off+3] = 255;
            }
          }
        }
      }
      ctx.putImageData(imgData, 0, 0);
      drawGrid();
      drawCenter();
    }

    function drawOverlay() {
      // Start with background
      ctx.save();
      const base = '__BACKGROUND__';
      ctx.fillStyle = base; ctx.fillRect(0,0,W*scale,H*scale);
      ctx.restore();
      for (let idx=0; idx<C; idx++){
        const channelName = channels[idx];
        const pal = palette[channelName] || { color: '#ffffff', alpha: 0.6 };
        const color = hexToRgb(pal.color || '#ffffff');
        const layer = grid[idx];
        ctx.save();
        for (let y=0;y<H;y++){
          for (let x=0;x<W;x++){
            const v = Math.max(0, Math.min(1, layer[y][x]));
            const a = v * (pal.alpha ?? 0.6);
            if (a <= 0) continue;
            ctx.globalAlpha = Math.max(0, Math.min(1, a));
            ctx.fillStyle = `rgb(${Math.round(color.r*v)},${Math.round(color.g*v)},${Math.round(color.b*v)})`;
            ctx.fillRect(x*scale, y*scale, scale, scale);
          }
        }
        ctx.restore();
      }
      drawGrid();
      drawCenter();
    }

    function render() {
      setCanvasSize();
      if (overlay) drawOverlay(); else drawSingle(currentIndex);
    }

    function updateLegend(){
      const legend = document.getElementById('legend');
      legend.innerHTML='';
      channels.forEach((name, i) => {
        const item = document.createElement('div');
        item.className = 'legend-item';
        const sw = document.createElement('div');
        sw.className = 'swatch';
        const pal = palette[name] || { color: '#ffffff' };
        sw.style.background = pal.color || '#ffffff';
        const label = document.createElement('span');
        label.textContent = name;
        if (i === currentIndex && !overlay) {
          item.classList.add('active');
          item.style.color = 'var(--fg)';
        }
        item.appendChild(sw); item.appendChild(label);
        legend.appendChild(item);
      });
    }

    // UI wiring
    const select = document.getElementById('channelSelect');
    channels.forEach((name, i) => {
      const opt = document.createElement('option'); opt.value = i; opt.text = name; select.appendChild(opt);
    });
    select.addEventListener('change', (e) => { currentIndex = parseInt(e.target.value,10) || 0; if(!overlay) render(); updateLegend(); });
    document.getElementById('prevBtn').addEventListener('click', () => { currentIndex = (currentIndex - 1 + C) % C; select.value = currentIndex; if(!overlay) render(); updateLegend(); });
    document.getElementById('nextBtn').addEventListener('click', () => { currentIndex = (currentIndex + 1) % C; select.value = currentIndex; if(!overlay) render(); updateLegend(); });
    document.getElementById('overlayToggle').addEventListener('change', (e) => { overlay = !!e.target.checked; render(); updateLegend(); });
    document.getElementById('gridToggle').addEventListener('change', (e) => { showGrid = !!e.target.checked; render(); });
    document.getElementById('centerToggle').addEventListener('change', (e) => { showCenter = !!e.target.checked; render(); });
    document.getElementById('scale').addEventListener('input', (e) => { scale = parseInt(e.target.value,10)||16; render(); });

    // Keyboard shortcuts
    window.addEventListener('keydown', (e) => {
      if (e.key === '[') { document.getElementById('prevBtn').click(); }
      if (e.key === ']') { document.getElementById('nextBtn').click(); }
    });

    // Hover tooltip
    canvas.addEventListener('mousemove', (e) => {
      const rect = canvas.getBoundingClientRect();
      const x = Math.floor((e.clientX - rect.left) / scale);
      const y = Math.floor((e.clientY - rect.top) / scale);
      if (x < 0 || y < 0 || x >= W || y >= H){ tooltip.style.display='none'; return; }
      const values = channels.map((name, i) => ({ name, v: grid[i][y][x] }));
      tooltip.style.display = 'block';
      tooltip.style.left = (e.clientX - rect.left + 12) + 'px';
      tooltip.style.top = (e.clientY - rect.top + 12) + 'px';
      const centerY = Math.floor(H/2), centerX = Math.floor(W/2);
      const dy = y - centerY, dx = x - centerX;
      const dist = Math.sqrt(dx*dx + dy*dy).toFixed(2);
      let html = `<div style="font-weight:600">(y=${y}, x=${x}) d=${dist}</div>`;
      values.forEach(it => { html += `<div><span style="color:#94a3b8">${it.name}</span>: ${it.v.toFixed(3)}</div>`; });
      tooltip.innerHTML = html;
    });
    canvas.addEventListener('mouseleave', () => { tooltip.style.display='none'; });

    // Initial
    render(); updateLegend();
  </script>
</body>
</html>"""
    html = html.replace("__TITLE__", title_placeholder)
    html = html.replace("__BACKGROUND__", background_placeholder)
    html = html.replace("__INITIAL_SCALE__", initial_scale_placeholder)
    html = html.replace("__DATA__", json_data)
    html = html.replace("__PALETTE__", json_palette)
    return html

