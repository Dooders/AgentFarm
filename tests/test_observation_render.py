import numpy as np

from farm.core.observation_render import ObservationRenderer, get_default_palette


def test_overlay_render_returns_pil_image():
    C, H, W = 3, 7, 7
    grid = np.zeros((C, H, W), dtype=np.float32)
    grid[0, 3, 3] = 1.0  # center hit on channel 0
    grid[1, 2:5, 2:5] = 0.5
    grid[2, :, :] = np.linspace(0.0, 1.0, H).reshape(H, 1)
    channels = ["A", "B", "C"]

    img = ObservationRenderer.render(
        grid, channels, mode="overlay", size=128, background="#101113"
    )
    assert img.width >= 128 or img.height >= 128


def test_gallery_render_dimensions():
    C, H, W = 4, 5, 5
    grid = np.random.rand(C, H, W).astype(np.float32)
    channels = [f"C{i}" for i in range(C)]
    img = ObservationRenderer.render(grid, channels, mode="gallery", size=200)
    assert img.width > 0 and img.height > 0


def test_to_interactive_json_shape_and_channels():
    C, H, W = 2, 3, 4
    grid = np.zeros((C, H, W), dtype=np.float32)
    channels = ["X", "Y"]
    payload = ObservationRenderer.to_interactive_json(grid, channels, meta={"R": 3})
    assert payload["shape"] == [C, H, W]
    assert payload["channels"] == channels
    assert payload["meta"]["R"] == 3
