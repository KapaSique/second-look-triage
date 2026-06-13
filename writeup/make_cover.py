"""Generate the 560x280 competition cover image (clean, non-overlapping layout)."""
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

W, H = 560, 280
fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
ax.set_xlim(0, W); ax.set_ylim(0, H)

# background
ax.add_patch(plt.Rectangle((0, 0), W, H, color="#0f1b2d", zorder=-1))
grad = np.linspace(0, 1, 256).reshape(1, -1)
ax.imshow(np.vstack([grad] * 8), extent=[0, W, 0, H], aspect="auto", cmap="GnBu_r", alpha=0.10, zorder=0)
ax.add_patch(plt.Rectangle((0, H - 7), W, 7, color="#e74c3c", zorder=2))   # top accent

# ECG motif — top-right corner, clear of the title
xs = np.linspace(395, 545, 200); ecg = np.zeros_like(xs)
for i, xv in enumerate(xs):
    t = (xv - 395) % 55
    ecg[i] = math.exp(-((t - 27) ** 2) / 3.0) * 18 - (4 if 22 < t < 27 else 0)
ax.plot(xs, 252 + ecg, color="#ff6b5e", lw=1.6, zorder=2)

# title + subtitle
ax.text(28, 196, "Second Look", fontsize=39, fontweight="bold", color="white", zorder=3)
ax.text(30, 169, "a triage safety-net that knows what it doesn't know",
        fontsize=12.5, color="#9fc5e8", style="italic", zorder=3)

# three stat chips in one row (no wrapping)
chips = [("text alone → 99.9%", "#2ecc71"),
         ("missingness trap 0.8%→78%", "#e74c3c"),
         ("red-flag recall 90%", "#f1c40f")]
x = 24
for txt, col in chips:
    w = 6.0 * len(txt) + 20
    ax.add_patch(FancyBboxPatch((x, 104), w, 28, boxstyle="round,pad=2,rounding_size=7",
                                fc=col + "22", ec=col, lw=1.6, zorder=3))
    ax.text(x + w / 2, 118, txt, fontsize=9, color="white", ha="center", va="center", zorder=4)
    x += w + 12

# footer
ax.text(28, 58, "Triagegeist  ·  emergency-department triage decision support",
        fontsize=11.5, color="#cbd5e1", zorder=3)
ax.text(28, 34, "calibrated ESI   ·   clinical red-flags   ·   honest audit   ·   live demo",
        fontsize=9.5, color="#7f93a8", zorder=3)

fig.savefig("writeup/cover.png", dpi=100)
print("saved writeup/cover.png")
