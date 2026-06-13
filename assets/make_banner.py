"""Generate the wide GitHub hero banner (1280x360), matching the cover aesthetic."""
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

W, H = 1280, 360
fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
ax.set_xlim(0, W); ax.set_ylim(0, H)

# background + subtle gradient
ax.add_patch(plt.Rectangle((0, 0), W, H, color="#0f1b2d", zorder=-1))
grad = np.linspace(0, 1, 256).reshape(1, -1)
ax.imshow(np.vstack([grad] * 8), extent=[0, W, 0, H], aspect="auto", cmap="GnBu_r", alpha=0.10, zorder=0)
ax.add_patch(plt.Rectangle((0, H - 9), W, 9, color="#e74c3c", zorder=2))  # top accent

# ECG motif across the upper band, clear of the title
xs = np.linspace(720, 1255, 600); ecg = np.zeros_like(xs)
for i, xv in enumerate(xs):
    t = (xv - 720) % 90
    ecg[i] = math.exp(-((t - 45) ** 2) / 6.0) * 30 - (7 if 37 < t < 45 else 0)
ax.plot(xs, 322 + ecg, color="#ff6b5e", lw=2.0, zorder=2)

# title + subtitle
ax.text(60, 250, "Second Look", fontsize=76, fontweight="bold", color="white", zorder=3)
ax.text(64, 207, "a triage safety-net that knows what it doesn't know",
        fontsize=21, color="#9fc5e8", style="italic", zorder=3)

# four stat chips in one row
chips = [("text alone → 99.9% acc", "#2ecc71"),
         ("undertriage 0.8% → 78% under shift", "#e74c3c"),
         ("red-flag recall 90% vs 30%", "#f1c40f"),
         ("44 tests · live demo", "#4aa3ff")]
x = 58
for txt, col in chips:
    w = 9.0 * len(txt) + 26
    ax.add_patch(FancyBboxPatch((x, 120), w, 40, boxstyle="round,pad=2,rounding_size=10",
                                fc=col + "22", ec=col, lw=1.8, zorder=3))
    ax.text(x + w / 2, 140, txt, fontsize=12.5, color="white", ha="center", va="center", zorder=4)
    x += w + 14

# footer
ax.text(60, 78, "Triagegeist  ·  emergency-department triage decision support",
        fontsize=15, color="#cbd5e1", zorder=3)
ax.text(60, 48, "calibrated ESI   ·   clinical red-flags   ·   informative-missingness audit   ·   fairness   ·   Gradio demo",
        fontsize=12, color="#7f93a8", zorder=3)

fig.savefig("assets/banner.png", dpi=100)
print("saved assets/banner.png")
