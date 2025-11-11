
# Save-only training plot (no GUI popup)
import matplotlib
matplotlib.use("Agg")  # headless backend

import matplotlib.pyplot as plt

def plot(scores, mean_scores, save_path="training_curve.png"):
    plt.clf()
    plt.title("Training")
    plt.xlabel("Number of Games")
    plt.ylabel("Score")
    plt.plot(scores, label="score")
    plt.plot(mean_scores, label="mean score")
    plt.legend()
    if scores:
        plt.text(len(scores)-1, scores[-1], str(scores[-1]))
        plt.text(len(mean_scores)-1, mean_scores[-1], f"{mean_scores[-1]:.1f}")
    plt.tight_layout()
    plt.savefig(save_path)  # <-- only save to file; no show()
