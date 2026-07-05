import os
import matplotlib.pyplot as plt


class TrainingPlotter:
    def __init__(self, save_dir="plots", filename="training_curves.png"):
        os.makedirs(save_dir, exist_ok=True)
        self.save_path = os.path.join(save_dir, filename)

        self.train_losses = []
        self.val_losses = []
        self.f1_scores = []

        plt.ion()  # Interactive mode

        self.fig, self.ax_loss = plt.subplots(figsize=(10, 6))
        self.ax_f1 = self.ax_loss.twinx()

        # Loss curves (left axis)
        self.train_line, = self.ax_loss.plot([], [], label="Train Loss", color="tab:gray", linewidth=2)
        self.val_line, = self.ax_loss.plot([], [], label="Validation Loss", color="tab:orange", linewidth=2)

        # F1 curve (right axis)
        self.f1_line, = self.ax_f1.plot([], [], label="F1 Score", color="tab:blue", linewidth=2)

        self.ax_loss.set_xlabel("Epoch")
        self.ax_loss.set_ylabel("Loss")
        self.ax_f1.set_ylabel("F1 Score")

        self.ax_loss.grid(True, linestyle="--", alpha=0.5)

        # Combine legends
        lines = [self.train_line, self.val_line, self.f1_line]
        labels = [l.get_label() for l in lines]
        self.ax_loss.legend(lines, labels, loc="best")

        self.fig.show()

    def update(self, train_loss, val_loss, f1_score):
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.f1_scores.append(f1_score)

        epochs = range(1, len(self.train_losses) + 1)

        self.train_line.set_data(epochs, self.train_losses)
        self.val_line.set_data(epochs, self.val_losses)
        self.f1_line.set_data(epochs, self.f1_scores)

        # Autoscale each axis independently
        self.ax_loss.relim()
        self.ax_loss.autoscale_view()

        self.ax_f1.relim()
        self.ax_f1.autoscale_view()

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        plt.pause(0.01)  # Allow the GUI to refresh

    def save(self):
        self.fig.savefig(self.save_path, dpi=300)
        print(f"Saved plot to: {self.save_path}")

    def close(self):
        plt.ioff()
        plt.close(self.fig)