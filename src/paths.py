from pathlib import Path

class ProjectPaths:
    def __init__(self, root):
        # Ensure that self.root is always a Path object
        self.root = Path(root)

        self.data = root / "data"

        self.saved_models = root / "saved_models"

        self.experiments = root / "experiments"

        self.checkpoints = root / "saved_models"

    def __truediv__(self, other):
        # Ensure that / works perfectly like using Path
        return self.root / other

    def ensure_directories(self):

        self.data.mkdir(
            parents=True,
            exist_ok=True
        )

        self.saved_models.mkdir(
            parents=True,
            exist_ok=True
        )

        self.experiments.mkdir(
            parents=True,
            exist_ok=True
        )