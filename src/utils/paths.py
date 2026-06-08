from pathlib import Path

class ProjectPaths:
    def __init__(self, root):
        # Ensure that self.root is always a Path object
        self.root = Path(root)

    def __truediv__(self, other):
        # Ensure that / works perfectly like using Path
        return self.root / other
    
    def ensure_directory(self): # Differentiate between this and the other init function
        self.mkdir(
            parents=True,
            exist_ok=True
        )

    def ensure_init_directories(self):

        self.data = self.root / "data"
        self.data.mkdir(
            parents=True,
            exist_ok=True
        )

        self.saved_models = self.root / "saved_models"
        self.saved_models.mkdir(
            parents=True,
            exist_ok=True
        )

        self.experiments = self.root / "experiments"
        self.experiments.mkdir(
            parents=True,
            exist_ok=True
        )

        self.checkpoints = self.root / "checkpoints"
        self.checkpoints.mkdir(
            parents=True,
            exist_ok=True
        )
