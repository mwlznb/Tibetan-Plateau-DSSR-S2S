from src.training import deterministic
from src.diffusion import training as diffusion


if __name__ == "__main__":
    deterministic.run()
    diffusion.run()
