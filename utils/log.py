from pathlib import Path


def log(instance, msg):
    log_dir = Path(__file__).resolve().parent.parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"log_{instance}.txt"

    with log_file.open("a") as f:
        f.write(msg + "\n")
