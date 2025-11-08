from pathlib import Path


def log(instance, msg, first_call=False):
    log_dir = Path(__file__).resolve().parent.parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"log_{instance}.txt"
    log_file.touch(exist_ok=True)

    if first_call:
        log_file.open("w").close()

    with log_file.open("a") as f:
        f.write(msg + "\n")
