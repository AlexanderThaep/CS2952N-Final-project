from pathlib import Path
import subprocess

INPUT_ROOT = Path("processed_data")
OUTPUT_ROOT = Path("latents")

for video_path in INPUT_ROOT.rglob("*.mp4"):

    relative_path = video_path.relative_to(INPUT_ROOT)

    output_dir = OUTPUT_ROOT / relative_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Processing: {video_path}")

    cmd = [
        "python",
        "scripts/vae_encode_save.py",
        "--input_video",
        str(video_path),
        "--save_path",
        str(output_dir)
    ]

    subprocess.run(cmd, check=True)

print("Done")