from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import ffmpeg
import sys

# --------------------------------------------------
# Config
# --------------------------------------------------

INPUT_ROOT = Path("data")
OUTPUT_ROOT = Path("processed_data_b")

TARGET_HEIGHT = 240

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

# Number of parallel ffmpeg processes
MAX_WORKERS = 2

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# Video Processing
# --------------------------------------------------

def process_video(input_file: Path):
    relative_path = input_file.relative_to(INPUT_ROOT)
    output_file = OUTPUT_ROOT / relative_path

    output_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"Processing: {relative_path}")

    try:
        (
            ffmpeg
            .input(str(input_file), t=8)
            .output(
                str(output_file),

                vf=f"fps=15,scale=-2:{TARGET_HEIGHT}",

                vcodec="h264_nvenc",
                preset="p4",

                acodec="copy",

                movflags="+faststart"
            )
            .overwrite_output()
            .run(quiet=True)
        )

        print(f"Saved -> {output_file}")

    except ffmpeg.Error as e:
        print(f"Failed: {relative_path}")

        if e.stderr:
            print(e.stderr.decode())

        print("-" * 50)


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    if not INPUT_ROOT.exists():
        print("Input directory does not exist.")
        sys.exit(1)

    videos = [
        f for f in INPUT_ROOT.rglob("*")
        if f.is_file() and f.suffix.lower() in VIDEO_EXTS
    ]

    if not videos:
        print("No videos found.")
        return

    print(f"Found {len(videos)} videos")

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        executor.map(process_video, videos)


if __name__ == "__main__":
    main()