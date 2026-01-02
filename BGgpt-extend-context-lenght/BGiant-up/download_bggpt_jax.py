import pathlib
from huggingface_hub import snapshot_download

REPO_ID = "INSAIT-Institute/BgGPT-Gemma-2-2.6B-IT-v1.0"


def main():
    local_dir = "/Volumes/SSD/r2/BGiant/BgGPT-Gemma-2-2.6B-IT-v1.0"

    print(f"Downloading {REPO_ID} to {local_dir} ...")
    snapshot_download(
        repo_id=REPO_ID,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
        revision="main",
    )
    print("Done.")


if __name__ == "__main__":
    main()

