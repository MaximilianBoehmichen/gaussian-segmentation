import argparse
import subprocess
import tarfile
import sys
from pathlib import Path


def download_oasis_tar_stream(
        remote_host: str,
        remote_base_dir: str,
        local_base_dir: Path,
        n_train: int = 10,
        n_val: int = 5,
        n_test: int = 5,
        flatten_splits: bool = False
) -> None:
    """
    Downloads files from the remote server over a single SSH connection by streaming
    them through a tar pipeline. This avoids Windows ControlMaster compatibility issues
    and only requires a single password prompt.
    """
    local_base_dir.mkdir(parents=True, exist_ok=True)
    split_limits = {"train": n_train, "val": n_val, "test": n_test}

    # 1. Build the remote command to locate the files and stream them to standard output
    find_commands = []
    for split, limit in split_limits.items():
        if limit <= 0:
            continue
        find_commands.append(
            f"if [ -d '{split}' ]; then "
            f"  find '{split}' -mindepth 1 -maxdepth 1 -type d | sort | head -n {limit} | while read -r subj; do "
            f"    for fname in aligned_orig.nii.gz aligned_seg4.nii.gz; do "
            f"      if [ -f \"$subj/$fname\" ]; then echo \"$subj/$fname\"; fi; "
            f"    done; "
            f"  done; "
            f"fi"
        )

    # Combine commands: CD into directory, run all finds, pipe the paths to tar
    all_finds = " ; ".join(find_commands)
    remote_cmd = f"cd '{remote_base_dir}' || exit 1 ; ( {all_finds} ) | tar -cf - -T -"

    print(f"Connecting to {remote_host}...")
    print("You will be prompted for your password once to establish the stream.")

    # 2. Open a single SSH subprocess to read the tar stream
    process = subprocess.Popen(
        ["ssh", remote_host, remote_cmd],
        stdout=subprocess.PIPE,
        stderr=None,  # Leave stderr bound to console so you can see the password prompt
        bufsize=8192
    )

    downloaded_count = 0

    # 3. Intercept the tar stream in Python and extract files to the target directory
    try:
        with tarfile.open(fileobj=process.stdout, mode="r|") as tar:
            for member in tar:
                # Modify the extraction path if the user wants to strip train/val/test folders
                if flatten_splits:
                    parts = Path(member.name).parts
                    if len(parts) > 1:
                        member.name = str(Path(*parts[1:]))

                print(f"Extracting: {member.name}")

                # The 'data_filter' argument is required in newer Python versions for security,
                # but if you are on an older version of Python, you can remove 'filter="data"'
                if sys.version_info >= (3, 11, 4):
                    tar.extract(member, path=local_base_dir, filter="data")
                else:
                    tar.extract(member, path=local_base_dir)

                downloaded_count += 1

    except tarfile.ReadError:
        print("\nError: Failed to read the file stream.")
        print("Did you enter the wrong password, or did the SSH connection drop?")
        sys.exit(1)

    process.wait()

    if process.returncode != 0:
        print(f"\nSSH process exited with code {process.returncode}")
    elif downloaded_count == 0:
        print("\nConnection succeeded, but no files were found matching your criteria.")
    else:
        print(f"\nSuccessfully downloaded and extracted {downloaded_count} files.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download N subjects per split from OASIS_L2R using a single SSH stream.")
    parser.add_argument(
        "--host",
        type=str,
        required=True,
        help="SSH host (e.g., user@server)"
    )
    parser.add_argument(
        "--remote-dir",
        type=str,
        default="/vol/miltank/projects/L2R/OASIS_L2R",
        help="Remote dataset directory"
    )
    parser.add_argument(
        "--local-dir",
        type=Path,
        default=Path(r"C:\Users\Nil\Documents\data\OASIS\OASIS_L2R"),
        help="Local destination directory"
    )
    parser.add_argument("--n-train", type=int, default=10, help="Number of train subjects to download")
    parser.add_argument("--n-val", type=int, default=2, help="Number of val subjects to download")
    parser.add_argument("--n-test", type=int, default=2, help="Number of test subjects to download")
    parser.add_argument(
        "--flatten",
        action="store_true",
        help="Place subject directories directly in --local-dir instead of train/val/test subfolders"
    )

    args = parser.parse_args()
    download_oasis_tar_stream(
        remote_host=args.host,
        remote_base_dir=args.remote_dir,
        local_base_dir=args.local_dir,
        n_train=args.n_train,
        n_val=args.n_val,
        n_test=args.n_test,
        flatten_splits=args.flatten
    )