from huggingface_hub import list_repo_files, get_hf_file_metadata, hf_hub_url

repo_id = "global-optima-research/HDTF"

def list_files():
    files = list_repo_files(repo_id, repo_type="dataset")
    print(f"Found {len(files)} files.")
    for f in files:
        # if f.endswith(".zip") or f.endswith(".part_aa"):
        if True:
            url = hf_hub_url(repo_id, f, repo_type="dataset")
            try:
                meta = get_hf_file_metadata(url)
                size_mb = meta.size / (1024 * 1024)
                print(f"{f}: {size_mb:.2f} MB")
            except:
                print(f"{f}: Metadata error")

if __name__ == "__main__":
    list_files()
