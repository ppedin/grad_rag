import os
import requests
from pathlib import Path
from bs4 import BeautifulSoup
import re


def download_quality_data(output_folder="quality"):
    """
    Download all data files from the QuALITY dataset GitHub repository.

    Args:
        output_folder (str): Folder to save downloaded files
    """
    base_url = "https://github.com/nyu-mll/quality/tree/main/data/v1.0.1"
    raw_base_url = "https://raw.githubusercontent.com/nyu-mll/quality/main/data/v1.0.1"

    # Create output folder if it doesn't exist
    Path(output_folder).mkdir(exist_ok=True)

    print(f"Downloading QuALITY dataset files to {output_folder}/")

    # Get the file listing page
    response = requests.get(base_url)
    response.raise_for_status()

    soup = BeautifulSoup(response.content, 'html.parser')

    # Find all file links in the data directory
    file_links = []

    # Look for file links in the GitHub file browser
    for link in soup.find_all('a', href=True):
        href = link['href']
        # GitHub file links contain /blob/main/data/v1.0.1/
        if '/blob/main/data/v1.0.1/' in href and not href.endswith('/'):
            # Extract filename from the link
            filename = href.split('/')[-1]
            if filename and not filename.startswith('.'):  # Skip hidden files
                file_links.append(filename)

    # If the above method doesn't work well, try a more direct approach
    # by looking for specific file patterns or known file extensions
    if not file_links:
        print("Could not find files via page parsing. Trying known file patterns...")

        # Common file patterns for ML datasets
        potential_files = [
            'QuALITY.v1.0.1.htmlstripped.train',
            'QuALITY.v1.0.1.htmlstripped.dev',
            'QuALITY.v1.0.1.htmlstripped.test',
            'QuALITY.v1.0.1.train',
            'QuALITY.v1.0.1.dev',
            'QuALITY.v1.0.1.test',
            'train.jsonl',
            'dev.jsonl',
            'test.jsonl',
            'README.md'
        ]

        for filename in potential_files:
            file_url = f"{raw_base_url}/{filename}"
            try:
                head_response = requests.head(file_url)
                if head_response.status_code == 200:
                    file_links.append(filename)
                    print(f"Found file: {filename}")
            except:
                continue

    if not file_links:
        # Try another approach - look for span elements with file names
        print("Trying alternative parsing method...")
        for span in soup.find_all('span', class_='PRIVATE_TreeView-item-content-text'):
            text = span.get_text().strip()
            if text and '.' in text and not text.startswith('.'):
                file_links.append(text)

    # Remove duplicates and sort
    file_links = sorted(list(set(file_links)))

    print(f"Found {len(file_links)} files to download: {file_links}")

    if not file_links:
        print("Warning: No files found. The repository structure might have changed.")
        print("You may need to check the repository manually at:")
        print(base_url)
        return

    # Download each file
    for filename in file_links:
        # Construct download URL (raw file URL)
        download_url = f"{raw_base_url}/{filename}"
        output_path = Path(output_folder) / filename

        print(f"Downloading {filename}...")

        try:
            file_response = requests.get(download_url, stream=True)
            file_response.raise_for_status()

            with open(output_path, 'wb') as f:
                for chunk in file_response.iter_content(chunk_size=8192):
                    f.write(chunk)

            file_size = os.path.getsize(output_path)
            print(f"  Downloaded {filename} ({file_size:,} bytes)")

        except Exception as e:
            print(f"  Failed to download {filename}: {e}")

    print(f"Download complete. Files saved to {output_folder}/")


if __name__ == "__main__":
    download_quality_data()
    