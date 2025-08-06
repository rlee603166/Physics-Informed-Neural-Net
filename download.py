import asyncio
import aiohttp
import json
import os
from pathlib import Path

# Load Box API token from environment
BOX_TOKEN = "PtqgVxeA7hi8vgSQxmD3f5I7aNwwgITe"
if not BOX_TOKEN:
    raise RuntimeError("Please set the BOX_ACCESS_TOKEN environment variable")

# Constants
JSON_FILE = "romberg.json"
CONCURRENT_DOWNLOADS = 1
OUTPUT_DIR = "data"

# Ensure output directory exists
Path(OUTPUT_DIR).mkdir(exist_ok=True)

async def download_file(session, file_info, semaphore):
    url = file_info["download_url"]
    filename = file_info["filename"]
    output_path = Path(OUTPUT_DIR) / filename

    async with semaphore:
        async with session.get(url, headers={"Authorization": f"Bearer {BOX_TOKEN}"}) as response:
            response.raise_for_status()
            with open(output_path, "wb") as f:
                while True:
                    chunk = await response.content.read(1024)
                    if not chunk:
                        break
                    f.write(chunk)
            print(f"Downloaded {filename}")

async def main():
    # Load JSON data
    with open(JSON_FILE, "r") as f:
        data = json.load(f)

    files = data["romberg_files"]

    # Setup semaphore and HTTP session
    semaphore = asyncio.Semaphore(CONCURRENT_DOWNLOADS)
    async with aiohttp.ClientSession() as session:
        tasks = [download_file(session, file_info, semaphore) for file_info in files]
        await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(main())
