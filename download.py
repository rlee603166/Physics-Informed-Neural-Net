# Load Box API token from environment
import asyncio
import aiohttp
import aiofiles
import json
import os
import time
from pathlib import Path
from typing import List, Dict

# Load Box API token from environment

BOX_TOKEN = "JvUnc7dzfAm6l6n3YXXJPvygnONCBBmT"
# Optimized constants for 30MB files
JSON_FILE = "romberg.json"
CONCURRENT_DOWNLOADS = 25  # Increased for better throughput
CHUNK_SIZE = 2 * 1024 * 1024  # 2MB chunks for optimal performance
OUTPUT_DIR = "data"
CONNECTOR_LIMIT = 100  # Connection pool size
TIMEOUT_SECONDS = 300  # 5 minute timeout per file

# Ensure output directory exists
Path(OUTPUT_DIR).mkdir(exist_ok=True)

class DownloadStats:
    def __init__(self):
        self.completed = 0
        self.failed = 0
        self.total_bytes = 0
        self.start_time = time.time()
    
    def update(self, success: bool, bytes_downloaded: int = 0):
        if success:
            self.completed += 1
            self.total_bytes += bytes_downloaded
        else:
            self.failed += 1
    
    def print_progress(self, total_files: int):
        elapsed = time.time() - self.start_time
        completed_pct = (self.completed / total_files) * 100
        mb_downloaded = self.total_bytes / (1024 * 1024)
        speed_mbps = mb_downloaded / elapsed if elapsed > 0 else 0
        
        print(f"Progress: {self.completed}/{total_files} ({completed_pct:.1f}%) | "
              f"Failed: {self.failed} | "
              f"Downloaded: {mb_downloaded:.1f}MB | "
              f"Speed: {speed_mbps:.1f}MB/s")

stats = DownloadStats()

async def download_file(session: aiohttp.ClientSession, file_info: Dict, semaphore: asyncio.Semaphore) -> bool:
    """Download a single file with optimized settings for 30MB files"""
    url = file_info["download_url"]
    filename = file_info["filename"]
    output_path = Path(OUTPUT_DIR) / filename
    
    # Skip if file already exists and has reasonable size
    if output_path.exists() and output_path.stat().st_size > 1024 * 1024:  # > 1MB
        print(f"Skipping {filename} (already exists)")
        stats.update(True, output_path.stat().st_size)
        return True
    
    async with semaphore:
        try:
            timeout = aiohttp.ClientTimeout(total=TIMEOUT_SECONDS)
            async with session.get(
                url, 
                headers={"Authorization": f"Bearer {BOX_TOKEN}"},
                timeout=timeout
            ) as response:
                response.raise_for_status()
                
                file_size = int(response.headers.get('Content-Length', 0))
                
                async with aiofiles.open(output_path, "wb") as f:
                    bytes_downloaded = 0
                    async for chunk in response.content.iter_chunked(CHUNK_SIZE):
                        await f.write(chunk)
                        bytes_downloaded += len(chunk)
                
                print(f"✓ Downloaded {filename} ({bytes_downloaded / (1024*1024):.1f}MB)")
                stats.update(True, bytes_downloaded)
                return True
                
        except asyncio.TimeoutError:
            print(f"✗ Timeout downloading {filename}")
            stats.update(False)
        except aiohttp.ClientError as e:
            print(f"✗ HTTP error downloading {filename}: {e}")
            stats.update(False)
        except Exception as e:
            print(f"✗ Error downloading {filename}: {e}")
            stats.update(False)
        
        # Clean up partial file on failure
        if output_path.exists():
            try:
                output_path.unlink()
            except:
                pass
        
        return False

async def progress_reporter(total_files: int):
    """Print progress every 10 seconds"""
    while stats.completed + stats.failed < total_files:
        await asyncio.sleep(10)
        stats.print_progress(total_files)

async def main():
    start_time = time.time()
    
    # Load JSON data
    try:
        with open(JSON_FILE, "r") as f:
            data = json.load(f)
        files = data["romberg_files"]
        print(f"Found {len(files)} files to download")
    except FileNotFoundError:
        print(f"Error: {JSON_FILE} not found")
        return
    except json.JSONDecodeError as e:
        print(f"Error parsing {JSON_FILE}: {e}")
        return
    
    # Setup optimized HTTP session
    connector = aiohttp.TCPConnector(
        limit=CONNECTOR_LIMIT,  # Connection pool size
        limit_per_host=30,      # Connections per host
        ttl_dns_cache=300,      # DNS cache TTL
        use_dns_cache=True,
        keepalive_timeout=30,   # Keep connections alive
        enable_cleanup_closed=True
    )
    
    timeout = aiohttp.ClientTimeout(
        total=TIMEOUT_SECONDS,
        connect=30,
        sock_read=30
    )
    
    semaphore = asyncio.Semaphore(CONCURRENT_DOWNLOADS)
    
    async with aiohttp.ClientSession(
        connector=connector,
        timeout=timeout,
        headers={
            "User-Agent": "Box-File-Downloader/1.0",
            "Accept": "*/*",
            "Accept-Encoding": "gzip, deflate"
        }
    ) as session:
        # Create download tasks
        tasks = [download_file(session, file_info, semaphore) for file_info in files]
        
        # Start progress reporter
        progress_task = asyncio.create_task(progress_reporter(len(files)))
        
        # Run all downloads concurrently
        print(f"Starting download of {len(files)} files with {CONCURRENT_DOWNLOADS} concurrent connections...")
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Cancel progress reporter
        progress_task.cancel()
        
        # Final statistics
        successful = sum(1 for r in results if r is True)
        failed = len(results) - successful
        
        elapsed = time.time() - start_time
        total_mb = stats.total_bytes / (1024 * 1024)
        avg_speed = total_mb / elapsed if elapsed > 0 else 0
        
        print(f"\n{'='*60}")
        print(f"Download Complete!")
        print(f"Total time: {elapsed:.1f} seconds")
        print(f"Successful: {successful}/{len(files)} files")
        print(f"Failed: {failed} files")
        print(f"Total downloaded: {total_mb:.1f}MB")
        print(f"Average speed: {avg_speed:.1f}MB/s")
        print(f"Files saved to: {OUTPUT_DIR}/")
        
        if failed > 0:
            print(f"\nNote: {failed} files failed to download. Run again to retry failed downloads.")

if __name__ == "__main__":
    print("Box File Downloader - Optimized for 30MB files")
    print("=" * 50)
    asyncio.run(main())
