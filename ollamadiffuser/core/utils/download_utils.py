#!/usr/bin/env python3
"""
Download utilities for robust model downloading with detailed progress tracking
"""

import os
import time
import logging
from typing import Optional, Callable, Any, Dict
from pathlib import Path
from huggingface_hub import snapshot_download, hf_hub_download, HfApi
from huggingface_hub.errors import (
    EntryNotFoundError,
    GatedRepoError,
    HfHubHTTPError,
    RepositoryNotFoundError,
    RevisionNotFoundError,
)
from tqdm import tqdm
import threading
import requests
import fnmatch

logger = logging.getLogger(__name__)

# Errors that retrying cannot fix: the repo needs access we do not have, or
# it is not there. Retried anyway, a 403 sleeps through five attempts while
# the progress bars keep drawing, and the one line that says what to do
# scrolls away under them — which is how a gated pack came back looking like
# a finished download of nothing.
_PERMANENT_ERRORS = (
    GatedRepoError,
    RepositoryNotFoundError,
    RevisionNotFoundError,
    EntryNotFoundError,
)
_PERMANENT_STATUSES = {401, 403, 404}


def access_problem(error: Exception) -> Optional[str]:
    """What to do about ``error``, when no amount of retrying would help.

    Returns a sentence for the user, or None when the error is the ordinary
    kind that a retry might get past.
    """
    if isinstance(error, GatedRepoError):
        repo = getattr(error, "repo_id", None) or ""
        where = f"https://huggingface.co/{repo}" if repo else "the model page"
        return (f"this repo is gated and this account is not approved yet: open {where}, "
                f"click Request access and wait for the author. If you were already "
                f"approved, log in first with: hf auth login")
    if isinstance(error, (RepositoryNotFoundError, RevisionNotFoundError)):
        return ("no such repo (or it is private): check the repo_id, and for a private "
                "one log in first with: hf auth login")
    if isinstance(error, EntryNotFoundError):
        return "the repo does not have that file: check allow_patterns against its file list."
    if isinstance(error, HfHubHTTPError):
        status = getattr(getattr(error, "response", None), "status_code", None)
        if status in _PERMANENT_STATUSES:
            if status == 401:
                return "HuggingFace says unauthorized (401): log in with: hf auth login"
            if status == 403:
                return "HuggingFace refused (403): this repo needs access granted first."
            return "HuggingFace has no such thing (404): check the repo_id and file names."
    return None

class EnhancedProgressTracker:
    """Enhanced progress tracker that provides Ollama-style detailed progress information"""
    
    def __init__(self, total_files: int = 0, progress_callback: Optional[Callable] = None):
        self.total_files = total_files
        self.completed_files = 0
        self.current_file = ""
        self.file_progress = {}
        self.file_start_times = {}
        self.file_speeds = {}
        self.progress_callback = progress_callback
        self.lock = threading.Lock()
        self.overall_start_time = time.time()
        self.total_size = 0
        self.downloaded_size = 0
        
    def set_total_size(self, total_size: int):
        """Set the total size for all files"""
        with self.lock:
            self.total_size = total_size
    
    def start_file(self, filename: str, file_size: int = 0):
        """Mark a file as started"""
        with self.lock:
            self.current_file = filename
            self.file_start_times[filename] = time.time()
            self.file_progress[filename] = (0, file_size)
            
            # Extract hash-like identifier for Ollama-style display
            import re
            hash_match = re.search(r'([a-f0-9]{8,})', filename)
            if hash_match:
                display_name = hash_match.group(1)[:12]  # First 12 characters
            else:
                # Fallback to filename without extension
                display_name = Path(filename).stem[:12]
            
            if self.progress_callback:
                self.progress_callback(f"pulling {display_name}")
    
    def update_file_progress(self, filename: str, downloaded: int, total: int):
        """Update progress for a specific file with speed calculation"""
        with self.lock:
            current_time = time.time()
            
            # Update file progress
            old_downloaded = self.file_progress.get(filename, (0, 0))[0]
            self.file_progress[filename] = (downloaded, total)
            
            # Update overall downloaded size
            size_diff = downloaded - old_downloaded
            self.downloaded_size += size_diff
            
            # Calculate speed for this file
            if filename in self.file_start_times:
                elapsed = current_time - self.file_start_times[filename]
                if elapsed > 0 and downloaded > 0:
                    speed = downloaded / elapsed  # bytes per second
                    self.file_speeds[filename] = speed
            
            # Report progress in Ollama style
            if self.progress_callback and total > 0:
                percentage = (downloaded / total) * 100
                
                # Format sizes
                downloaded_mb = downloaded / (1024 * 1024)
                total_mb = total / (1024 * 1024)
                
                # Calculate speed in MB/s
                speed_mbps = self.file_speeds.get(filename, 0) / (1024 * 1024)
                
                # Calculate ETA
                if speed_mbps > 0:
                    remaining_mb = total_mb - downloaded_mb
                    eta_seconds = remaining_mb / speed_mbps
                    eta_min = int(eta_seconds // 60)
                    eta_sec = int(eta_seconds % 60)
                    eta_str = f"{eta_min}m{eta_sec:02d}s"
                else:
                    eta_str = "?"
                
                # Extract hash for display
                import re
                hash_match = re.search(r'([a-f0-9]{8,})', filename)
                if hash_match:
                    display_name = hash_match.group(1)[:12]
                else:
                    display_name = Path(filename).stem[:12]
                
                # Create progress bar
                bar_width = 20
                filled = int((percentage / 100) * bar_width)
                bar = "█" * filled + " " * (bar_width - filled)
                
                progress_msg = f"pulling {display_name}: {percentage:3.0f}% ▕{bar}▏ {downloaded_mb:.0f} MB/{total_mb:.0f} MB {speed_mbps:.0f} MB/s {eta_str}"
                
                self.progress_callback(progress_msg)
    
    def complete_file(self, filename: str):
        """Mark a file as completed"""
        with self.lock:
            self.completed_files += 1
            if filename in self.file_progress:
                downloaded, total = self.file_progress[filename]
                self.file_progress[filename] = (total, total)
            
            # Report completion
            if self.progress_callback:
                import re
                hash_match = re.search(r'([a-f0-9]{8,})', filename)
                if hash_match:
                    display_name = hash_match.group(1)[:12]
                else:
                    display_name = Path(filename).stem[:12]
                
                total_mb = self.file_progress.get(filename, (0, 0))[1] / (1024 * 1024)
                self.progress_callback(f"pulling {display_name}: 100% ▕████████████████████▏ {total_mb:.0f} MB/{total_mb:.0f} MB")
    
    def report_overall_progress(self):
        """Report overall progress"""
        if self.progress_callback:
            if self.total_size > 0:
                overall_percent = (self.downloaded_size / self.total_size) * 100
                downloaded_gb = self.downloaded_size / (1024 * 1024 * 1024)
                total_gb = self.total_size / (1024 * 1024 * 1024)
                
                elapsed = time.time() - self.overall_start_time
                if elapsed > 0:
                    overall_speed = self.downloaded_size / elapsed / (1024 * 1024)  # MB/s
                    
                    if overall_speed > 0:
                        remaining_gb = total_gb - downloaded_gb
                        eta_seconds = (remaining_gb * 1024) / overall_speed  # Convert GB to MB for calculation
                        eta_min = int(eta_seconds // 60)
                        eta_sec = int(eta_seconds % 60)
                        eta_str = f"{eta_min}m{eta_sec:02d}s"
                    else:
                        eta_str = "?"
                    
                    progress_msg = f"Overall progress: {overall_percent:.1f}% | {downloaded_gb:.1f} GB/{total_gb:.1f} GB | {overall_speed:.1f} MB/s | ETA: {eta_str}"
                    self.progress_callback(progress_msg)

def configure_hf_environment():
    """Configure HuggingFace Hub environment for better downloads"""
    # Set reasonable timeouts
    os.environ.setdefault('HF_HUB_DOWNLOAD_TIMEOUT', '600')  # 10 minutes
    os.environ.setdefault('HF_HUB_CONNECTION_TIMEOUT', '120')  # 2 minutes
    
    # Disable symlinks for better compatibility
    os.environ.setdefault('HF_HUB_LOCAL_DIR_USE_SYMLINKS', 'False')
    
    # Enable resume downloads
    os.environ.setdefault('HF_HUB_ENABLE_HF_TRANSFER', 'False')  # Disable for better compatibility

def get_repo_file_list(repo_id: str) -> Dict[str, int]:
    """Get list of files in repository with their sizes"""
    try:
        api = HfApi()
        # files_metadata=True is what carries the sizes. Without it every
        # sibling's size is None, the total comes out 0, and the progress bar
        # renders a 30 GB download as "0 MB / 0 MB — 100%".
        repo_info = api.repo_info(repo_id=repo_id, files_metadata=True)
        
        file_sizes = {}
        for sibling in repo_info.siblings:
            # Include all files, use 0 as default size if not available
            size = sibling.size if sibling.size is not None else 0
            file_sizes[sibling.rfilename] = size
        
        return file_sizes
    except Exception as e:
        logger.warning(f"Could not get file list for {repo_id}: {e}")
        return {}

def format_size(size_bytes: int) -> str:
    """Format size in human readable format"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"

def robust_snapshot_download(
    repo_id: str,
    local_dir: str,
    cache_dir: Optional[str] = None,
    max_retries: int = 3,
    initial_workers: int = 2,
    force_download: bool = False,
    progress_callback: Optional[Callable] = None,
    allow_patterns: Optional[list] = None,
    ignore_patterns: Optional[list] = None
) -> str:
    """
    Download repository snapshot with robust error handling and detailed progress tracking
    
    Args:
        repo_id: Repository ID on HuggingFace Hub
        local_dir: Local directory to download to
        cache_dir: Cache directory
        max_retries: Maximum number of retry attempts
        initial_workers: Initial number of workers (reduced on retries)
        force_download: Force re-download
        progress_callback: Optional progress callback function
        allow_patterns: List of file patterns to include (e.g., ["*.gguf", "*.safetensors"])
        ignore_patterns: List of file patterns to exclude (e.g., ["*.txt", "*.md"])
    
    Returns:
        Path to downloaded repository
    """
    configure_hf_environment()
    
    # Get file list and sizes for progress tracking
    if progress_callback:
        progress_callback("pulling manifest")
    
    file_sizes = get_repo_file_list(repo_id)
    
    # Filter files based on patterns if provided
    if allow_patterns or ignore_patterns:
        filtered_files = {}
        for filename, size in file_sizes.items():
            # Check allow patterns (if provided, file must match at least one)
            if allow_patterns:
                allowed = any(fnmatch.fnmatch(filename, pattern) for pattern in allow_patterns)
                if not allowed:
                    continue
            
            # Check ignore patterns (if file matches any, skip it)
            if ignore_patterns:
                ignored = any(fnmatch.fnmatch(filename, pattern) for pattern in ignore_patterns)
                if ignored:
                    continue
            
            filtered_files[filename] = size
        
        file_sizes = filtered_files
        
        if progress_callback and allow_patterns:
            progress_callback(f"🔍 Filtering files with patterns: {allow_patterns}")
        if progress_callback and ignore_patterns:
            progress_callback(f"🚫 Ignoring patterns: {ignore_patterns}")
    
    total_size = sum(file_sizes.values())
    
    if progress_callback and file_sizes:
        progress_callback(f"📦 Repository: {len(file_sizes)} files, {format_size(total_size)} total")
    
    # Initialize enhanced progress tracker
    progress_tracker = EnhancedProgressTracker(len(file_sizes), progress_callback)
    progress_tracker.set_total_size(total_size)
    
    # Check what's already downloaded
    local_path = Path(local_dir)
    existing_size = 0
    if local_path.exists() and not force_download:
        existing_files = []
        for file_path in local_path.rglob('*'):
            if file_path.is_file():
                rel_path = file_path.relative_to(local_path)
                existing_files.append(str(rel_path))
                file_size = file_path.stat().st_size
                existing_size += file_size
                # Mark existing files as completed in progress tracker
                progress_tracker.file_progress[str(rel_path)] = (file_size, file_size)
                progress_tracker.downloaded_size += file_size
                progress_tracker.completed_files += 1
        
        if progress_callback and existing_files:
            progress_callback(f"📁 Found {len(existing_files)} existing files ({format_size(existing_size)})")
    
    # Custom tqdm class to capture HuggingFace download progress
    class OllamaStyleTqdm(tqdm):
        def __init__(self, *args, **kwargs):
            # Extract description to get filename
            desc = kwargs.get('desc', '')
            self.current_filename = desc

            # Get file size from our pre-fetched data
            file_size = file_sizes.get(self.current_filename, 0)
            if file_size > 0:
                kwargs['total'] = file_size

            # Remove kwargs that huggingface_hub passes but tqdm doesn't accept
            kwargs.pop("name", None)

            super().__init__(*args, **kwargs)
            
            # Start tracking this file
            if self.current_filename and progress_callback:
                progress_tracker.start_file(self.current_filename, file_size)
        
        def update(self, n=1):
            super().update(n)
            
            # Update our progress tracker
            if self.current_filename and progress_callback:
                downloaded = getattr(self, 'n', 0)
                total = getattr(self, 'total', 0) or file_sizes.get(self.current_filename, 0)
                
                if total > 0:
                    progress_tracker.update_file_progress(self.current_filename, downloaded, total)
        
        def close(self):
            super().close()
            
            # Mark file as completed
            if self.current_filename and progress_callback:
                progress_tracker.complete_file(self.current_filename)
    
    last_exception = None
    
    for attempt in range(max_retries):
        try:
            # Reduce workers on retry attempts to avoid overwhelming connections
            workers = 1 if attempt > 0 else initial_workers
            
            if progress_callback:
                progress_callback(f"🔄 Download attempt {attempt + 1}/{max_retries} (workers: {workers})")
            
            logger.info(f"Download attempt {attempt + 1}/{max_retries} with {workers} workers")
            
            # NOTE: resume_download and local_dir_use_symlinks were removed
            # in huggingface_hub >= 1.0. Resume is now always enabled by
            # default and symlinks are handled automatically.
            result = snapshot_download(
                repo_id=repo_id,
                local_dir=local_dir,
                cache_dir=cache_dir,
                max_workers=workers,
                etag_timeout=300 + (attempt * 60),  # Increase timeout on retries
                force_download=force_download,
                tqdm_class=OllamaStyleTqdm if progress_callback else None,
                allow_patterns=allow_patterns,
                ignore_patterns=ignore_patterns
            )
            
            if progress_callback:
                progress_callback(f"✅ Successfully downloaded {repo_id}")
            
            logger.info(f"Successfully downloaded {repo_id}")
            return result
            
        except Exception as e:
            last_exception = e
            error_msg = str(e)

            # A permission error on attempt 1 is a permission error on attempt
            # 5. Say what to do and stop, rather than sleeping through four
            # more attempts with the reason buried under progress bars.
            advice = access_problem(e)
            if advice or isinstance(e, _PERMANENT_ERRORS):
                logger.error(f"Download refused for {repo_id}: {error_msg}")
                if progress_callback:
                    progress_callback(f"❌ Cannot download: {advice or error_msg}")
                raise RuntimeError(f"{repo_id}: {advice or error_msg}") from e

            # Log the specific error
            logger.warning(f"Download attempt {attempt + 1} failed: {error_msg}")
            
            if attempt < max_retries - 1:
                # Determine wait time based on error type
                if "timeout" in error_msg.lower():
                    wait_time = 30 + (attempt * 15)  # Longer wait for timeouts
                elif "connection" in error_msg.lower():
                    wait_time = 20 + (attempt * 10)  # Medium wait for connection errors
                else:
                    wait_time = 10 + (attempt * 5)   # Shorter wait for other errors
                
                logger.info(f"Waiting {wait_time} seconds before retry...")
                
                if progress_callback:
                    progress_callback(f"⚠️ Download failed, retrying in {wait_time}s... (Error: {error_msg[:100]})")
                
                time.sleep(wait_time)
            else:
                logger.error(f"All download attempts failed. Final error: {error_msg}")
                if progress_callback:
                    progress_callback(f"❌ All download attempts failed: {error_msg}")
    
    # If we get here, all retries failed
    raise last_exception

def robust_file_download(
    repo_id: str,
    filename: str,
    local_dir: str,
    cache_dir: Optional[str] = None,
    max_retries: int = 3,
    progress_callback: Optional[Callable] = None
) -> str:
    """
    Download single file with robust error handling and progress tracking
    
    Args:
        repo_id: Repository ID on HuggingFace Hub
        filename: File to download
        local_dir: Local directory to download to
        cache_dir: Cache directory
        max_retries: Maximum number of retry attempts
        progress_callback: Optional progress callback function
    
    Returns:
        Path to downloaded file
    """
    configure_hf_environment()
    
    last_exception = None
    
    for attempt in range(max_retries):
        try:
            if progress_callback:
                progress_callback(f"📥 Downloading {filename} (attempt {attempt + 1}/{max_retries})")
            
            logger.info(f"File download attempt {attempt + 1}/{max_retries}: {filename}")
            
            # NOTE: resume_download was removed in huggingface_hub >= 1.0.
            # Resume is now always enabled by default.
            result = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=local_dir,
                cache_dir=cache_dir,
                etag_timeout=180 + (attempt * 30)
            )
            
            if progress_callback:
                progress_callback(f"✅ Downloaded {filename}")
            
            logger.info(f"Successfully downloaded {filename}")
            return result
            
        except Exception as e:
            last_exception = e
            error_msg = str(e)
            
            logger.warning(f"File download attempt {attempt + 1} failed: {error_msg}")
            
            if attempt < max_retries - 1:
                wait_time = 5 + (attempt * 3)  # Progressive backoff
                
                if progress_callback:
                    progress_callback(f"⚠️ Retrying {filename} in {wait_time}s...")
                
                time.sleep(wait_time)
            else:
                logger.error(f"All file download attempts failed. Final error: {error_msg}")
                if progress_callback:
                    progress_callback(f"❌ Failed to download {filename}: {error_msg}")
    
    # If we get here, all retries failed
    raise last_exception

def check_download_integrity(local_dir: str, repo_id: str) -> bool:
    """Check if downloaded files are complete and valid"""
    try:
        local_path = Path(local_dir)
        if not local_path.exists():
            return False
        
        # Determine model type based on repo_id
        is_controlnet = 'controlnet' in repo_id.lower()
        
        # Check for essential files based on model type
        if is_controlnet:
            # ControlNet models have different essential files
            essential_files = ['config.json']  # ControlNet models use config.json instead of model_index.json
            # Also check for model files
            model_files = ['diffusion_pytorch_model.safetensors', 'diffusion_pytorch_model.bin']
            has_model_file = any((local_path / model_file).exists() for model_file in model_files)
            if not has_model_file:
                logger.warning(f"Missing model file: expected one of {model_files}")
                return False
        else:
            # Regular diffusion models
            essential_files = ['model_index.json']
        
        for essential_file in essential_files:
            if not (local_path / essential_file).exists():
                logger.warning(f"Missing essential file: {essential_file}")
                return False
        
        # Files to ignore during integrity check
        ignore_patterns = [
            '.lock',           # HuggingFace lock files
            '.metadata',       # HuggingFace metadata files
            '.incomplete',     # Incomplete download files
            '.cache',          # Cache directory
            '.git',            # Git files
            '.gitattributes',  # Git attributes
            'README.md',       # Documentation files
            'LICENSE.md',      # License files
            'dev_grid.jpg'     # Sample images
        ]
        
        # Check for empty files (excluding ignored patterns)
        for file_path in local_path.rglob('*'):
            if file_path.is_file():
                # Skip files that match ignore patterns
                should_ignore = any(pattern in str(file_path) for pattern in ignore_patterns)
                if should_ignore:
                    continue
                
                # Check if file is empty
                if file_path.stat().st_size == 0:
                    logger.warning(f"Empty file detected: {file_path}")
                    return False
        
        # Check for critical model files based on model type
        if is_controlnet:
            # ControlNet models are simpler - just need config.json and model weights
            logger.info("ControlNet model integrity check passed")
        else:
            # Check for critical directories in regular diffusion models
            critical_dirs = ['transformer', 'text_encoder', 'text_encoder_2', 'tokenizer', 'tokenizer_2']
            for critical_dir in critical_dirs:
                dir_path = local_path / critical_dir
                if dir_path.exists():
                    # Check if directory has any non-empty files
                    has_content = False
                    for file_path in dir_path.rglob('*'):
                        if file_path.is_file() and file_path.stat().st_size > 0:
                            # Skip ignored files
                            should_ignore = any(pattern in str(file_path) for pattern in ignore_patterns)
                            if not should_ignore:
                                has_content = True
                                break
                    
                    if not has_content:
                        logger.warning(f"Critical directory {critical_dir} appears to be empty or incomplete")
                        return False
        
        logger.info("Download integrity check passed")
        return True
        
    except Exception as e:
        logger.error(f"Error checking download integrity: {e}")
        return False
