"""`pull` links weights another tool already downloaded instead of fetching them.

The hub cache is faked on disk in tmp_path, laid out the way huggingface_hub
lays it out — ``models--org--name/blobs/<content hash>`` — and the repo's file
list comes from a stubbed ``HfApi``. Nothing touches the network.
"""
from __future__ import annotations

import hashlib
import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from ollamadiffuser.core.utils import download_utils
from ollamadiffuser.core.utils.download_utils import (
    link_from_hf_cache,
    robust_snapshot_download,
)

REPO = "acme/model"


def _file(name, content, lfs):
    """A repo file as HfApi reports it, plus the name its blob has in the cache."""
    sha1 = hashlib.sha1(content).hexdigest()
    if lfs:
        sha256 = hashlib.sha256(content).hexdigest()
        sib = SimpleNamespace(rfilename=name, size=len(content), blob_id=sha1,
                              lfs={"sha256": sha256, "size": len(content)})
        return sib, sha256
    return SimpleNamespace(rfilename=name, size=len(content), blob_id=sha1, lfs=None), sha1


FILES = {
    "model_index.json": (b'{"_class_name": "X"}', False),
    "transformer/config.json": (b'{"a": 1}', False),
    "transformer/model.safetensors": (b"W" * 4096, True),
    "vae/model.safetensors": (b"V" * 2048, True),
    "comfyui/single_file.safetensors": (b"C" * 8192, True),   # what patterns drop
}


@pytest.fixture
def hub(tmp_path, monkeypatch):
    """A populated hub cache and the stubbed file list; returns helpers."""
    cache = tmp_path / "hub"
    blobs = cache / ("models--" + REPO.replace("/", "--")) / "blobs"
    blobs.mkdir(parents=True)
    siblings = []
    for name, (content, lfs) in FILES.items():
        sib, blob_name = _file(name, content, lfs)
        (blobs / blob_name).write_bytes(content)
        siblings.append(sib)
    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(cache))
    api = SimpleNamespace(repo_info=lambda **kw: SimpleNamespace(siblings=siblings))
    monkeypatch.setattr(download_utils, "HfApi", lambda: api)
    return SimpleNamespace(cache=cache, blobs=blobs, siblings=siblings,
                           target=tmp_path / "models" / "my-model")


def _blob_for(hub, name):
    content, lfs = FILES[name]
    return hub.blobs / _file(name, content, lfs)[1]


class TestLinkFromHfCache:
    def test_every_file_present_is_linked_not_copied(self, hub):
        assert link_from_hf_cache(REPO, str(hub.target)) is True
        for name in FILES:
            dst = hub.target / name
            assert dst.is_file() and not dst.is_symlink()
            # Same inode as the cache blob: a hard link, no second copy on disk.
            assert os.path.samefile(dst, _blob_for(hub, name))

    def test_patterns_decide_what_is_needed_and_linked(self, hub):
        ok = link_from_hf_cache(REPO, str(hub.target),
                                ignore_patterns=["comfyui/*"])
        assert ok is True
        assert not (hub.target / "comfyui").exists()
        assert (hub.target / "transformer/model.safetensors").exists()

    def test_a_missing_blob_means_download_and_leaves_nothing(self, hub):
        _blob_for(hub, "vae/model.safetensors").unlink()
        assert link_from_hf_cache(REPO, str(hub.target)) is False
        assert not hub.target.exists() or not any(hub.target.rglob("*"))

    def test_a_missing_file_outside_the_patterns_does_not_matter(self, hub):
        _blob_for(hub, "comfyui/single_file.safetensors").unlink()
        assert link_from_hf_cache(REPO, str(hub.target),
                                  ignore_patterns=["comfyui/*"]) is True

    def test_a_blob_of_the_wrong_size_is_not_trusted(self, hub):
        _blob_for(hub, "transformer/model.safetensors").write_bytes(b"W" * 10)
        assert link_from_hf_cache(REPO, str(hub.target)) is False

    def test_no_cached_repo_means_no_network_call(self, hub, monkeypatch):
        import shutil
        shutil.rmtree(hub.blobs.parent)
        called = []
        monkeypatch.setattr(download_utils, "HfApi",
                            lambda: called.append(1) or SimpleNamespace())
        assert link_from_hf_cache(REPO, str(hub.target)) is False
        assert called == []

    def test_a_listing_failure_falls_back_to_downloading(self, hub, monkeypatch):
        def boom(**kw):
            raise ConnectionError("offline")
        monkeypatch.setattr(download_utils, "HfApi",
                            lambda: SimpleNamespace(repo_info=boom))
        assert link_from_hf_cache(REPO, str(hub.target)) is False

    def test_another_disk_means_download(self, hub):
        real_stat = os.stat
        def fake_stat(path, *a, **kw):
            st = real_stat(path, *a, **kw)
            if str(path).startswith(str(hub.blobs)):
                return os.stat_result((st.st_mode, st.st_ino, st.st_dev + 1) + tuple(st)[3:])
            return st
        with patch.object(download_utils.os, "stat", side_effect=fake_stat):
            assert link_from_hf_cache(REPO, str(hub.target)) is False

    def test_a_link_failure_midway_removes_the_links_already_made(self, hub):
        real_link = os.link
        calls = []
        def flaky(src, dst):
            calls.append(dst)
            if len(calls) == 3:
                raise OSError(18, "Invalid cross-device link")
            return real_link(src, dst)
        with patch.object(download_utils.os, "link", side_effect=flaky):
            assert link_from_hf_cache(REPO, str(hub.target)) is False
        assert not [p for p in hub.target.rglob("*") if p.is_file()]

    def test_a_partial_file_from_an_earlier_pull_is_replaced(self, hub):
        partial = hub.target / "transformer/model.safetensors"
        partial.parent.mkdir(parents=True)
        partial.write_bytes(b"W" * 100)
        assert link_from_hf_cache(REPO, str(hub.target)) is True
        assert os.path.samefile(partial, _blob_for(hub, "transformer/model.safetensors"))


class TestPullUsesTheCache:
    def test_a_cached_repo_is_never_downloaded(self, hub):
        with patch.object(download_utils, "snapshot_download") as dl:
            out = robust_snapshot_download(repo_id=REPO, local_dir=str(hub.target))
        dl.assert_not_called()
        assert out == str(hub.target)
        assert (hub.target / "vae/model.safetensors").exists()

    def test_force_download_skips_the_cache(self, hub):
        with patch.object(download_utils, "link_from_hf_cache") as link, \
             patch.object(download_utils, "get_repo_file_list", return_value={}), \
             patch.object(download_utils, "snapshot_download",
                          return_value=str(hub.target)) as dl:
            robust_snapshot_download(repo_id=REPO, local_dir=str(hub.target),
                                     force_download=True)
        link.assert_not_called()
        dl.assert_called_once()

    def test_an_incomplete_cache_still_downloads(self, hub):
        _blob_for(hub, "vae/model.safetensors").unlink()
        with patch.object(download_utils, "get_repo_file_list", return_value={}), \
             patch.object(download_utils, "snapshot_download",
                          return_value=str(hub.target)) as dl:
            robust_snapshot_download(repo_id=REPO, local_dir=str(hub.target))
        dl.assert_called_once()


# --------------------------------------------------------------------------
# Another installed entry from the same repo
# --------------------------------------------------------------------------

@pytest.fixture
def sibling(hub, tmp_path):
    """An installed model folder holding the repo's files (no hub cache needed)."""
    import shutil
    shutil.rmtree(hub.cache)                      # only the installed copy is on disk
    folder = tmp_path / "models" / "my-model-edit"
    for name, (content, _) in FILES.items():
        f = folder / name
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(content)
    return folder


def _write_metadata(folder, name, etag):
    meta = folder / ".cache" / "huggingface" / "download" / f"{name}.metadata"
    meta.parent.mkdir(parents=True, exist_ok=True)
    meta.write_text(f"deadbeef\n{etag}\n1700000000.0\n")


class TestLinkFromInstalledCopies:
    def test_files_another_entry_downloaded_are_linked(self, hub, sibling):
        from ollamadiffuser.core.utils.download_utils import link_from_local_dirs
        messages = []
        ok = link_from_local_dirs(REPO, str(hub.target), [str(sibling)],
                                  progress_callback=messages.append)
        assert ok is True
        for name in FILES:
            assert os.path.samefile(hub.target / name, sibling / name)
        assert "installed as my-model-edit" in messages[-1]

    def test_a_recorded_hash_that_disagrees_rules_the_file_out(self, hub, sibling):
        from ollamadiffuser.core.utils.download_utils import link_from_local_dirs
        _write_metadata(sibling, "vae/model.safetensors", "0" * 64)
        assert link_from_local_dirs(REPO, str(hub.target), [str(sibling)]) is False
        assert not hub.target.exists() or not any(hub.target.rglob("*.safetensors"))

    def test_a_recorded_hash_that_agrees_is_accepted(self, hub, sibling):
        from ollamadiffuser.core.utils.download_utils import link_from_local_dirs
        content, lfs = FILES["vae/model.safetensors"]
        _write_metadata(sibling, "vae/model.safetensors", hashlib.sha256(content).hexdigest())
        assert link_from_local_dirs(REPO, str(hub.target), [str(sibling)]) is True

    def test_a_wrong_size_or_a_missing_file_downloads_instead(self, hub, sibling):
        from ollamadiffuser.core.utils.download_utils import link_from_local_dirs
        (sibling / "vae/model.safetensors").write_bytes(b"short")
        assert link_from_local_dirs(REPO, str(hub.target), [str(sibling)]) is False
        (sibling / "vae/model.safetensors").unlink()
        assert link_from_local_dirs(REPO, str(hub.target), [str(sibling)]) is False

    def test_its_own_folder_is_not_a_source(self, hub):
        from ollamadiffuser.core.utils.download_utils import link_from_local_dirs
        assert link_from_local_dirs(REPO, str(hub.target), [str(hub.target)]) is False

    def test_pull_links_from_an_installed_copy_before_downloading(self, hub, sibling):
        with patch.object(download_utils, "snapshot_download") as dl:
            out = robust_snapshot_download(repo_id=REPO, local_dir=str(hub.target),
                                           local_sources=[str(sibling)])
        dl.assert_not_called()
        assert out == str(hub.target)
        assert os.path.samefile(hub.target / "transformer/model.safetensors",
                                sibling / "transformer/model.safetensors")

    def test_force_skips_installed_copies_too(self, hub, sibling):
        with patch.object(download_utils, "link_from_local_dirs") as link, \
             patch.object(download_utils, "get_repo_file_list", return_value={}), \
             patch.object(download_utils, "snapshot_download", return_value=str(hub.target)):
            robust_snapshot_download(repo_id=REPO, local_dir=str(hub.target),
                                     local_sources=[str(sibling)], force_download=True)
        link.assert_not_called()


class TestInstalledCopiesLookup:
    def test_only_other_installed_entries_from_the_same_repo(self, monkeypatch):
        from types import SimpleNamespace as NS
        from ollamadiffuser.core.models import manager as mgr
        installed = {
            "qwen-image-2.1-mlx": NS(path="/m/qwen-image-2.1-mlx"),
            "qwen-image-2.1-edit-mlx": NS(path="/m/qwen-image-2.1-edit-mlx"),
            "z-image-turbo-mlx": NS(path="/m/z-image-turbo-mlx"),
        }
        repos = {"qwen-image-2.1-mlx": "Qwen/Qwen-Image-2.1",
                 "qwen-image-2.1-edit-mlx": "Qwen/Qwen-Image-2.1",
                 "z-image-turbo-mlx": "Tongyi-MAI/Z-Image-Turbo"}
        monkeypatch.setattr(mgr.settings, "models", installed)
        monkeypatch.setattr(mgr.model_registry, "get_model", lambda n: {"repo_id": repos.get(n)})
        folders = mgr.model_manager._installed_copies("qwen-image-2.1-controlnet-mlx", "Qwen/Qwen-Image-2.1")
        assert sorted(folders) == ["/m/qwen-image-2.1-edit-mlx", "/m/qwen-image-2.1-mlx"]
        assert mgr.model_manager._installed_copies("qwen-image-2.1-mlx", "Qwen/Qwen-Image-2.1") == \
            ["/m/qwen-image-2.1-edit-mlx"]
