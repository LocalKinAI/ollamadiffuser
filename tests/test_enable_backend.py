"""Tests for `ollamadiffuser enable <backend>` (Tier 0 install-friction fix).

These never actually run pip — `subprocess.run` is mocked. They verify
the command each backend builds, the platform guard for MLX, the Metal
build-flag injection for GGUF on Mac, and the exit-code plumbing.
"""
import sys
from unittest.mock import MagicMock, patch

import pytest

from ollamadiffuser.cli import commands
from ollamadiffuser.cli.commands import _build_enable_command, enable_backend


# --------------------------------------------------------------------------
# _build_enable_command — pure, platform-parametrized
# --------------------------------------------------------------------------

class TestBuildEnableCommand:
    def test_mlx_on_apple_silicon(self):
        with patch("platform.system", return_value="Darwin"), \
             patch("platform.machine", return_value="arm64"):
            argv, env, note = _build_enable_command("mlx")
        assert argv[:4] == [sys.executable, "-m", "pip", "install"]
        assert any("mflux" in a for a in argv)
        assert env is None
        assert "2-3x" in note or "2-3×" in note

    def test_mlx_rejected_on_non_apple_silicon(self):
        with patch("platform.system", return_value="Linux"), \
             patch("platform.machine", return_value="x86_64"):
            with pytest.raises(ValueError, match="Apple Silicon"):
                _build_enable_command("mlx")

    def test_mlx_rejected_on_intel_mac(self):
        with patch("platform.system", return_value="Darwin"), \
             patch("platform.machine", return_value="x86_64"):
            with pytest.raises(ValueError, match="Apple Silicon"):
                _build_enable_command("mlx")

    def test_gguf_sets_metal_flag_on_mac(self, monkeypatch):
        monkeypatch.delenv("CMAKE_ARGS", raising=False)
        with patch("platform.system", return_value="Darwin"), \
             patch("platform.machine", return_value="arm64"):
            argv, env, _ = _build_enable_command("gguf")
        assert any("stable-diffusion-cpp-python" in a for a in argv)
        assert any(a.startswith("gguf") for a in argv)
        assert env == {"CMAKE_ARGS": "-DSD_METAL=ON"}

    def test_gguf_builds_cuda_when_nvidia_is_present(self, monkeypatch):
        """The bug this guards: on a CUDA box the build was silently CPU-only."""
        monkeypatch.delenv("CMAKE_ARGS", raising=False)
        with patch("platform.system", return_value="Linux"), \
             patch("platform.machine", return_value="x86_64"), \
             patch.object(commands, "_has_nvidia_gpu", return_value=True):
            argv, env, note = _build_enable_command("gguf")
        assert any("stable-diffusion-cpp-python" in a for a in argv)
        assert env == {"CMAKE_ARGS": "-DSD_CUDA=ON"}
        assert "CUDA" in note

    def test_gguf_without_a_gpu_says_it_is_a_cpu_build(self, monkeypatch):
        monkeypatch.delenv("CMAKE_ARGS", raising=False)
        with patch("platform.system", return_value="Linux"), \
             patch("platform.machine", return_value="x86_64"), \
             patch.object(commands, "_has_nvidia_gpu", return_value=False):
            _, env, note = _build_enable_command("gguf")
        assert env is None
        assert "CPU build" in note

    def test_gguf_leaves_a_users_cmake_args_alone(self, monkeypatch):
        """Someone who exported -DSD_VULKAN=ON must not get CUDA or Metal instead."""
        monkeypatch.setenv("CMAKE_ARGS", "-DSD_VULKAN=ON")
        with patch("platform.system", return_value="Linux"), \
             patch.object(commands, "_has_nvidia_gpu", return_value=True):
            _, env, note = _build_enable_command("gguf")
        assert env is None          # nothing overridden; theirs is inherited
        assert "your CMAKE_ARGS" in note

    def test_mcp_builds_cli_extra(self):
        argv, env, _ = _build_enable_command("mcp")
        assert any("mcp[cli]" in a for a in argv)
        assert env is None

    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError, match="Unknown backend"):
            _build_enable_command("bogus")


# --------------------------------------------------------------------------
# enable_backend — exit-code plumbing, subprocess mocked
# --------------------------------------------------------------------------

class TestEnableBackend:
    def test_success_returns_zero(self):
        with patch("platform.system", return_value="Darwin"), \
             patch("platform.machine", return_value="arm64"), \
             patch.object(commands.subprocess, "run",
                          return_value=MagicMock(returncode=0)) as run_mock:
            code = enable_backend("mlx")
        assert code == 0
        run_mock.assert_called_once()

    def test_pip_failure_propagates_exit_code(self):
        with patch("platform.system", return_value="Darwin"), \
             patch("platform.machine", return_value="arm64"), \
             patch.object(commands.subprocess, "run",
                          return_value=MagicMock(returncode=1)):
            code = enable_backend("gguf")
        assert code == 1

    def test_platform_guard_returns_one_without_running_pip(self):
        with patch("platform.system", return_value="Linux"), \
             patch("platform.machine", return_value="x86_64"), \
             patch.object(commands.subprocess, "run") as run_mock:
            code = enable_backend("mlx")
        assert code == 1
        run_mock.assert_not_called()  # never reached pip

    def test_gguf_passes_metal_env_to_subprocess(self, monkeypatch):
        monkeypatch.delenv("CMAKE_ARGS", raising=False)
        captured = {}

        def fake_run(argv, env=None):
            captured["env"] = env
            return MagicMock(returncode=0)

        with patch("platform.system", return_value="Darwin"), \
             patch("platform.machine", return_value="arm64"), \
             patch.object(commands.subprocess, "run", side_effect=fake_run):
            enable_backend("gguf")

        assert captured["env"] is not None
        assert captured["env"].get("CMAKE_ARGS") == "-DSD_METAL=ON"

    def test_unknown_backend_returns_one(self):
        with patch.object(commands.subprocess, "run") as run_mock:
            code = enable_backend("bogus")
        assert code == 1
        run_mock.assert_not_called()


class TestHasNvidiaGpu:
    def test_no_nvidia_smi_means_no(self):
        with patch("shutil.which", return_value=None):
            assert commands._has_nvidia_gpu() is False

    def test_a_working_nvidia_smi_means_yes(self):
        with patch("shutil.which", return_value="/usr/bin/nvidia-smi"), \
             patch.object(commands.subprocess, "run",
                          return_value=MagicMock(returncode=0)):
            assert commands._has_nvidia_gpu() is True

    def test_a_broken_driver_means_no(self):
        """nvidia-smi installed but failing (no driver loaded) is not a GPU."""
        with patch("shutil.which", return_value="/usr/bin/nvidia-smi"), \
             patch.object(commands.subprocess, "run",
                          return_value=MagicMock(returncode=9)):
            assert commands._has_nvidia_gpu() is False
