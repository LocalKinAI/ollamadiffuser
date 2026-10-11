"""`ollamadiffuser generate / edit / video`: one command, one file, an exit code."""
import json

import pytest
from click.testing import CliRunner
from PIL import Image

from ollamadiffuser.cli import job_commands
from ollamadiffuser.core import jobs

RESULT = {"path": "/tmp/out.png", "kind": "image", "model": "z-image-turbo-mlx", "seed": 42,
          "elapsed_s": 3.1, "server": None}


@pytest.fixture
def ran(monkeypatch):
    monkeypatch.delenv(jobs.SERVER_ENV, raising=False)
    calls = []

    def run(job, output=None, force=False, server=None, **kwargs):
        print("a loader talking on stdout")  # must not reach the JSON on stdout
        calls.append({"job": job, "output": output, "force": force, "server": server})
        return RESULT

    monkeypatch.setattr(jobs, "run", run)
    return calls


def test_generate_json_is_one_object_on_stdout(ran):
    result = CliRunner().invoke(job_commands.generate, [
        "a fox", "-m", "z-image-turbo-mlx", "--seed", "42", "--steps", "8", "-W", "768",
        "--scheduler", "viggle_turbo", "--json", "--force"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == RESULT
    job = ran[0]["job"]
    assert (job.kind, job.prompt, job.model, job.seed, job.steps, job.width, job.height) == \
        ("image", "a fox", "z-image-turbo-mlx", 42, 8, 768, 1024)
    assert job.scheduler == "viggle_turbo" and ran[0]["force"] is True


def test_without_json_the_path_is_what_stdout_says(ran):
    result = CliRunner().invoke(job_commands.generate, ["a fox", "-o", "fox.png"])
    assert result.exit_code == 0
    assert result.stdout == "/tmp/out.png\n" and ran[0]["output"] == "fox.png"


def test_an_error_is_json_and_exit_code_1(monkeypatch):
    def refuse(job, **kwargs):
        raise jobs.JobError("No model loaded or named")

    monkeypatch.setattr(jobs, "run", refuse)
    result = CliRunner().invoke(job_commands.generate, ["a fox", "--json"])
    assert result.exit_code == 1 and json.loads(result.stdout) == {"error": "No model loaded or named"}


def test_an_unexpected_failure_is_still_json(monkeypatch):
    def broken(job, **kwargs):
        raise KeyError("parameters")

    monkeypatch.setattr(jobs, "run", broken)
    result = CliRunner().invoke(job_commands.generate, ["a fox", "--json"])
    assert result.exit_code == 1 and json.loads(result.stdout)["error"].startswith("KeyError")


def test_edit_takes_the_picture_then_the_references(ran, tmp_path):
    a, b, m = (tmp_path / n for n in ("a.png", "b.png", "m.png"))
    for f in (a, b, m):
        Image.new("RGB", (8, 8)).save(f)
    result = CliRunner().invoke(job_commands.edit, [
        "put the person from image 2 in image 1", "-i", str(a), "-i", str(b), "--mask", str(m),
        "--strength", "0.6", "-m", "qwen-image-2.1-edit-mlx", "--json"])
    assert result.exit_code == 0, result.output
    job = ran[0]["job"]
    assert job.kind == "edit" and job.image == str(a) and job.references == [str(b)]
    assert job.mask == str(m) and job.strength == 0.6


def test_edit_without_a_picture_is_refused_by_the_cli(ran):
    result = CliRunner().invoke(job_commands.edit, ["make it night"])
    assert result.exit_code != 0 and not ran


def test_video(ran, tmp_path):
    first = tmp_path / "first.png"
    Image.new("RGB", (8, 8)).save(first)
    result = CliRunner().invoke(job_commands.video, [
        "she turns", "-i", str(first), "-s", "4", "-m", "ltx-2.3-mlx-q8", "--json"])
    assert result.exit_code == 0, result.output
    job = ran[0]["job"]
    assert (job.kind, job.image, job.seconds, job.model) == ("video", str(first), 4.0, "ltx-2.3-mlx-q8")


def test_the_commands_are_on_the_cli():
    from ollamadiffuser.cli.main import cli
    assert {"generate", "edit", "video"} <= set(cli.commands)


def test_list_json(monkeypatch):
    from ollamadiffuser.cli import model_commands
    monkeypatch.setattr(model_commands.model_manager, "list_installed_models", lambda: ["z-image-turbo-mlx"])
    monkeypatch.setattr(model_commands.model_manager, "get_model_info", lambda name: {
        "model_type": "mlx", "parameters": {"mlx_variant": "z_image"}, "local_path": "/m/z",
        "size": "8 GB", "hardware_requirements": {"min_vram_gb": 6, "performance_notes": "fast"}})
    result = CliRunner().invoke(model_commands.list, ["--json"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == [{"name": "z-image-turbo-mlx", "type": "mlx", "mlx_variant": "z_image",
                                          "min_memory_gb": 6, "size": "8 GB", "path": "/m/z", "notes": "fast"}]


def test_run_passes_force_and_says_why_a_load_failed(monkeypatch):
    from ollamadiffuser.cli import model_commands
    mm = model_commands.model_manager
    monkeypatch.setattr(mm, "is_model_installed", lambda name: True)
    monkeypatch.setattr(mm, "last_error", None)  # restored after the test
    seen = {}

    def load(name, force=False, persist=True):
        seen["force"] = force
        mm.last_error = "qwen-image-2.1-mlx needs about 46 GB of memory and 12.0 GB is free."
        return False

    monkeypatch.setattr(mm, "load_model", load)
    result = CliRunner().invoke(model_commands.run, ["qwen-image-2.1-mlx", "--force"], terminal_width=200)
    assert result.exit_code == 1 and seen["force"] is True
    assert "needs about 46 GB" in result.output


def test_server_from_the_option_or_the_environment(ran):
    CliRunner().invoke(job_commands.generate, ["a fox", "--server", "192.168.0.21:8010", "--json"])
    assert ran[-1]["server"] == "192.168.0.21:8010"
    CliRunner(env={jobs.SERVER_ENV: "http://192.168.0.21:8013"}).invoke(job_commands.edit, [
        "make it night", "-i", __file__.replace("test_job_commands.py", "__init__.py"), "--json"])
    assert ran[-1]["server"] == "http://192.168.0.21:8013"
    CliRunner().invoke(job_commands.video, ["waves", "--json"])
    assert ran[-1]["server"] is None
