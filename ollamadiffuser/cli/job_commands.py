"""One-off jobs: ``generate``, ``edit`` and ``video``.

One command, one file, and the way out is the exit code — what a script or
an agent with a shell wants. The job goes to a running server that holds the
model, or loads it here and lets it go afterwards (core/jobs.py). ``--json``
prints the result as one JSON object on stdout, and an error as
``{"error": ...}`` with exit code 1; everything else goes to stderr.
"""

import contextlib
import json
import logging
import sys

import click
from rich.console import Console

from ..core import jobs

err_console = Console(stderr=True)


def _job_options(f):
    """The options every kind of job takes."""
    for option in reversed([
        click.option("--model", "-m", help="Model to use. Left out: the one a running server holds."),
        click.option("--output", "-o", type=click.Path(dir_okay=False),
                     help="File to write. Default: ollamadiffuser-<time>-<seed> in this folder."),
        click.option("--seed", type=int, help="Seed. Left out: one is picked, and printed."),
        click.option("--steps", type=int, help="Denoising steps. Default: the model's."),
        click.option("--guidance", "-g", type=float, help="Guidance scale. Default: the model's."),
        click.option("--json", "as_json", is_flag=True, help="Print the result as JSON (for scripts and agents)."),
        click.option("--force", is_flag=True, help="Load the model even if free memory looks too low."),
    ]):
        f = option(f)
    return f


def _lora_options(f):
    for option in reversed([
        click.option("--lora", help="A LoRA: a .safetensors file or a Hugging Face repo."),
        click.option("--lora-weight", help="The file inside that repo, when it holds several."),
        click.option("--lora-scale", type=float, default=1.0, show_default=True, help="How strongly to apply it."),
    ]):
        f = option(f)
    return f


def _run(job: jobs.Job, output, as_json: bool, force: bool) -> None:
    root = logging.getLogger()
    if root.level > logging.DEBUG:  # -v keeps the logs
        root.setLevel(logging.WARNING)
    try:
        # stdout is for the result: whatever a loader prints goes to stderr.
        with contextlib.redirect_stdout(sys.stderr):
            if not as_json:
                where = job.model or "the running server's model"
                err_console.print(f"[dim]Making {'a video' if job.kind == 'video' else 'an image'} with {where}…[/dim]")
            result = jobs.run(job, output=output, force=force)
    except Exception as e:
        message = str(e) if isinstance(e, jobs.JobError) else f"{type(e).__name__}: {e}"
        if as_json:
            click.echo(json.dumps({"error": message}, ensure_ascii=False))
        else:
            err_console.print(f"[red]{message}[/red]")
        sys.exit(1)
    if as_json:
        click.echo(json.dumps(result, ensure_ascii=False))
        return
    via = f", via {result['server']}" if result["server"] else ""
    click.echo(result["path"])
    err_console.print(f"[green]Saved[/green] [dim]({result['model']}, seed {result['seed']}, "
                      f"{result['elapsed_s']} s{via})[/dim]")


@click.command()
@click.argument("prompt")
@_job_options
@click.option("--width", "-W", type=int, default=1024, show_default=True)
@click.option("--height", "-H", type=int, default=1024, show_default=True)
@click.option("--negative", "-n", help="What to keep out of the picture.")
@click.option("--scheduler", help="A sampling schedule the model takes (e.g. viggle_turbo).")
@_lora_options
def generate(prompt, model, output, seed, steps, guidance, as_json, force,
             width, height, negative, scheduler, lora, lora_weight, lora_scale):
    """Make an image from a prompt, and print where it went.

    \b
    Examples:
      ollamadiffuser generate "a red fox in snow" -m z-image-turbo-mlx
      ollamadiffuser generate "a poster that reads 海边小屋" -m qwen-image-2.1-turbo-mlx -o poster.png --json
    """
    _run(jobs.Job(kind="image", prompt=prompt, model=model, negative_prompt=negative,
                  width=width, height=height, steps=steps, guidance=guidance, seed=seed,
                  scheduler=scheduler, lora=lora, lora_weight=lora_weight, lora_scale=lora_scale),
         output, as_json, force)


@click.command()
@click.argument("prompt")
@click.option("--image", "-i", "images", multiple=True, required=True,
              type=click.Path(exists=True, dir_okay=False),
              help="The picture to edit. Repeat for more references (models that take several); "
                   "the first sets the size.")
@_job_options
@click.option("--mask", type=click.Path(exists=True, dir_okay=False),
              help="Where to change the picture (white), for models that take a mask.")
@click.option("--strength", type=float, help="How far to move from the picture. Default: the model's.")
@click.option("--negative", "-n", help="What to keep out of the picture.")
@click.option("--scheduler", help="A sampling schedule the model takes.")
@_lora_options
def edit(prompt, images, model, output, seed, steps, guidance, as_json, force,
         mask, strength, negative, scheduler, lora, lora_weight, lora_scale):
    """Edit a picture by instruction, and print where the result went.

    \b
    Examples:
      ollamadiffuser edit "make it a snowy night" -i cabin.png -m qwen-image-2.1-edit-mlx
      ollamadiffuser edit "put the person from image 2 in image 1" -i room.png -i face.png -m flux.2-klein-4b-edit-mlx
    """
    _run(jobs.Job(kind="edit", prompt=prompt, model=model, image=images[0], references=list(images[1:]),
                  mask=mask, strength=strength, negative_prompt=negative, steps=steps,
                  guidance=guidance, seed=seed, scheduler=scheduler, lora=lora,
                  lora_weight=lora_weight, lora_scale=lora_scale),
         output, as_json, force)


@click.command()
@click.argument("prompt")
@_job_options
@click.option("--seconds", "-s", type=float, help="Length. Default: the model's.")
@click.option("--frames", type=int, help="Length in frames (wins over --seconds).")
@click.option("--width", "-W", type=int, help="Default: the model's.")
@click.option("--height", "-H", type=int, help="Default: the model's.")
@click.option("--image", "-i", type=click.Path(exists=True, dir_okay=False), help="First frame (image-to-video).")
@click.option("--audio", type=click.Path(exists=True, dir_okay=False), help="An audio track to drive it.")
@click.option("--control", type=click.Path(exists=True, dir_okay=False),
              help="A reference video to follow (pose, depth, edges).")
@click.option("--lora", help="The control LoRA for --control (a file or a Hugging Face repo).")
@click.option("--lora-scale", type=float, help="How strongly to apply it.")
def video(prompt, model, output, seed, steps, guidance, as_json, force,
          seconds, frames, width, height, image, audio, control, lora, lora_scale):
    """Make a video, and print where it went.

    \b
    Examples:
      ollamadiffuser video "waves on a beach at dusk" -m ltx-2.3-mlx-q8 -s 4
      ollamadiffuser video "she turns and smiles" -i portrait.png -m ltx-2.3-mlx-q8 --json
    """
    _run(jobs.Job(kind="video", prompt=prompt, model=model, seconds=seconds, frames=frames,
                  width=width, height=height, image=image, audio=audio, control=control,
                  steps=steps, guidance=guidance, seed=seed, lora=lora,
                  lora_scale=lora_scale if lora_scale is not None else 1.0),
         output, as_json, force)
