"""
Command-line interface for the taggy package.
Provides commands for tagging images, searching images, and finding duplicates.
Now supports both sync and async execution via --async-run.
"""

import os
import click
import asyncio

from utils.file_utils import (
    create_directory,
    load_config,
    perform_file_operation,
    save_metadata_to_json
)
from utils.logger import get_logger
from utils.image_tagger import ImageTagger

logger = get_logger(__name__)


@click.group()
@click.option('--config-file', default='config.ini', help='Path to the config file.')
@click.option('--images-path', '-i', type=click.Path(exists=True), help='Absolute path to the folder containing images.')
@click.option('--output-path', '-o', type=click.Path(), help="Path for output (copying images or saving shortcuts).")
@click.pass_context
def cli(ctx, images_path, output_path, config_file):
    """
    Taggy CLI tool for tagging, searching, and organizing images.
    """
    ctx.ensure_object(dict)
    defaults = load_config(config_file)
    ctx.obj['defaults'] = defaults

    if not images_path:
        images_path = defaults.get("images_path", "./images")
    ctx.obj['images_path'] = images_path

    if not output_path:
        output_path = defaults.get("output_path", "./output")
    ctx.obj['output_path'] = output_path

    tagger = ImageTagger(model_name="CLIP")
    ctx.obj['tagger'] = tagger


@cli.command("tag")
@click.option('--threshold', '-t', type=float, help='Minimum probability threshold for assigning a label.')
@click.option('--metadata-output', '-m', help='File in which metadata will be saved if operation includes "metadata".')
@click.option('--operation', '-op', multiple=True, type=click.Choice(['copy', 'symlink', 'metadata']),
              help='File operation(s) to perform on images.')
@click.option('--labels', '-l', multiple=True, help='List of labels used for tagging.')
@click.option('--async-run', is_flag=True, default=False, help="Run asynchronously?")
@click.pass_context
def taggy(ctx, threshold, metadata_output, operation, labels, async_run):
    """
    Assigns labels to each image in the images_path using the CLIP model
    and performs the specified operation(s) on these images.
    If --async-run is set, uses asynchronous methods.
    """
    defaults = ctx.obj['defaults']
    images_path = ctx.obj['images_path']
    tagger = ctx.obj['tagger']

    if not labels:
        labels_from_config = defaults.get("labels", "other, pets, people").split(',')
        labels = [label.strip() for label in labels_from_config]

    if not operation:
        operation = [defaults.get("operation", "copy")]

    if threshold is None:
        threshold = float(defaults.get("threshold", 0.3))

    if not metadata_output:
        metadata_output = defaults.get("metadata_output", "metadata.json")

    logger.info("Running 'tag' command for tagging images.")
    logger.debug(
        f"images_path={images_path}, labels={labels}, operation={operation}, "
        f"threshold={threshold}, metadata_output={metadata_output}, async_run={async_run}"
    )

    image_files = [
        os.path.join(images_path, f) for f in os.listdir(images_path)
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp'))
    ]

    metadata_list = []

    # --- ASYNC VERSION ---
    async def async_main():
        for file_path in image_files:
            results = await tagger.tag_image_async(
                file_path, labels=labels, threshold=threshold
            )
            assigned_labels = [r["tag"] for r in results]
            logger.info(f"[Async] Processing file: {file_path}")
            for label in assigned_labels:
                label_dir = os.path.join(images_path, label)
                create_directory(label_dir)
                for op in operation:
                    if op != 'metadata':
                        perform_file_operation(file_path, label_dir, op)
            if 'metadata' in operation:
                metadata_list.append({
                    "file": file_path,
                    "assigned_labels": assigned_labels,
                    "full_results": results
                })

        if 'metadata' in operation and metadata_list:
            output_file = os.path.join(images_path, metadata_output)
            save_metadata_to_json(metadata_list, output_file)

    # --- SYNC VERSION ---
    def sync_main():
        for file_path in image_files:
            results = tagger.tag_image(
                file_path, labels=labels, threshold=threshold
            )
            assigned_labels = [r["tag"] for r in results]
            logger.info(f"[Sync] Processing file: {file_path}")
            for label in assigned_labels:
                label_dir = os.path.join(images_path, label)
                create_directory(label_dir)
                for op in operation:
                    if op != 'metadata':
                        perform_file_operation(file_path, label_dir, op)
            if 'metadata' in operation:
                metadata_list.append({
                    "file": file_path,
                    "assigned_labels": assigned_labels,
                    "full_results": results
                })

        if 'metadata' in operation and metadata_list:
            output_file = os.path.join(images_path, metadata_output)
            save_metadata_to_json(metadata_list, output_file)

    if async_run:
        asyncio.run(async_main())
    else:
        sync_main()


@cli.command("search")
@click.option('--query', '-q', type=str, required=True, help='Text query for searching similar images.')
@click.option('--top-k', '-k', type=int, default=5, help='Number of top results to return.')
@click.option('--async-run', is_flag=True, default=False, help="Run asynchronously?")
@click.pass_context
def search_images(ctx, query, top_k, async_run):
    """
    Searches for images most similar to the provided text query using CLIP embeddings.
    If --async-run is set, uses asynchronous method.
    """
    images_path = ctx.obj['images_path']
    tagger = ctx.obj['tagger']

    logger.info(f"Running 'search' command with query='{query}', top_k={top_k}, async_run={async_run}")

    # --- ASYNC VERSION ---
    async def async_main():
        results = await tagger.search_images_async(query, images_path, top_k=top_k)
        click.echo(f"[Async] Search results for query: '{query}'")
        for i, (file, score) in enumerate(results):
            click.echo(f"{i+1}. {file} (similarity: {score:.4f})")

    # --- SYNC VERSION ---
    def sync_main():
        results = tagger.search_images(query, images_path, top_k=top_k)
        click.echo(f"[Sync] Search results for query: '{query}'")
        for i, (file, score) in enumerate(results):
            click.echo(f"{i+1}. {file} (similarity: {score:.4f})")

    if async_run:
        asyncio.run(async_main())
    else:
        sync_main()


@cli.command("duplicates")
@click.option('--similarity-threshold', '-t', default=0.9, type=float, help="Threshold for considering images duplicates.")
@click.option('--operation', '-op', multiple=True, type=click.Choice(['copy', 'symlink', 'metadata']),
              help='File operation(s) to perform when grouping duplicates.')
@click.option('--async-run', is_flag=True, default=False, help="Run asynchronously?")
@click.pass_context
def find_duplicates(ctx, similarity_threshold, operation, async_run):
    """
    Finds and groups duplicate images based on their embedding similarity.
    If --async-run is set, uses asynchronous method.
    """
    images_path = ctx.obj['images_path']
    output_path = ctx.obj['output_path']
    defaults = ctx.obj['defaults']
    tagger = ctx.obj['tagger']

    logger.info(f"Running 'duplicates' command with threshold={similarity_threshold}, async_run={async_run}")

    if not operation:
        operation = [defaults.get("operation", "copy")]

    # --- ASYNC VERSION ---
    async def async_main():
        duplicates = await tagger.find_duplicates_async(images_path, similarity_threshold)
        await tagger.group_duplicates_async(duplicates, output_path, operation)
        click.echo(f"[Async] Duplicates grouped into folders at {output_path}.")

    # --- SYNC VERSION ---
    def sync_main():
        duplicates = tagger.find_duplicates(images_path, similarity_threshold)
        tagger.group_duplicates(duplicates, output_path, operation)
        click.echo(f"[Sync] Duplicates grouped into folders at {output_path}.")

    if async_run:
        asyncio.run(async_main())
    else:
        sync_main()


if __name__ == "__main__":
    cli()
