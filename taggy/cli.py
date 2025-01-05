""""
Command-line interface for the taggy package.
"""
import os
import click
from utils.FILE_utils import  create_directory, load_config, perform_file_operation, \
    save_metadata_to_json

from utils.logger import get_logger
from ai_tagger import ImageTagger

logger = get_logger(__name__)



    
@click.group()
@click.option('--config-file', default='config.ini', help='Plik konfiguracyjny.')
@click.option('--images-path', '-i', type=click.Path(exists=True), help='Absolutna cieżka do folderu z obrazami.')
@click.option('--output-path', '-o', type=click.Path(),help="Ścieżka wyjściowa do zapisywania kopiowanych grafik bądź skrótów.")
@click.pass_context
def cli(ctx, images_path, output_path, config_file,):
    """Taggy - tool for tagging, searching and organizing images."""
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
@click.option('--threshold', '-t', type=float,
              help='Minimalny próg prawdopodobieństwa dla przypisania etykiety.')
@click.option('--metadata-output', '-m',
              help='Plik, w którym zapisywane będą metadane (jeśli operation=metadata).')
@click.option('--operation', '-o', multiple=True, type=click.Choice(['copy', 'symlink',  'metadata']),
              help='Wybierz, czy pliki mają być kopiowane, tworzone skróty, czy zapisywane metadane.')
@click.option('--labels', '-l', multiple=True, help='Lista etykiet używana przy tagowaniu zdjęć i grupowaniu duplikatów')
@click.pass_context
def taggy(ctx, threshold=None, metadata_output=None, operation=None, labels=None):
        """
        For each image in the images_path, assign labels using CLIP and perform an operation on the image.
        
        """
        defaults = ctx.obj['defaults']
        images_path = ctx.obj['images_path']
        tagger = ctx.obj['tagger']
        
        if not labels:
            labels_from_config = defaults.get("labels", "other, pets, people").split(',')
            labels = [label.strip() for label in labels_from_config]
        if not operation:
            operation = defaults.get("operation", "copy")
        if threshold is None:
            threshold = float(defaults.get("threshold", 0.3))
        if not metadata_output:
            metadata_output = defaults.get("metadata_output", "metadata.json")
        
        logger.info("Uruchomienie taggy w trybie tagowania obrazów za pomocą CLIP.")
        logger.debug(
            f"images_path={images_path}, labels={labels}, operation={operation}, threshold={threshold}, metadata_output={metadata_output}")
        
        
        image_files = [
            os.path.join(images_path, f) for f in os.listdir(images_path)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp'))
        ]
        metadata_list = []
            
        for file_path in image_files:
            results = tagger.tag_image(image_path=file_path, labels=labels, threshold=threshold)
            assigned_labels = [r["tag"] for r in results]
            logger.info(f"Przetwarzanie pliku: {file_path}")
            for label in assigned_labels:
                label_dir = os.path.join(images_path, label)
                create_directory(label_dir)
                perform_file_operation(file_path, label_dir, operation)
    
            if 'metadata' in operation:
                metadata_list.append({
                    "file":            file_path,
                    "assigned_labels": assigned_labels,
                    "full_results": 	 results
                    })
                    
        if  'metadata' in operation and metadata_list:
            output_file = os.path.join(images_path, metadata_output)
            save_metadata_to_json(metadata_list, output_file)


@cli.command("search")
@click.option('--query', '-q', type=str, required=True, help='Zapytanie tekstowe do wyszukiwania obrazów.')
@click.option('--top-k', '-k', type=int, default=5, help='Liczba najlepszych wyników do zwrócenia.')
@click.pass_context
def search_images(ctx, query, top_k):
    """Search for images similar to the query."""
    images_path = ctx.obj['images_path']
    tagger =ctx.obj['tagger']
    
    results = tagger.search_images(query, images_path, top_k=top_k)
    
    click.echo(f"Wyniki wyszukiwania dla zapytania: '{query}'")
    for i, (file, score) in enumerate(results):
        click.echo(f"{i + 1}. {file} (similarity: {score:.4f})")


@cli.command("duplicates")
@click.option('--similarity-threshold', '-t', default=0.9, type=float, help="Threshold for similarity (default: 0.9).")
@click.option('--operation', '-o', multiple=True, type=click.Choice(['copy', 'symlink',  'metadata']),
              help='Wybierz, czy pliki mają być kopiowane tworzone skróty.')
@click.pass_context
def find_duplicates(ctx, similarity_threshold=None, operation=None):
    """
    Find and group duplicate images.
    """
    images_path = ctx.obj['images_path']
    output_path = ctx.obj['output_path']
    defaults = ctx.obj['defaults']
    tagger = ctx.obj['tagger']
    if not operation:
        operation = defaults.get("operation", "copy")
        
    duplicates = tagger.find_duplicates(images_path, similarity_threshold)
    tagger.group_duplicates(duplicates, output_path, operation)
    click.echo(f"Duplicates grouped into folders at {output_path}.")


    
if __name__ == "__main__":
    cli()
