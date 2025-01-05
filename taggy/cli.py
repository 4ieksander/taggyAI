""""
Konsolowa aplikacja CLI do tagowania obrazów przy użyciu modelu CLIP.
Umożliwia kopiowanie plików do katalogów odpowiadających etykietom
lub zapisywanie metadanych z przypisanymi etykietami w pliku JSON.
Posiada również opcję wyszukiwania obrazów na podstawie zapytania tekstowego.
"""
import configparser
import os
import click
import shutil
import json

# import winshell

from utils.logger import get_logger
from ai_tagger import ImageTagger

logger = get_logger(__name__)


def load_config(config_file: str = "config.ini"):
    """
    Ładuje domyślne wartości z pliku konfiguracyjnego (ini).
    Zwraca obiekt config (dict-like) z wartościami w sekcji DEFAULT.
    """
    config = configparser.ConfigParser()
    if os.path.exists(config_file):
        config.read(config_file, encoding='utf-8')
        logger.info(f"Wczytano plik konfiguracyjny: {config_file}")
    else:
        logger.warning(f"Plik konfiguracyjny '{config_file}' nie istnieje. Używam wartości domyślnych w kodzie.")
        # Możemy w razie potrzeby dodać tutaj jakieś wartości fallback.
        config["DEFAULT"] = {
            "images_path":     "./images",
            "labels":          "people,family,friends,pets,nature,travel,holiday,fun,food,selfie",
            "operation":       "copy",
            "threshold":       "0.3",
            "metadata_output": "metadata.json"
            }
    return config["DEFAULT"]

    
@click.group()
def cli():
    """Tagowanie i wyszukiwanie obrazów za pomocą CLIP."""
    pass

@cli.command("tag")
@click.option('--images-path', '-i', type=click.Path(exists=True),
              help='Ścieżka do folderu z obrazami.')
@click.option('--labels', '-l', multiple=True,
              help='Lista etykiet (można podać wiele razy, np. -l cat -l dog -l car).')
@click.option('--operation', '-o', multiple=True, type=click.Choice(['copy', 'symlink',  'metadata', 'edit-metadata']),
              help='Wybierz, czy pliki mają być kopiowane, zapisywane metadane, czy edytowane metadane plików.')

@click.option('--threshold', '-t', type=float,
              help='Minimalny próg prawdopodobieństwa dla przypisania etykiety.')
@click.option('--metadata-output', '-m',
              help='Plik, w którym zapisywane będą metadane (jeśli operation=metadata).')
@click.option('--config-file', default='config.ini',
              help='Plik konfiguracyjny z domyślnymi wartościami.')
def taggy(images_path, labels, operation, threshold, metadata_output, config_file):
    """
    Taguje obrazy za pomocą modelu CLIP.
    1. Wczytuje konfigurację z pliku .ini (jeśli istnieje).
    2. Parametry z CLI nadpisują wartości konfiguracyjne.
    3. Dla każdego obrazu wyznacza etykiety z modelu CLIP.
    4. Kopiuje pliki do folderów lub zapisuje metadane, w zależności od wybranej opcji.
    """
    defaults = load_config(config_file)
    
    if not images_path:
        images_path = defaults.get("images_path", "./images")
    if not labels:
        labels_from_config = defaults.get("labels", "people,family,friends").split(',')
        labels = [label.strip() for label in labels_from_config]
    if not operation:
        operation = defaults.get("operation", "copy")
    if threshold is None:
        threshold = float(defaults.get("threshold", 0.3))
    if not metadata_output:
        metadata_output = defaults.get("metadata_output", "metadata.json")
    
    logger.info("Uruchomienie CLI do tagowania obrazów za pomocą CLIP.")
    logger.info(
        f"images_path={images_path}, labels={labels}, operation={operation}, threshold={threshold}, metadata_output={metadata_output}")
    
    tagger = ImageTagger(model_name="CLIP")
    
    files_in_dir = [f for f in os.listdir(images_path) if os.path.isfile(os.path.join(images_path, f))]
    
    metadata_list = []
    
    for file_name in files_in_dir:
        full_path = os.path.join(images_path, file_name)
        
        if not file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp')):
            continue
        
        logger.info(f"Przetwarzanie pliku: {file_name}")
        
        results = tagger.tag_image(image_path=full_path, labels=list(labels))
        
        assigned_labels = [r["tag"] for r in results if r["probability"] >= threshold]
        
        if 'copy' in operation or 'symlink' in operation:
            for label in assigned_labels:
                label_dir = os.path.join(images_path, label)
                if not os.path.exists(label_dir):
                    os.makedirs(label_dir)
                
                dest_path = os.path.join(label_dir, file_name)
                if  'copy' in operation:
                    shutil.copy2(full_path, dest_path)
                    logger.info(f"Skopiowano {file_name} do {label_dir}")
                elif  'symlink' in operation:
                    if os.name == 'posix':
                        print("To jest system Linux (lub inny system oparty na POSIX).")
                        if not os.path.exists(dest_path):
                            os.symlink(full_path, dest_path)
                            logger.info(f"Utworzono skrót do {file_name} w {label_dir}")
                        else:
                            logger.warning(f"Skrót do pliku {file_name} już istnieje w {label_dir}")
                    elif os.name == 'nt':
                        print("To jest system Windows.")
                        shortcut_path = os.path.join(label_dir, f"{file_name}.lnk")
                        
                        if not os.path.exists(shortcut_path):  # Unikaj powielania skrótów
                            # winshell.CreateShortcut(
                            # 	Path=shortcut_path,
                            # 	Target=full_path,
                            # 	Icon=(full_path, 0)
                            # 	)
                            logger.info(f"Utworzono skrót do {file_name} w {label_dir}")
                        else:
                            logger.warning(f"Skrót do pliku {file_name} już istnieje w {label_dir}")
                    else:
                        print("Nieznany system operacyjny.")


        if 'metadata' in operation:
            metadata_list.append({
                "file":            file_name,
                "assigned_labels": assigned_labels,
                "full_results": 	 results
                })
    
    if 'edit-metadata' in operation:
        for file_name in files_in_dir:
            full_path = os.path.join(images_path, file_name)
            
            if not file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            
            logger.info(f"Przetwarzanie pliku: {file_name}")
            
            # Przypisz tagi z modelu CLIP
            results = tagger.tag_image(image_path=full_path, labels=list(labels))
            assigned_labels = [r["tag"] for r in results if r["probability"] >= threshold]
            
            if assigned_labels:
                success, message = tagger.add_tags_to_metadata(full_path, assigned_labels)
                if success:
                    logger.info(message)
                else:
                    logger.error(message)
                    
    if  'metadata' in operation and metadata_list:
        output_file = os.path.join(images_path, metadata_output)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(metadata_list, f, indent=4, ensure_ascii=False)
        logger.info(f"Zapisano metadane do pliku: {output_file}")


@cli.command("search")
@click.option('--images-path', '-i', type=click.Path(exists=True), required=True,
              help='Ścieżka do katalogu z obrazami.')
@click.option('--query', '-q', type=str, required=True,
              help='Zapytanie tekstowe do wyszukiwania obrazów.')
@click.option('--top-k', '-k', type=int, default=5,
              help='Liczba najlepszych wyników do zwrócenia.')
def search_images(images_path, query, top_k):
    """Wyszukuje obrazy pasujące do zapytania tekstowego."""
    tagger = ImageTagger(model_name="CLIP")
    results = tagger.search_images(query, images_path, top_k=top_k)
    
    click.echo(f"Wyniki wyszukiwania dla zapytania: '{query}'")
    for i, (file, score) in enumerate(results):
        click.echo(f"{i + 1}. {file} (similarity: {score:.4f})")


@cli.command("find-duplicates")
@click.option('--images-path', '-i', required=True, type=click.Path(exists=True),
              help="Path to the folder with images.")
@click.option('--output-path', '-o', required=True, type=click.Path(),
              help="Path to the folder to save grouped duplicates.")
@click.option('--similarity-threshold', '-t', default=0.9, type=float, help="Threshold for similarity (default: 0.9).")
def find_duplicates(images_path, output_path, similarity_threshold):
    """
    Find and group duplicate images.
    """
    tagger = ImageTagger()
    duplicates = tagger.find_duplicates(images_path, similarity_threshold)
    tagger.group_duplicates(duplicates, output_path)
    click.echo(f"Duplicates grouped into folders at {output_path}.")
    
        
if __name__ == "__main__":
    cli()