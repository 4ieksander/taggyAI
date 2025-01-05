import os
import shutil
import win32com.client
import json
import configparser

from .logger import get_logger


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
            "labels":          "people, documents, gadgets, cables, festivals, work, pets, random, nature, food, travel, architecture, art",
            "operation":       "copy",
            "threshold":       "0.3",
            "metadata_output": "metadata.json"
            }
    return config["DEFAULT"]


def create_directory(path):
    """Tworzy katalog, jeśli nie istnieje."""
    if not os.path.exists(path):
        os.makedirs(path)

def copy_file(src, dest):
    """Kopiuje plik ze źródła do celu."""
    shutil.copy2(src, dest)

def create_shortcut(target, shortcut_path, description=None, icon_path=None):
    """
    Tworzy skrót w systemie Windows.
    Używa bezwzględnych ścieżek do plików.
    """
    target = os.path.abspath(target)
    shortcut_path = os.path.abspath(shortcut_path)
    shell = win32com.client.Dispatch("WScript.Shell")
    shortcut = shell.CreateShortcut(shortcut_path)
    shortcut.TargetPath = target
    shortcut.WorkingDirectory = os.path.dirname(target)
    if description:
        shortcut.Description = description
    if icon_path and os.path.exists(icon_path):
        shortcut.IconLocation = icon_path
    shortcut.save()

def perform_file_operation(src, dest_dir, operation, description=None, icon_path=None):
    """
    Wykonuje operację na pliku (kopiowanie lub tworzenie skrótu).
    """
    dest = os.path.join(dest_dir, os.path.basename(src))
    if "copy" in operation:
        copy_file(src, dest)
    if "symlink" in operation:
        if os.name == "nt":
            create_shortcut(src, dest + ".lnk", description, icon_path)
        else:
            os.symlink(src, dest)

def save_metadata_to_json(metadata, output_file):
    """Zapisuje metadane do pliku JSON."""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)
    logger.info(f"Zapisano metadane do pliku: {output_file}")
