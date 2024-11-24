import toml

def read_config(file_path='config.toml'):
    """Read and parse the configuration file."""
    try:
        config = toml.load(file_path)
        return config
    except Exception as e:
        print(f"Error reading config: {e}")
        return {}

def get_default_tag(config):
    """Get the default tag from the configuration."""
    return config.get('settings', {}).get('default_tag', 'general')
