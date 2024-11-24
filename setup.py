from setuptools import setup, find_packages

setup(
    name='taggy',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'click',
        'rich',
        'toml'
    ],
    entry_points={
        'console_scripts': [
            'taggy=taggy.cli:add_tag',  # This links the CLI to the `taggy` command
        ]
    },
)
