from setuptools import setup, find_packages

setup(
    name='taggy',
    version='0.2 ',
    packages=find_packages(),
    install_requires=[
        'click',
        ],
    dependency_links=[
        'git+https://github.com/openai/CLIP.git#egg=clip'
        ],
    entry_points={
        'console_scripts': [
            'taggy=taggy.taggy_cli:cli',
        ]
    },
)
