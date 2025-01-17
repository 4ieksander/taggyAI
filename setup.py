from setuptools import setup, find_packages

setup(
    name='taggy',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'click',
        'torch,'   # for torch.hub
        'torchvision,' # for torchvision.models
        'rich,'    # for logging
        'numpy,'   # for image loading
        'pillow,' # for image loading
        'git+https,://github.com/openai/CLIP.git' # for CLIP (main model for analyzing images)
        'opencv-python # for simple analysis quality of images (sharpness, detecting faces)'
    ],
    entry_points={
        'console_scripts': [
            'taggy=taggy.taggy_cli',  # This links the CLI to the `taggy` command
        ]
    },
)
