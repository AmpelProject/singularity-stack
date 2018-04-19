
from distutils.core import setup

setup(
    name='singularity-stack',
    version='0.2',
    py_modules=['singularity_stack'],
    entry_points = {
        'console_scripts' : ['singularity-stack = singularity_stack:main']
    }
)
