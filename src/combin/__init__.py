import importlib.metadata
from pathlib import Path
__version__ = importlib.metadata.version("combin")

from .combinatorial import * 

## Based on Numpy's usage: https://github.com/numpy/numpy/blob/v1.25.0/numpy/lib/utils.py#L75-L101
def get_include():
  """Return the directory that contains the combin's \\*.h header files.

  Extension modules that need to compile against combins should use this
  function to locate the appropriate include directory.

  Notes: 
    When using `distutils`, for example in `setup.py`:
      ```python
      import combin
      ...
      Extension('extension_name', ..., include_dirs=[combin.get_include()])
      ...
      ```
    Or with `meson-python`, for example in `meson.build`:
      ```meson
      ...
      run_command(py, ['-c', 'import combin; print(combin.get_include())', check : true).stdout().strip()
      ...
      ```
  """
  import os 
  # d = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'include')
  d = os.path.join(Path(__file__).parent, 'include')
  return d
