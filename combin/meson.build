## Extension modules
py.extension_module(
  '_combinatorial',
  '_combinatorial.cpp',
  include_directories: [inc_pkg, inc_numpy, inc_pybind11],
  install: true, 
  cpp_args: _cpp_args
)

python_sources = [
  '__init__.py',
  'combinatorial.py'
]

## Print install directory
message('Source install dir = '+py.get_install_dir())
py.install_sources(
  python_sources,
  subdir: 'combin', 
  pure: false # Will be installed next to binaries
)