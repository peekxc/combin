## Extension modules
inc_pkg = include_directories('include')

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

include_sources = [
  'include' / 'combinatorial.h', 
  'include' / 'combin_ranges.h', 
]
include_utility_sources = [
  'include' / 'utility' / 'combinations.h',
  'include' / 'utility' / 'set_utilities.h',
  'include' / 'utility' / 'short_alloc.h'
]

## Print install directory
message('Source install dir = '+py.get_install_dir())
py.install_sources(
  python_sources + ['include/combinatorial.h'],
  subdir: 'combin', # keep as package name
  pure: false       # Will be installed next to binaries
)
py.install_sources(
  include_sources,
  subdir: 'combin' / 'include' / 'combin', # installs include files under combin namespace
  pure: false        # Will be installed next to binaries
)
py.install_sources(
  include_utility_sources,
  subdir: 'combin' / 'include' / 'combin' / 'utility', # installs include files under combin/utility namespace
  pure: false        # Will be installed next to binaries
)
# install_data(include_sources, install_dir: py.get_install_dir())
