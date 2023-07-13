# CHANGELOG

## UNRELEASED

### Changed

- DaskManager.chunk_execute(compute=False) returns a List of Delayeds instead of
  a Generator

### Added

- Worker.read() takes an optional masked parameter, when set to False, ignores 
  dataset mask.

### Fixed

- Fixed issue where Worker.lnglat_bounds() would fail inverse transforms for 
  chunks >10000 px

## v0.1.4 - 2021-11-29

### Changed

- Update to depend on rio_tiler v3 and morecantile v3

### Added

- Add parameter `compute=False` to have DaskManager return Delayed objects 
  instead of computing immediately.

## v0.1.3 - 2021-10-02

### Fixed

- Fix tutorial notebooks
- Limit number of retries on CPLE_AppDefinedError failures on COG reads
- Fix inverted masks being written

## v0.1.2 - 2021-08-18

### Fixed

- Updated documentation to link to the right places.

## v0.1.1 - 2021-08-18

Initial release
