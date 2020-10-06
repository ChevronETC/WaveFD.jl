# WaveFD.jl
| **Documentation** | **Action Statuses** |
|:---:|:---:|
| [![][docs-dev-img]][docs-dev-url] [![][docs-stable-img]][docs-stable-url] | [![][doc-build-status-img]][doc-build-status-url] [![][build-status-img]][build-status-url] [![][code-coverage-img]][code-coverage-results] |

WaveFD.jl contains a reference collection of single time step wavefield propagation codes. 

## Note on unit tests and coverage
These single time step propagators are used to contruct nonlinear modeling operators and their linearizations in the `JetPackWaveFD` package. An extensive suite of unit tests guaranteeing correctness is implemented in `JetPackWaveFD`, and therefore is not re-implemented here. 

Coverage statistics for tests in this package do not reflect the code as exercised by the unit tests in `JetPackWaveFD`.

## Additional Notes
In addition to propagators, the Julia components of WaveFD.jl are meant to provide:
* imaging conditions
* wavelets
* wavefield injection/extraction
* wavefield compression (via CvxCompress)
* absorbing boundary construction
* extraction of a subset of an earth model for a given experiment aperture (ginsu)

[docs-dev-img]: https://img.shields.io/badge/docs-dev-blue.svg
[docs-dev-url]: https://chevronetc.github.io/WaveFD.jl/dev/

[docs-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg
[docs-stable-url]: https://ChevronETC.github.io/WaveFD.jl/stable

[doc-build-status-img]: https://github.com/ChevronETC/WaveFD.jl/workflows/Documentation/badge.svg
[doc-build-status-url]: https://github.com/ChevronETC/WaveFD.jl/actions?query=workflow%3ADocumentation

[build-status-img]: https://github.com/ChevronETC/WaveFD.jl/workflows/Tests/badge.svg
[build-status-url]: https://github.com/ChevronETC/WaveFD.jl/actions?query=workflow%3A"Tests"

[code-coverage-img]: https://codecov.io/gh/ChevronETC/WaveFD.jl/branch/master/graph/badge.svg
[code-coverage-results]: https://codecov.io/gh/ChevronETC/WaveFD.jl
