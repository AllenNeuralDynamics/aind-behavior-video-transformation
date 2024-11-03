# aind-behavior-video-transformation

[![License](https://img.shields.io/badge/license-MIT-brightgreen)](LICENSE)
![Code Style](https://img.shields.io/badge/code%20style-black-black)
[![semantic-release: angular](https://img.shields.io/badge/semantic--release-angular-e10079?logo=semantic-release)](https://github.com/semantic-release/semantic-release)
![Interrogate](https://img.shields.io/badge/interrogate-100.0%25-brightgreen)
![Coverage](https://img.shields.io/badge/coverage-97%25-brightgreen?logo=codecov)
![Python](https://img.shields.io/badge/python->=3.10-blue?logo=python)

This project is used to encode behavior videos and, depending on the user
request, will compress those videos while ensuring that they are formatted for
correct display across many devices. This may include performing common image
pre-processing steps that are common across many commercial video products, but
have to be done post-hoc for our behavior videos.

## Goals

This will attempt to compress videos so that the results:

* Retain the majority of the detail of the input video
* Take up as little space as possible for a target amount of visual detail
* Are in a format that can be widely viewed across devices, and streamed by
  browsers
* Have pixel data in a color space that most players can properly display

This video compression is often lossy, and the original videos are not kept, so
this library will attempt to perform the highest-quality video compression once.
The _speed_ of this compression is strictly secondary to the _quality_ of the
compression, as measured by the visual detail retained and the compression
ratio. See
[this section](#brief-benchmarks-on-video-compression-with-cpu-based-encoders-and-gpu-based-encoders)
for more details.


Additionally, this package should provide an easy to use interface that:

* Presents users with a curated set of compression settings, which have been
  rigorously tested
* Allow users to also provide their own compression settings, if they have
  specific requirements

## Non-goals

* Spending as little time and compute resources as possible to transform videos,
  at the expense of compression quality.

## Usage
 - The BehaviorVideoJob.run_job method in the transform_videos should be the primary method to call for processing video files.
 - On a merge to main, this package will be published as a singularity container, which can easily be run on a SLURM cluster.

## Brief benchmarks on video compression with CPU-based encoders and GPU-based encoders

A perhaps surprising fact is that video encoders implementing the same algorithm
written for different compute resources
_do not have the same visual performance_, in that for a given compression ratio
they do not retain the same amount of visual detail.

This can be seen in the plot below, where the GPU encoder and CPU encoders
retain different amounts of visual detail, as assessed with VMAF.

![visual performance vs compress ratio](/assets/compression-vs-quality.png)

This also shows that different presets of the same encoder, like _CPU Slow_ and
_CPU Fast_, also have different performance. For comrpession ratios greater than
100, it often makes sense to take your time and use a slow preset of a CPU-based
encoder to retain as much visual information for a given amount of compression.

While it may be tempting to select a faster preset, or faster compute resource
like GPU for dramatic speedups shown below, the lossy compression and intent to
delete the original means that taking time to do it right once might well be
worth it.

![throughput vs compress ratio](/assets/compression-vs-speed.png)

## Development

To develop the code, run
```bash
pip install -e .[dev]
```

## Contributing

### Linters and testing

There are several libraries used to run linters, check documentation, and run tests.

- Please test your changes using the **coverage** library, which will run the tests and log a coverage report:

```bash
coverage run -m unittest discover && coverage report
```

- Use **interrogate** to check that modules, methods, etc. have been documented thoroughly:

```bash
interrogate .
```

- Use **flake8** to check that code is up to standards (no unused imports, etc.):
```bash
flake8 .
```

- Use **black** to automatically format the code into PEP standards:
```bash
black .
```

- Use **isort** to automatically sort import statements:
```bash
isort .
```

### Pull requests

For internal members, please create a branch. For external members, please fork the repository and open a pull request from the fork. We'll primarily use [Angular](https://github.com/angular/angular/blob/main/CONTRIBUTING.md#commit) style for commit messages. Roughly, they should follow the pattern:
```text
<type>(<scope>): <short summary>
```

where scope (optional) describes the packages affected by the code changes and type (mandatory) is one of:

- **build**: Changes that affect build tools or external dependencies (example scopes: pyproject.toml, setup.py)
- **ci**: Changes to our CI configuration files and scripts (examples: .github/workflows/ci.yml)
- **docs**: Documentation only changes
- **feat**: A new feature
- **fix**: A bugfix
- **perf**: A code change that improves performance
- **refactor**: A code change that neither fixes a bug nor adds a feature
- **test**: Adding missing tests or correcting existing tests

### Semantic Release

The table below, from [semantic release](https://github.com/semantic-release/semantic-release), shows which commit message gets you which release type when `semantic-release` runs (using the default configuration):

| Commit message                                                                                                                                                                                   | Release type                                                                                                    |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------- |
| `fix(pencil): stop graphite breaking when too much pressure applied`                                                                                                                             | ~~Patch~~ Fix Release, Default release                                                                          |
| `feat(pencil): add 'graphiteWidth' option`                                                                                                                                                       | ~~Minor~~ Feature Release                                                                                       |
| `perf(pencil): remove graphiteWidth option`<br><br>`BREAKING CHANGE: The graphiteWidth option has been removed.`<br>`The default graphite width of 10mm is always used for performance reasons.` | ~~Major~~ Breaking Release <br /> (Note that the `BREAKING CHANGE: ` token must be in the footer of the commit) |

### Documentation
To generate the rst files source files for documentation, run
```bash
sphinx-apidoc -o doc_template/source/ src
```
Then to create the documentation HTML files, run
```bash
sphinx-build -b html doc_template/source/ doc_template/build/html
```
More info on sphinx installation can be found [here](https://www.sphinx-doc.org/en/master/usage/installation.html).
