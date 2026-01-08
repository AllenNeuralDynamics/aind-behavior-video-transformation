# aind-behavior-video-transformation

[![License](https://img.shields.io/badge/license-MIT-brightgreen)](LICENSE)
![Code Style](https://img.shields.io/badge/code%20style-black-black)
[![semantic-release: angular](https://img.shields.io/badge/semantic--release-angular-e10079?logo=semantic-release)](https://github.com/semantic-release/semantic-release)
![Interrogate](https://img.shields.io/badge/interrogate-100.0%25-brightgreen)
![Coverage](https://img.shields.io/badge/coverage-92%25-brightgreen?logo=codecov)
![Python](https://img.shields.io/badge/python->=3.10-blue?logo=python)

This project is used to encode behavior videos and, depending on the user
request, will compress those videos while ensuring that they are formatted for
correct display across many devices. This may include common image
preprocessing steps, such as gamma encoding, that are necessary for correct
display, but have to be done post-hoc for our behavior videos.

## Goals

This will attempt to compress videos so that the results:

* Retain the majority of the detail of the input video
* Take up as little space as possible for a target amount of visual detail
* Are in a format that can be widely viewed across devices, and streamed by
  browsers
* Have pixel data in a color space that most players can properly display

This video compression is often lossy, and the original videos are not kept, so
this library will attempt to produce the highest-quality video for a target
compression ratio. The _speed_ of this compression is strictly secondary to the
_quality_ of the compression, as measured by the visual detail retained and the
compression ratio. See
[this section](#brief-benchmarks-on-video-compression-with-cpu-based-encoders-and-gpu-based-encoders)
for more details.


Additionally, this package should provide an easy to use interface that:

* Presents users with a curated set of compression settings, which have been
  rigorously tested in terms of their visual quality using perception-based
  metrics like VMAF.
* Allow users to also provide their own compression settings, if they have
  specific requirements

## Non-goals

* Sacrifice the visual fidelity of videos in order to decrease encoding time.

## Brief benchmarks on video compression with CPU-based encoders and GPU-based encoders

A surprising fact is that video encoders implementing the same algorithm, but
written for different compute resources do _not_ have the same visual
performance; for a given compression ratio, or similar settings, they do not
retain the same amount of visual detail. This is also true for different presets
of the same encoder and compute resource even if the other settings are
identical. For example, the presets `-preset fast` and `-preset veryslow` of the
encoder `libx264` produce videos with the same compression ratio, but differing
visual quality.

This can be seen in the plot below, where the GPU encoder and CPU encoders
retain different amounts of visual detail, as assessed with visual
perception-based metric
[VMAF](https://en.wikipedia.org/wiki/Video_Multimethod_Assessment_Fusion). Also
note the difference between presets for the same encoder and compute resource:
_CPU Fast_ and _CPU Slow_.

![visual performance vs compress ratio](/assets/compression-vs-quality.png)

This figure shows that for compression ratios greater than 100, it often makes
sense to take your time and use a slow preset of a CPU-based encoder to retain
as much visual information for a given amount of compression.

While it may be tempting to select a faster preset, or faster compute resource
like GPU for dramatic speedups shown below, doing will degrade the quality of
the resulting video.

![throughput vs compress ratio](/assets/compression-vs-speed.png)

Because the output of this package are permanent video artifacts, the
compression is lossy, and the intent is to delete the original, taking the CPU
time to produce the highest quality video possible might well be worth it.

## Usage
 - The BehaviorVideoJob.run_job method in the transform_videos should be the
   primary method to call for processing video files.
 - On a merge to main, this package will be published as a docker
   image, which can easily be run on a SLURM cluster.

## Docker build for local testing
In the same directory as the Dockerfile, run
```bash
docker build -t aind-behavior-video-transformation-local .
```

Now a docker container can be run. You may need to change the mount locations and file permissions:
```bash
docker run -v /home/local_videos/input_source:/mnt/input_source -v /home/local_videos/output_directory:/mnt/output_directory aind-behavior-video-transformation-local python -m aind_behavior_video_transformation.etl --job-settings '{"compression_requested": {"compression_enum": "gamma fix colorspace"}, "parallel_compression": true, "input_source": "/mnt/input_source", "output_directory": "/mnt/output_directory"}'
```

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
