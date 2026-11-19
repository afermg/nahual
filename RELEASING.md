# Releasing Nahual

Releases are published by `.github/workflows/release.yaml` when a GitHub Release
is published. Package versions come from Git tags through `hatch-vcs`; do not
add a static version to `pyproject.toml` or edit the generated `_version.py`.

## One-time configuration

Register a GitHub Trusted Publisher in the existing `nahual` project's PyPI
Publishing settings with owner `afermg`, repository `nahual`, workflow filename
`release.yaml`, and environment `pypi`. Create the matching GitHub environment;
it can require reviewer approval before publication. No PyPI API token is needed.

## Release procedure

1. Ensure the intended release commit is on `master` and its tests pass.
2. Tag that commit with a new version, such as `v0.0.12`, and push the tag.
3. Create and publish a GitHub Release for that tag with release notes.
4. Check the Release workflow and verify the new version on PyPI.

The workflow tests the tagged commit on supported Python versions, builds a
source distribution and wheel, checks their metadata against the release tag,
and smoke-tests the installed wheel. A separate job downloads those artifacts
and publishes them only after tests and build checks succeed. Only that job has
OIDC token permissions, and it runs in the `pypi` environment.

The test matrix pins Ubuntu 24.04 because GitHub's Ubuntu 26.04 Python toolcache
does not provide Python 3.9. Check all supported Python versions before updating
the runner image.

For local builds, use a full Git checkout with tags and run `uv build`; the wheel
is rebuilt from the source distribution. Builds between tags have development
versions. Plain source snapshots without Git metadata are not release sources;
published source distributions include the metadata needed to rebuild wheels.

If a release fails, inspect its logs and the existing files on PyPI before
retrying. Never move a published release tag or reuse a version for changed
artifacts. Fix source problems in a new version; authentication failures may be
retried for the same tag after correcting the Trusted Publisher configuration.
