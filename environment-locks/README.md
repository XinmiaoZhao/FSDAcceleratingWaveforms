# Environment identity

`linux-64.conda.lock` and `osx-arm64.conda.lock` pin every Conda package URL
and SHA256 for the two CI platforms. They include PyCBC 2.9.0 and the concrete
LAL component builds needed by the release tests. CI verifies the installed
package set against the selected lock before reproduction and uploads its
runtime export with the full record.

The human-maintained `environment.yml` remains the cross-platform source
constraint. Lock files are platform-specific and are never treated as
interchangeable across operating systems.
