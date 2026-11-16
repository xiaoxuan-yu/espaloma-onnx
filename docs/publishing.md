# Release automation

GitHub: xiaoxuan-yu/espaloma-onnx. Hugging Face and ModelScope targets: youmans/espaloma-onnx. PyPI distribution: espaloma-onnx.

Pushes and pull requests run locked uv tests and package builds. Publishing a GitHub Release with a tag matching the package version (e.g. v0.1.0) runs the same checks and independently publishes PyPI distributions and model assets to both hubs. No publication happens on ordinary pushes. Failed platform jobs can be retried independently; tags must not be reused for changed assets.

Repository variables: HF_REPO_ID and MODELSCOPE_REPO_ID. Repository secrets: HF_TOKEN and MODELSCOPE_API_KEY with write access. The script never logs tokens. GitHub environments are pypi and model-hubs.

Configure a PyPI pending Trusted Publisher (https://pypi.org/manage/account/publishing/) using project espaloma-onnx, GitHub owner xiaoxuan-yu, repository espaloma-onnx, workflow release.yml, environment pypi. This account-side step requires the PyPI owner's session and cannot be replaced by a GitHub secret. No PyPI API token is used. Check package name ownership first. See https://docs.pypi.org/trusted-publishers/ .

Model publication uses a public repository and MIT license, then creates a version tag. Existing tags should be treated as immutable. Live hub upload is performed only when the GitHub Release event is explicitly published.

The daily upstream watcher checks the official latest release and requires a versioned checkpoint asset. Candidate conversion and reference tests run before opening an update PR. Current golden fixtures and runtime model-version checks intentionally reject unreviewed weight/architecture changes; such changes fail the workflow and require updating support and regenerating reviewed reference data. This is an automatic build-and-check trigger, not unconditional automatic publication of arbitrary upstream weights. Enable GitHub Actions permission to create PRs if using the watcher.

ONNX binaries are excluded from Git; CI exports from the pinned original checkpoint and passes staged model assets via Actions artifacts to the platform jobs. docs/model-card.md and root LICENSE/NOTICE are tracked sources for distribution.
