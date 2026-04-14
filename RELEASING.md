# Releasing `ai-agent-tools`

This package is published to PyPI as `ai-agent-tools` and installed as:

```bash
pip install ai-agent-tools
```

The CLI command remains:

```bash
agent-tools
```

## Before first release

1. Confirm the GitHub repo is public: `ai-nd-co/agent-tools`
2. Confirm the release workflow exists at `.github/workflows/release.yml`
3. On PyPI, create or claim the project `ai-agent-tools`
4. In PyPI project settings, add a Trusted Publisher with:
   - owner: `ai-nd-co`
   - repository: `agent-tools`
   - workflow: `.github/workflows/release.yml`
   - environment: `pypi`
5. In GitHub repo settings, keep the `pypi` environment and add manual approval if desired

PyPI docs:
- https://docs.pypi.org/trusted-publishers/
- https://docs.pypi.org/trusted-publishers/adding-a-publisher/

## Safe dry run

Manual workflow dispatch is safe by default and only builds artifacts.

From GitHub Actions:
- run `Release`
- leave `publish_to_pypi` unchecked

This validates the build without trying to publish.

## Real release

1. Make sure `main` is clean and pushed
2. Update version in `pyproject.toml`
3. Commit the version bump
4. Create and push a tag:

```bash
git tag -a v0.1.0 -m "Release v0.1.0"
git push origin main
git push origin v0.1.0
```

Tag pushes matching `v*` trigger the release workflow and publish to PyPI.

## Verification

After publish:

```bash
pip install ai-agent-tools
agent-tools --help
```

Also smoke-test the two commands:

```bash
echo "hello" | agent-tools transform --system-prompt-file prompt_examples/rewrite_for_tts.md

echo "hello" | agent-tools tts --output-file hello.wav
```

## Notes

- `transform` depends on local Codex ChatGPT login and `~/.codex/auth.json`
- the backend path is intentionally private/experimental and may break with Codex changes
- tag pushes publish; manual workflow runs build only unless explicitly told to publish
