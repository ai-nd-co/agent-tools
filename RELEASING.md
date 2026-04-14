# Releasing `ai-nd-co-agent-tools`

This package is published to PyPI as `ai-nd-co-agent-tools` and installed as:

```bash
pip install ai-nd-co-agent-tools
```

The CLI command remains:

```bash
agent-tools
```

## Before first release

1. Confirm the GitHub repo is public: `ai-nd-co/agent-tools`
2. Confirm the release workflow exists at `.github/workflows/release.yml`
3. On PyPI, create or claim the project `ai-nd-co-agent-tools`
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

## Automatic semantic-release flow

This repo now uses semantic-release on pushes to `main`.

Normal flow:

1. merge a Conventional Commit to `main`
2. semantic-release computes the next version
3. it updates version files + changelog
4. it commits the release bump back to `main`
5. it creates a `py-vX.Y.Z` tag
6. the tag-triggered release workflow publishes to PyPI

Versioning rules:

- `feat:` => minor
- `fix:` => patch
- `feat!:` / `BREAKING CHANGE:` => major

Commits like `chore:` or `docs:` do not cut a release.

## Bootstrap

The automated Python release flow uses `py-v*` tags.

Older `v0.1.x` tags are intentionally ignored.

One-time bootstrap tag:

```bash
git tag -a py-v0.1.1 -m "Bootstrap semantic-release at 0.1.1"
git push origin py-v0.1.1
```

If the bootstrap version already exists on PyPI, the publish workflow skips upload cleanly.

## Verification

After publish:

```bash
pip install ai-nd-co-agent-tools
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
- semantic-release owns version bumps and `py-v*` tags; manual workflow runs build only unless explicitly told to publish
