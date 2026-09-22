# AI Runtime repository rename

The application is **AI Runtime** and its repository is now **`astigmatism/ai-runtime`**, formerly `astigmatism/local-ai-runtime`.

```sh
git clone https://github.com/astigmatism/ai-runtime.git
cd ai-runtime
```

AI Runtime manages model launch configuration, lifecycle, and catalog publication. [LLM Router](https://github.com/astigmatism/llm-router) owns the OpenAI-compatible and Ollama-compatible inference endpoints. Runtime's HTTP interface is read-only: `/api/status`, `/healthz`, and its status UI on port 11436. Administrative changes use the container CLI and Service Portal updater.

## Compatibility for existing deployments

This release updates repository URLs, the image source label, and new ownership metadata. It accepts checkout origins using either repository name, over the existing HTTPS and SSH forms, while fetching future updates from the canonical public HTTPS URL. Unrelated repositories, dirty checkouts, unexpected branches/upstreams, and non-fast-forward history remain rejected.

The production container and Compose project remain `local-ai-runtime`. Existing home wrappers, router release tooling, recovery checks, state paths, catalog source fields, and maintenance ownership markers depend on those identifiers. The image namespace `local/ai-runtime` remains valid as well. These compatibility identifiers do not identify an obsolete application; they refer to the current controller. Renaming them requires a coordinated deployment migration and is separate from the repository rename.

Do not move the production checkout or `.state`, rename its container, or change its Compose project as part of adopting the repository name. The rename does not alter model definitions, GPU assignments, inference containers, or selection. Preserve older images and recovery kits under their recorded names.

### First update after the rename

The updater in releases before this rename accepts only the old Git remote. Keep the production checkout's `origin` at `https://github.com/astigmatism/local-ai-runtime.git` for the first normal Service Portal update; GitHub redirects that URL to the renamed repository. Do not manually pull the production checkout ahead of its deployed release, since the updater checks that they match.

Once the rename-aware release is deployed successfully, the deployment origin may be updated:

```sh
git remote set-url origin https://github.com/astigmatism/ai-runtime.git
```

Leaving the old origin in place also works with the new updater. Development checkouts can use the new URL immediately; preserve unfinished work before pulling. Fresh clones use the new repository name and directory.

For a replacement-disk restore, use the remote accepted by the exact source revision being restored. Pre-rename releases require the old URL, even though GitHub redirects it. Recover the recorded source/image/state first, then use the normal updater before switching the remote. See [the rebuild runbook](rebuild.md).

## Deployment identity verified on 2026-09-16

On `192.168.1.4`, the healthy `local-ai-runtime` container used image `local/ai-runtime:git-ddd2c84d257c6834ae7795440bbcb6ff0660986f`. The OCI title was already `AI Runtime`; its source label pointed to the former repository URL. All 18 runtime/config files present in the image matched that Git revision by SHA-256. The host checkout was clean at the same revision on `main`.

This repository rename was prepared without changing or restarting the production deployment.
