# AI Runtime development and production boundary

Read `README.md` and `docs/deployment.md` before changing deployment behavior.

- Develop and test in `/Users/astigmatism/Projects/ai-runtime` on the Mac.
- Commit and push reviewed source to this repository's `main`, then deploy using AI Runtime's **Update and restart** action in Service Portal.
- `/home/astigmatism/deployments/ai-runtime` on Rosalina is a production deployment checkout. Do not edit or commit application source there, modify running container files, or manually advance the checkout ahead of the deployed revision.
- Preserve the updater's clean-checkout, fast-forward, revision, validation, and workload-draining safeguards. Follow documented recovery procedures for exceptional recovery; do not manufacture active-release receipts.
- Every destructive transition must be survivable without human intervention, and the recovery path must be exercised by tests against the exact failures the deploy paths can produce. Never add a path that can leave the router drained with nothing serving, or a failed recovery resting instead of scheduling the next attempt: this house is run by agents, and a human with no lever is the failure mode. Changes to drain, recovery, or restoration go with `tests/test_restoration.py`.
- Model/profile behavior, configuration rendering, validation, and UI belong in source. Hardware UUIDs, paths, credentials, selected release, and operational state belong in the existing Git-excluded deployment settings. Never commit `.state`, `.env`, secrets, model weights, or production conversations.
- Run `python3 -m unittest discover -s tests -v` before publication. The Portal updater also tests its immutable candidate before deployment.
