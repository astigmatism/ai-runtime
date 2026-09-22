# AI Runtime development and production boundary

Read `README.md` and `docs/deployment.md` before changing deployment behavior.

- Develop and test in `/Users/astigmatism/Projects/ai-runtime` on the Mac.
- Commit and push reviewed source to this repository's `main`, then deploy using AI Runtime's **Update and restart** action in Service Portal.
- `/home/astigmatism/deployments/ai-runtime` on Rosalina is a production deployment checkout. Do not edit or commit application source there, modify running container files, or manually advance the checkout ahead of the deployed revision.
- Preserve the updater's clean-checkout, fast-forward, revision, validation, and workload-draining safeguards. Follow documented recovery procedures for exceptional recovery; do not manufacture active-release receipts.
- Model/profile behavior, configuration rendering, validation, and UI belong in source. Hardware UUIDs, paths, credentials, selected release, and operational state belong in the existing Git-excluded deployment settings. Never commit `.state`, `.env`, secrets, model weights, or production conversations.
- Run `python3 -m unittest discover -s tests -v` before publication. The Portal updater also tests its immutable candidate before deployment.
