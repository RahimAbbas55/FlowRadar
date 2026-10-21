# Deploying the dashboard to Streamlit Community Cloud

The deployed app is a static demo: it reads `snapshot/snapshot.json`, which holds results computed once by the real pipeline. There is no live model, API or retrain loop behind it.

## 1. Build the snapshot

    python -m flowradar.dashboard.build_snapshot

Add `--regenerate` to rebuild `data/default` first. Commit `snapshot/snapshot.json`.

## 2. Check it locally

    streamlit run deploy/streamlit_app.py

## 3. Deploy

1. Push the repository to GitHub.
2. In Streamlit Community Cloud choose to create an app from that repository and branch.
3. Set the main file path to `deploy/streamlit_app.py`.
4. In the advanced settings (where secrets and the Python version are set) choose the same Python version you develop with.
5. Deploy. Community Cloud installs `deploy/requirements.txt`, because a dependency file next to the entrypoint takes precedence over the one in the repository root.

## Updating

Rebuild the snapshot, commit it and push. The app updates from the branch.

Apps hibernate after a period of inactivity, so the first visit after a quiet spell may show a wake-up screen.