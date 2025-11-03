# MLflow Server: how it was created

This document explains how the MLflow server used for this project was created, the exact command that was run, what each flag does, required prerequisites, verification steps, and common troubleshooting (including the `Read-only file system: '/root'` artifact error).

## Command that was used

```bash
nohup ~/mlflow-venv/bin/mlflow server \
  --backend-store-uri sqlite:///$HOME/mlflow_data/mlflow.db \
  --default-artifact-root file://$HOME/mlflow_data/artifacts \
  --host 0.0.0.0 --port 5050 \
  --allowed-hosts '*' > mlflow.log 2>&1 &
```

This command was run from the server where you want the MLflow UI and API to be reachable.

---

## Quick summary / intent
- Start an MLflow server process in the background using the `mlflow` binary from a virtualenv at `~/mlflow-venv`.
- Use a local SQLite database as the backend store (`mlflow.db` under `$HOME/mlflow_data`).
- Store model artifacts on the server filesystem under `$HOME/mlflow_data/artifacts`.
- Listen on all interfaces (`0.0.0.0`) and port `5050`.
- Allow connections from any host (`--allowed-hosts '*'`).
- Redirect stdout/stderr to `mlflow.log` and run the process in the background (`nohup ... &`).

---

## Breakdown of the important parts

- `~/mlflow-venv/bin/mlflow server`
  - Runs the `mlflow server` command from the `mlflow` executable installed inside the `~/mlflow-venv` virtual environment.
  - Using a virtualenv ensures you run the version of MLflow and Python packages you installed for this service.

- `--backend-store-uri sqlite:///$HOME/mlflow_data/mlflow.db`
  - Sets the metadata backend where MLflow stores experiments, runs, and metadata.
  - The SQLite URI `sqlite:///<absolute-path-to-db>` points to a file-based DB. For `$HOME` expansion to work reliably in a shell, you should either expand `$HOME` before passing the URI (recommended) or use an absolute path.
  - Example expanded value might be: `sqlite:////home/youruser/mlflow_data/mlflow.db` (note the 4 slashes when absolute path starts with `/`).
  - NOTE: SQLite is OK for single-user or low-concurrency testing, but for production or concurrent clients you should use Postgres or MySQL.

- `--default-artifact-root file://$HOME/mlflow_data/artifacts`
  - Controls where MLflow stores artifacts (logged models, files, etc.).
  - `file://` indicates a filesystem-based artifact store on the machine running the server. The path should be writable by the MLflow server process.
  - If you want artifact storage on S3/GCS/Azure, you would supply an S3/GCS URI (e.g. `s3://my-bucket/mlflow-artifacts`) and ensure the server has credentials.

- `--host 0.0.0.0` and `--port 5050`
  - Make the MLflow HTTP UI/API accessible from other machines on port 5050.

- `--allowed-hosts '*'`
  - Tells MLflow to accept requests from any host header. This is permissive; restricting it or sitting MLflow behind a reverse proxy is recommended for production.

- `nohup ... > mlflow.log 2>&1 &`
  - Runs the process in the background and redirects logs to `mlflow.log`. `nohup` allows it to survive logout.
  - The background process ID is printed to the shell; you can also find it with `ps`.

---

## Recommended pre-setup commands (run before starting the server)

Create the directories and ensure they are writable by the user that will run the MLflow server:

```bash
# create directories
mkdir -p "$HOME/mlflow_data/artifacts"
mkdir -p "$HOME/mlflow_data"

# ensure permissions (replace username if running as a different user)
chown -R $(whoami) "$HOME/mlflow_data"
chmod -R u+rwX "$HOME/mlflow_data"

# create the sqlite file parent dir and (optionally) the DB file
touch "$HOME/mlflow_data/mlflow.db"
```

If you run MLflow inside a container or as a different system user, ensure the container or service has the same directories mounted and writable.

---

## Verifying the server is running

- Check the process and listening port:

```bash
ps aux | grep mlflow
ss -ltnp | grep 5050   # or: lsof -i :5050
```

- Check logs (the `nohup` redirection target):

```bash
tail -f mlflow.log
```

- Open the UI in your browser:

```
http://<server-ip>:5050
```

- From a client machine, verify the server responds using curl:

```bash
curl -v http://<server-ip>:5050
```

---

## Why you might see "status - failed" for a logged model in the MLflow UI

If MLflow cannot write model artifacts to the artifact store, logged models can end up in a partial/failed state in the UI. Typical causes:

- The artifact root directory is not writable by the MLflow server process (permission error). This manifests as errors like:

```
OSError: [Errno 30] Read-only file system: '/root'
```

- The server's `--default-artifact-root` is misconfigured (points to `/root` or a path that does not exist on the server), or the server is running inside a container where the artifact path is unavailable or read-only.

- The client is pointing at a different MLflow server than the one you inspected locally (check `mlflow.get_tracking_uri()` on the client side).

How to fix those issues:

- Ensure the `--default-artifact-root` path is correct and exists on the MLflow server host, and that the MLflow process user has write permission.

```bash
# example
sudo mkdir -p /var/lib/mlflow/artifacts
sudo chown mlflow_user:mlflow_user /var/lib/mlflow/artifacts
# then start server with
--default-artifact-root file:///var/lib/mlflow/artifacts
```

- For multi-machine usage, prefer a remote object store (S3/GCS/Azure) for the artifact root:

```bash
# example with S3 (server needs AWS creds/environment configured)
--default-artifact-root s3://my-mlflow-bucket/artifacts
```

- If you do not control the server, ask the server admin to update the default artifact root or ensure the server process has write access.

---

## Example: start the server with S3 artifact store

```bash
# make sure server environment has AWS credentials configured
nohup ~/mlflow-venv/bin/mlflow server \
  --backend-store-uri postgresql://user:pass@db-host:5432/mlflow \
  --default-artifact-root s3://my-bucket/mlflow-artifacts \
  --host 0.0.0.0 --port 5050 > mlflow.log 2>&1 &
```

Using S3 (or other object storage) removes the risk of the server trying to write to `/root` on the local filesystem.

---

## Systemd unit example (recommended over nohup for a long-lived server)

Create `/etc/systemd/system/mlflow.service` (adjust paths and user):

```ini
[Unit]
Description=MLflow server
After=network.target

[Service]
User=youruser
Group=yourgroup
WorkingDirectory=/home/youruser
Environment="PATH=/home/youruser/mlflow-venv/bin:/usr/bin"
ExecStart=/home/youruser/mlflow-venv/bin/mlflow server \
  --backend-store-uri sqlite:////home/youruser/mlflow_data/mlflow.db \
  --default-artifact-root file:////home/youruser/mlflow_data/artifacts \
  --host 0.0.0.0 --port 5050 --allowed-hosts '*'
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

Then enable & start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable mlflow
sudo systemctl start mlflow
sudo journalctl -u mlflow -f
```

---

## Docker Compose example

If you'd prefer to run MLflow in Docker and persist artifacts & DB via volumes:

```yaml
version: '3.7'
services:
  mlflow:
    image: mlflow/mlflow:latest
    command: server --backend-store-uri sqlite:////mlflow_data/mlflow.db --default-artifact-root file:////mlflow_data/artifacts --host 0.0.0.0 --port 5050
    ports:
      - '5050:5050'
    volumes:
      - ./mlflow_data:/mlflow_data
    environment:
      - AWS_ACCESS_KEY_ID=...   # if using S3
      - AWS_SECRET_ACCESS_KEY=...
```

---

## Client configuration (how to point your code to the MLflow server)

- Set the tracking URI in your environment or in your project config (example in this repo: `proyecto_final/config.py` sets `MLFLOW_TRACKING_URI`).

```bash
export MLFLOW_TRACKING_URI="http://<server-ip>:5050"
```

- Alternatively, set it programmatically in Python before calling MLflow APIs:

```python
import mlflow
mlflow.set_tracking_uri('http://<server-ip>:5050')
mlflow.set_experiment('EnergyEfficiency')
```

- Verify client can reach server and that the server's artifact root is writable by logging a small artifact and checking server logs.

---

## Troubleshooting checklist (quick)

- [ ] Confirm the `mlflow` process is running and listening on the configured port.
- [ ] Tail `mlflow.log` for errors (`tail -f mlflow.log`).
- [ ] Confirm the artifact root path exists on the server and is writable by the mlflow process (`ls -ld /path/to/artifacts`).
- [ ] If you see `Read-only file system: '/root'` in logs, check whether you started the server with the correct `--default-artifact-root` and that the process environment expands `$HOME` as expected. Prefer absolute paths.
- [ ] If you need concurrent/multi-user usage, migrate backend store from SQLite to Postgres/MySQL and/or use an object store for artifacts.

---

## Security considerations

- `--allowed-hosts '*'` is permissive. In production:
  - Restrict allowed hosts or run behind a reverse proxy (NGINX) that handles TLS and authentication.
  - Consider running MLflow only on internal networks and exposing the UI via a VPN.

- If using S3 or GCS for artifacts, ensure credentials are stored securely (IAM roles, instance profile, or environment injected by a secrets manager).

---

## Useful commands

- Tail logs:

```bash
tail -f mlflow.log
```

- Stop the nohup process (find pid and kill):

```bash
ps aux | grep mlflow
kill <PID>
```

- Start as systemd service (recommended):

```bash
sudo systemctl start mlflow
sudo systemctl status mlflow
sudo journalctl -u mlflow -f
```

---

## Notes & best practices

- For production: use a robust backend store (Postgres/MySQL) and an object store (S3/GCS/Azure) for artifacts.
- Use systemd or container orchestration (Docker Compose/Kubernetes) rather than `nohup` for reliability.
- Keep the MLflow server and artifact store separate from the training client machines if multiple users will log artifacts. Use a shared object store where possible.


If you want, I can:
- Add a systemd unit file to this repo (under `deploy/`) prefilled for your environment.
- Create a Docker Compose file tuned for local testing.
- Search your repo for any existing documentation references and update them to point to this file.

---

_Last updated: 2025-11-02_

