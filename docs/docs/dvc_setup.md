# 🧱 DVC Setup Guide

This guide explains how to set up Data Version Control (DVC) for this project
and connect it to a DigitalOcean droplet for secure dataset storage.

⸻

## ⚙️ 1️⃣ Install DVC

Install DVC with SSH support:

```bash
pip install "dvc[ssh]"
```

Or, inside a Jupyter notebook:

```bash
%pip install "dvc[ssh]"
```

Verify installation:

```bash
dvc --version
```

Expected output:

```bash
dvc 3.x.x
```

⸻

## 🧭 Initialize DVC in the Project

From the project’s root directory (same level as data/ and notebooks/):

```bash
dvc init
```

This creates:

```bash
.dvc/
.dvcignore
.gitignore   (updated)
```

Commit the configuration:

```bash
git add .dvc .dvcignore .gitignore
git commit -m "Initialize DVC for project"
```

⸻

## ☁️ Configure DigitalOcean as DVC Remote

We use an SSH-based remote that points to your droplet.
Your droplet is accessible at:

```bash
root@138.68.15.126
```

and stores data at:

```bash
/opt/dvc_storage
```

Add it as your default remote:

```bash
dvc remote add -d do_remote ssh://root@138.68.15.126/opt/dvc_storage
```

Specify which SSH key DVC should use (replace the path with yours):

```bash
dvc remote modify do_remote keyfile /Users/cbarrien/mlops-ssh
```

Verify:

```bash
dvc remote list
```

Expected:

```bash
do_remote  ssh://root@138.68.15.126/opt/dvc_storage (default)
```

⸻

## 💾 Add Datasets to DVC

Track the datasets you want versioned:

```bash
dvc add data/raw/energy_noisy.csv
dvc add data/interim/energy_drop.csv
dvc add data/interim/energy_fill.csv
```

DVC will:
• Move each file to its internal cache (.dvc/cache/)
• Create lightweight .dvc tracking files
• Automatically update .gitignore to exclude large files

Commit these tracking files:

```bash
git add data/*.dvc .gitignore
git commit -m "Track raw and interim datasets with DVC"
```

⸻

## 🚀 Push Data to DigitalOcean

Push your data to your droplet:

```bash
dvc push -v
```

This will connect via SSH and upload all tracked data into:

```bash
/opt/dvc_storage
```

To avoid repeated passphrase prompts, start the SSH agent once per session:

```bash
eval "$(ssh-agent -s)"
ssh-add /Users/cbarrien/mlops-ssh
```

Then push again:

```bash
dvc push
```

⸻

## 🔁 Verify and Sync Data

Check if data and Git are in sync:

```bash
dvc status
```

To restore or clone the project on another machine:

```bash
git clone <your_repo_url>
cd <your_project_folder>
dvc pull
```

Ensure that the same SSH key (mlops-ssh) is available locally,
and that it’s added to ~/.ssh/authorized_keys on the droplet.

⸻

## 📘 Example Workflow Summary

Initialize

```bash
git init
dvc init
```

Configure remote

```bash
dvc remote add -d do_remote ssh://root@138.68.15.126/opt/dvc_storage
dvc remote modify do_remote keyfile /Users/cbarrien/mlops-ssh
```

Add and track data

```bash
dvc add data/raw/energy_noisy.csv
dvc add data/interim/energy_drop.csv
dvc add data/interim/energy_fill.csv
```

Commit and push

```bash
git add .
git commit -m "Initial commit with DVC configuration and datasets"
dvc push
```

⸻

## ✅ Final Notes
• Git stores code and metadata.
• DVC stores datasets in your DigitalOcean droplet.
• To reproduce the environment and data anywhere:

```bash
git clone <repo_url>
dvc pull
```
