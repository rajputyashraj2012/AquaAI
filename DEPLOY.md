# Deploying AquaAI

## Files in this folder
| File | Purpose |
|---|---|
| `aquaai.py` | The application (user login, admin console, all AquaAI pages) |
| `requirements.txt` | Python dependencies |
| `Dockerfile`, `.dockerignore` | Container build for any Docker host |
| `render.yaml` | One-click Render Blueprint (service + persistent disk) |
| `.streamlit/config.toml` | Server settings (headless, XSRF protection on) |
| `.streamlit/secrets.toml.example` | Template for admin credentials |
| `.gitignore` | Keeps secrets and the user database out of Git |

## Important: where users are stored
Accounts live in a SQLite file. It must be on a disk that survives restarts.
Set `AQUAAI_DB_PATH` to a path on that disk (the Dockerfile uses `/data`).
Streamlit Community Cloud has no persistent disk, so use it for demos only:
registered users will be lost whenever the app restarts or redeploys.

## Option A: Render (recommended)
1. Push this folder to a GitHub repository (secrets are git-ignored).
2. Render dashboard, New, Blueprint, pick the repo. `render.yaml` creates the web
   service and a 1 GB disk at `/data`.
3. When prompted, enter `AQUAAI_ADMIN_USERNAME` and `AQUAAI_ADMIN_PASSWORD`
   (password: 10+ characters, with a letter and a number).
4. Deploy. First start takes about 1 to 2 minutes (the AI model trains once, then is cached).
5. Open the site, then the **Admin** tab to sign in. Change the password under
   Admin, Admin Profile, then remove `AQUAAI_ADMIN_PASSWORD` from Render's environment.

## Option B: Any Docker host (VPS, Fly.io, Railway, etc.)
```bash
docker build -t aquaai .
docker run -d --name aquaai -p 8501:8501 \
  -v aquaai-data:/data \
  -e AQUAAI_ADMIN_USERNAME="admin@yourdomain.com" \
  -e AQUAAI_ADMIN_PASSWORD="CHANGE-ME-long-passphrase-1" \
  aquaai
```
Then put a reverse proxy with HTTPS in front (Caddy, Nginx, or your platform's
built-in TLS). Do not expose plain HTTP: passwords travel over this connection.
On Railway/Fly.io, attach a volume mounted at `/data` and set the same variables.

## Option C: Run directly with Python
```bash
python3 -m pip install -r requirements.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml   # then edit it
streamlit run aquaai.py
```

## Streamlit Community Cloud (demo only)
Deploy from GitHub with `aquaai.py` as the main file, paste the `[admin]` block
into Settings, Secrets. Remember: no persistent disk.

## Adding more administrators (up to 3 in total)
Only the first administrator comes from the environment/secrets. After signing in,
open **Admin Accounts** in the sidebar to create up to two more administrators
(username + temporary password; 10+ characters with a letter and a number), or to
remove one. Each new administrator should change the temporary password in
**Admin Profile**. You can't remove your own account, so at least one admin always
remains. Creations and removals appear in Security Monitoring, Administrator Activity.
The limit is `MAX_ADMINS` near the top of the admin section in `aquaai.py`.

## After deploying
- Sign in on the **Admin** tab, review Security Monitoring regularly.
- Back up the database file (`/data/aquaai_users.db`) on a schedule.
- Run a single instance only. SQLite does not support several app copies
  sharing one file. For heavy traffic, move to PostgreSQL.
- Sessions end when the browser tab is refreshed (members and admins sign in again).
- To rotate the admin password later, use Admin Profile; the environment
  variables are only read when no admin exists yet.
