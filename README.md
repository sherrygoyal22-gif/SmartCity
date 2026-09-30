# SmartCity — AI Garbage Detection & Reporting System

SmartCity is a web-based platform that uses computer vision to detect
garbage in images and live camera feeds, tag each detection with its real
GPS location, assess a risk level, and route it through a cleanup workflow
until it is resolved. City administrators and citizens can track reports on
an interactive map, monitor case status, and view city-wide cleanliness
analytics from a single dashboard.

## Key Features

- **AI-based garbage detection** — YOLO object-detection model for both
  uploaded images and live browser camera capture, with multi-frame
  confirmation to reduce false positives.
- **GPS-linked reporting** — every detection is tied to a real latitude and
  longitude, either from the reporter's device or a manually supplied
  location.
- **Interactive hotspot map** — Leaflet-based map showing every report as a
  color-coded risk marker (High / Medium / Low), automatic hotspot
  clustering for areas with repeated reports, live distance and direction
  from the viewer and from the admin office, and one-tap navigation.
- **Case management workflow** — reports move through Pending → Assigned →
  Cleaning → Completed, with assignee tracking and status history.
- **Notifications** — automatic reminders and escalations for cases that
  are overdue, plus a read/unread notification center.
- **Reporting & analytics dashboard** — city-wide statistics, trend charts,
  and a filterable, exportable report history (CSV export included).
- **Separate admin, staff and user logins** — administrators sign in at
  `/admin/login`, municipal workers at `/staff/login`, citizens at `/login`
  (with self-registration). Each role gets its own dashboard and menu;
  citizens can only see their own reports.
- **Staff dashboard** — municipal workers see the cases assigned to them with
  the garbage photo, location and Google Maps directions, accept a task,
  upload before/after evidence, mark it cleaned (Assigned → In Progress →
  Cleaned) or report an issue when the location cannot be cleaned.
- **Profile menu** — an avatar dropdown (My Profile, Notifications, Logout)
  in the top bar, as on shopping sites; there are no Login/Logout entries
  in the sidebar.
- **Responsive UI** — usable on desktop and mobile browsers alike.

## Technology Stack

| Layer | Technology |
|---|---|
| Backend | Python, Flask |
| Detection model | YOLO (Ultralytics), OpenCV |
| Database | SQLite |
| Frontend | HTML, CSS, JavaScript |
| Mapping | Leaflet.js, OpenStreetMap |

## Project Structure

```
SmartCity_AI_Garbage_Detection/
├── run.py                  # One-command launcher
├── requirements.txt
├── README.md
└── app/
    ├── app.py              # Flask routes
    ├── database/           # SQLite layer + bundled demo data
    ├── model/              # Trained YOLO weights
    ├── services/           # Detection, risk scoring, location logic
    ├── static/             # CSS, JavaScript, images
    └── templates/          # Jinja2 HTML templates
```

## Getting Started

Requires Python 3.10 or later. From the project folder, run one command:

```bash
python run.py
```

On first run it installs any missing packages, sets up the database, starts
the server and opens `http://127.0.0.1:5000/` in your browser. The site is
also reachable from a phone on the same Wi-Fi at
`http://<your-computer's-LAN-IP>:5000/` (useful for camera and GPS testing).
An internet connection is needed for the map tiles.

### Optional: Admin Office Location

To display the admin office marker and "distance from admin" on the map,
set the following environment variables before starting the server:

```bash
export SMARTCITY_ADMIN_LAT=<latitude>
export SMARTCITY_ADMIN_LNG=<longitude>
```

## Accounts and Roles

On the first run three demo accounts are created (the credentials are also
printed in the terminal):

| Role | Sign in at | Username | Password |
|---|---|---|---|
| Admin | `/admin/login` | `admin` | `admin123` |
| Staff (demo worker) | `/staff/login` | `staff` | `staff123` |
| Citizen (demo) | `/login` | `user` | `user123` |

If the admin login ever says "wrong credentials" (for example an older
database already contains an admin with a different password), run
`python run.py --reset-admin` to set the admin back to `admin` / `admin123`.

Change the admin password under **Profile → Change password**, or set
`SMARTCITY_ADMIN_USER` / `SMARTCITY_ADMIN_PASSWORD` before the very first
start. Reports that existed before accounts were added belong to the demo
citizen. Anyone can register a new citizen account; admin accounts are not
self-service.

**Admin portal**: Dashboard, All Garbage Reports (search, filter, CSV
export), Garbage Map & Hotspots, Case Management (view, assign, start
cleaning, upload after-photo, mark resolved, delete case), Live Detection,
User Reports (per-citizen counts), Notifications, User Management (add,
deactivate, delete), Admin Location (click the map to set the office point),
Profile.

**Staff portal**: Dashboard (assigned / in progress / cleaned counts), My
Assigned Cases, task page with photo, location, directions, Accept & Start,
before-cleaning photo, after-cleaning photo, Mark as Cleaned, Report Issue,
Notifications, Profile. Admins create staff accounts in **User Management →
Add user → Staff**, then assign cases to them from **Case Management**. A
reported issue returns the case to Pending and notifies the admin.

**Citizen portal**: Dashboard, Report Garbage (upload or camera, GPS),
My Reports (with CSV export), Report Status timeline with before/after
photos once resolved, Garbage Map, Notifications about their own reports,
Profile.

## Usage Overview

1. **Report Garbage** (citizen) — upload a photo or use the camera to scan
   for garbage; every valid detection opens a case automatically.
2. **View Map** — see all reports, filter by risk or status, and open any
   location (admins get links into the case system).
3. **Manage Cases** (admin) — assign, clean, add the after-photo and mark
   the case resolved; the citizen is notified at each step.
4. **Review Reports** — search, filter and export as CSV (admins: all
   reports; citizens: their own).

## Author

Developed as a minor project for the Bachelor of Computer Applications
(BCA) programme.

## Troubleshooting

- **Detection shows an error** - stop the server and run `python run.py`
  again. It upgrades the detection engine (the bundled model needs
  Ultralytics 8.4 or newer) and prints `Checking detection model ... OK`
  or the exact reason it failed.
- **Live camera does not start** - browsers only allow the camera on
  `http://127.0.0.1:5000` or `https://` addresses; allow camera and
  location when the browser asks. On a phone over Wi-Fi use an HTTPS tunnel.
- **Live camera says garbage was not confirmed** - hold the camera steady on
  the garbage for a moment; 2 of 3 captured frames must show it.


## Use it on phones and tablets (Android, iPhone, iPad) - no PC needed

SmartCity is now an installable web app (PWA). Once it is hosted at an
`https://` address, anyone can open the link on any phone or tablet:

- **Android (Chrome / Edge / Samsung Internet):** an "Install SmartCity"
  banner appears, or use menu -> *Install app / Add to Home screen*.
- **iPhone / iPad (Safari):** tap Share -> *Add to Home Screen*.
- **PC:** the install icon in the address bar of Chrome / Edge.

The camera and GPS only work over **HTTPS** (or `localhost`), so the app must
be hosted online rather than opened from a PC's `http://192.168...` address.

### Hosting it so the PC can stay off

1. Put the project on GitHub.
2. On Render.com choose *New -> Blueprint* and pick the repo. `render.yaml`
   builds it (`requirements-deploy.txt`), starts it with gunicorn (`wsgi.py`)
   and gives you the https link to share.
3. Use a plan with at least ~1 GB RAM: the YOLO model will not fit on a 512 MB
   free instance. The disk in `render.yaml` keeps the database between deploys;
   uploaded photos live in `app/static/uploads` and `app/static/results`, which
   need a persistent disk or object storage on any host that resets files.

Any host that can run `gunicorn wsgi:app` works the same way (Railway, Fly.io,
a VPS). Set `SMARTCITY_BEHIND_PROXY=1` when the host terminates HTTPS for you.

## Admin location

*Admin Location* now takes a **State / UT** and **City**. Latitude and
longitude are optional: they are filled in automatically when a state or city
is entered (OpenStreetMap lookup, with a built-in state-capital fallback when
offline), and moving the pin fills the city and state back in.


## Keeping your reports and cases safe

Reports, cases, users and notifications are stored in `app/database/smartcity.db`,
and photos in `app/static/uploads` and `app/static/results`. The app never clears
them on start-up; they are only lost if those files are replaced or deleted.

- **Updating the code:** use a *code-only* zip (no `smartcity.db`, no photos) and
  extract it over your folder. Never copy an old `smartcity.db` over the current one.
- **Automatic backups:** every start-up saves a copy to `app/database/backups/`
  (newest 10 kept). To restore, stop the app and copy a backup over `smartcity.db`.
- **Keep data outside the code (recommended):** set `SMARTCITY_DATA_DIR` to a
  folder of your choice, e.g. `set SMARTCITY_DATA_DIR=D:\SmartCityData` (Windows)
  or `export SMARTCITY_DATA_DIR=~/smartcity-data`. On first start the existing
  database and photos are moved there, and replacing the code folder can no
  longer touch them.
- **Hosted:** use a host with a persistent disk (see `render.yaml`). Without
  one, most hosts reset the files on every deploy and the data disappears.
- Always open the app from the same folder / link; a second copy of the project
  has its own separate database.
