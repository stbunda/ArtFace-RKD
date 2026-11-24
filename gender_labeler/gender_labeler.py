"""
Gender Labeling Tool
--------------------
Interactive GUI to label persons in paintings as Male, Female, or Unknown.

One label per PERSON (person_id folder).  All face crops for that person are
shown together, along with the corresponding paintings (with bounding boxes).

Labels are saved to gender_labels.csv and the tool resumes automatically.
"""

import csv
import json
import xml.etree.ElementTree as ET
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

# -- Configuration -----------------------------------------------------------
import sys as _sys

# When frozen by PyInstaller (--onefile), __file__ is inside a temp extraction
# folder. Use the exe's own directory instead so config/CSV sit next to it.
if getattr(_sys, "frozen", False):
    BASE_DIR = Path(_sys.executable).parent
else:
    BASE_DIR = Path(__file__).parent
DATA_DIR         = BASE_DIR.parent / "data"   # repo-level data/ folder
CONFIG_FILE      = BASE_DIR / "gender_labeler_config.json"

# Mutable paths – overridden by the startup dialog
SITTERS_DIR      = DATA_DIR / "sitters"
DETECTOR_DIR     = DATA_DIR / "detections"
PAINTINGS_DIR    = DATA_DIR / "paintings"
XML_PATH         = DATA_DIR / "rkd_metadata.xml"
OUTPUT_CSV       = BASE_DIR / "gender_labels.csv"

FACE_THUMB_SIZE     = (160, 160)   # each face crop thumbnail
PAINTING_THUMB_SIZE = (220, 220)   # each painting thumbnail


# -- Config persistence ------------------------------------------------------
def load_config() -> dict:
    if CONFIG_FILE.exists():
        try:
            return json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def save_config(cfg: dict):
    CONFIG_FILE.write_text(json.dumps(cfg, indent=2), encoding="utf-8")


# -- Startup folder-picker dialog --------------------------------------------
class SettingsDialog:
    """Modal dialog shown at startup to confirm/change all folder paths."""

    def __init__(self, parent):
        self.result = None   # set to dict of paths on OK, None on Cancel

        cfg = load_config()

        self.win = tk.Toplevel(parent)
        self.win.title("Gender Labeler – Folder Settings")
        self.win.configure(bg="#1e1e1e")
        self.win.resizable(True, False)
        self.win.grab_set()

        fields = [
            ("Faces folder (sitters/)",        "sitters_dir",   str(cfg.get("sitters_dir",   SITTERS_DIR)),     "dir"),
            ("Paintings folder (detections/)", "detector_dir", str(cfg.get("detector_dir", DETECTOR_DIR)),    "dir"),
            ("Metadata XML",                   "xml_path",      str(cfg.get("xml_path",      XML_PATH)),          "file"),
            ("Output CSV",                     "csv_path",      str(cfg.get("csv_path",      OUTPUT_CSV)),        "file"),
        ]

        self._vars = {}
        pad = dict(padx=10, pady=4)

        tk.Label(self.win, text="Configure paths before starting",
                 bg="#1e1e1e", fg="#aaaaaa",
                 font=("Helvetica", 10)).grid(row=0, column=0, columnspan=3,
                                              sticky="w", **pad)

        for r, (label, key, default, kind) in enumerate(fields, start=1):
            var = tk.StringVar(value=default)
            self._vars[key] = var

            tk.Label(self.win, text=label, bg="#1e1e1e", fg="#cccccc",
                     font=("Helvetica", 10), anchor="w", width=32
                     ).grid(row=r, column=0, sticky="w", **pad)

            tk.Entry(self.win, textvariable=var, width=55,
                     bg="#2d2d2d", fg="#ffffff", insertbackground="white",
                     relief="flat"
                     ).grid(row=r, column=1, sticky="ew", **pad)

            cmd = (self._browse_dir if kind == "dir" else self._browse_file)
            tk.Button(self.win, text="Browse",
                      command=lambda v=var, k=kind: self._browse(v, k),
                      bg="#374151", fg="#d1d5db", relief="flat",
                      cursor="hand2", padx=8
                      ).grid(row=r, column=2, **pad)

        self.win.columnconfigure(1, weight=1)

        btn_row = len(fields) + 1
        tk.Button(self.win, text="Start", font=("Helvetica", 11, "bold"),
                  bg="#2563eb", fg="white", relief="flat", padx=20, pady=6,
                  cursor="hand2", command=self._ok
                  ).grid(row=btn_row, column=0, columnspan=3, pady=12)

        self.win.bind("<Return>", lambda _: self._ok())
        self.win.bind("<Escape>", lambda _: self.win.destroy())
        self.win.protocol("WM_DELETE_WINDOW", self.win.destroy)

    def _browse(self, var: tk.StringVar, kind: str):
        if kind == "dir":
            path = filedialog.askdirectory(
                initialdir=var.get() or str(BASE_DIR), title="Select folder")
        else:
            path = filedialog.askopenfilename(
                initialdir=str(Path(var.get()).parent) if var.get() else str(BASE_DIR),
                title="Select file")
        if path:
            var.set(path)

    # kept for backward compat (not used directly)
    def _browse_dir(self, var):  self._browse(var, "dir")
    def _browse_file(self, var): self._browse(var, "file")

    def _ok(self):
        self.result = {k: v.get() for k, v in self._vars.items()}
        save_config(self.result)
        self.win.destroy()


# -- XML metadata ------------------------------------------------------------
def load_metadata():
    """Returns person_names: dict  person_id -> full_name."""
    tree = ET.parse(XML_PATH)
    root = tree.getroot()

    person_names: dict[str, str] = {}
    for record in root:
        ids   = [m.text.strip() for m in record.findall(".//persoonsnummer")
                 if m.text and m.text.strip() not in ("", "\n")]
        names = [m.text.strip() for m in record.findall(".//naam_volledig")
                 if m.text and m.text.strip() not in ("", "\n")]
        for pid, name in zip(ids, names):
            person_names[pid] = name

    return person_names


# -- Build per-person item list ----------------------------------------------
def build_person_list(person_names):
    """
    Returns a list of dicts, one per person_id folder:
        person_id   str
        full_name   str
        crops       list of {face: Path, painting: Path|None, painting_stem: str}
    """
    persons = []
    for person_dir in sorted(SITTERS_DIR.iterdir()):
        if not person_dir.is_dir():
            continue
        person_id = person_dir.name
        full_name = person_names.get(person_id, f"Person {person_id}")

        crops = []
        for face_file in sorted(person_dir.glob("*.jp*g")):
            painting_stem = face_file.stem.rsplit("-", 1)[0]
            detector_img  = DETECTOR_DIR / f"{painting_stem}.jpg"
            if not detector_img.exists():
                detector_img = PAINTINGS_DIR / f"{painting_stem}.jpg"
            if not detector_img.exists():
                detector_img = None
            crops.append({"face": face_file, "painting": detector_img,
                           "painting_stem": painting_stem})

        if crops:
            persons.append({"person_id": person_id,
                            "full_name": full_name,
                            "crops": crops})
    return persons


# -- Persistence -------------------------------------------------------------
CSV_FIELDS = ["painting_stem", "person_id", "full_name", "gender"]


def load_existing_labels():
    """
    Reads the painting-centric CSV and returns dict  person_id -> row dict.
    Multiple painting rows for the same person are collapsed to one entry
    (the UI only needs to know whether a person has been labelled).
    """
    labels: dict[str, dict] = {}
    if OUTPUT_CSV.exists():
        with open(OUTPUT_CSV, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                pid = row.get("person_id", "")
                if pid and pid not in labels:
                    labels[pid] = {"person_id": pid,
                                   "full_name": row.get("full_name", ""),
                                   "gender":    row.get("gender", "")}
    return labels


def persist_labels(labels: dict, person_lookup: dict):
    """
    Write one CSV row per crop (painting) so each painting has its own row.
    person_lookup: person_id -> person dict (contains 'crops' list).
    """
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for pid, row in labels.items():
            crops = person_lookup.get(pid, {}).get("crops", [])
            if crops:
                for crop in crops:
                    writer.writerow({
                        "painting_stem": crop["painting_stem"],
                        "person_id":     pid,
                        "full_name":     row["full_name"],
                        "gender":        row["gender"],
                    })
            else:
                # fallback if person no longer in directory
                writer.writerow({"painting_stem": "", "person_id": pid,
                                 "full_name": row["full_name"],
                                 "gender": row["gender"]})


# -- Image helpers -----------------------------------------------------------
def load_thumb(path, max_size):
    if path is None or not Path(path).exists():
        img = Image.new("RGB", max_size, color=(50, 50, 50))
    else:
        img = Image.open(path).convert("RGB")
        img.thumbnail(max_size, Image.LANCZOS)
    return ImageTk.PhotoImage(img)


# -- Main GUI ----------------------------------------------------------------
class GenderLabelerApp:
    def __init__(self, root, persons, existing_labels):
        self.root    = root
        self.persons = persons
        self.labels  = existing_labels   # person_id -> row dict
        self.person_lookup = {p["person_id"]: p for p in persons}

        # Find first unlabelled person
        self.index = next(
            (i for i, p in enumerate(persons)
             if p["person_id"] not in self.labels),
            len(persons)
        )

        self._img_refs = []   # keep PhotoImage references alive
        self._build_ui()
        self._show_current()

    # -- UI construction -----------------------------------------------------
    def _build_ui(self):
        self.root.title("Gender Labeler - ArtFace")
        self.root.configure(bg="#1e1e1e")
        self.root.resizable(True, True)

        # Top bar
        top = tk.Frame(self.root, bg="#1e1e1e")
        top.pack(fill="x", padx=12, pady=6)

        self.progress_var = tk.StringVar()
        tk.Label(top, textvariable=self.progress_var, bg="#1e1e1e",
                 fg="#aaaaaa", font=("Helvetica", 11)).pack(side="left")

        self.name_var = tk.StringVar()
        tk.Label(top, textvariable=self.name_var, bg="#1e1e1e",
                 fg="#ffffff", font=("Helvetica", 15, "bold")).pack(side="left", padx=16)

        self.id_var = tk.StringVar()
        tk.Label(top, textvariable=self.id_var, bg="#1e1e1e",
                 fg="#777777", font=("Helvetica", 10)).pack(side="left")

        self.stats_var = tk.StringVar()
        tk.Label(top, textvariable=self.stats_var, bg="#1e1e1e",
                 fg="#66aaff", font=("Helvetica", 10)).pack(side="right")

        # Scrollable image area (horizontal scroll for many crops)
        canvas_outer = tk.Frame(self.root, bg="#1e1e1e")
        canvas_outer.pack(fill="both", expand=True, padx=8, pady=4)

        self.canvas = tk.Canvas(canvas_outer, bg="#1e1e1e", highlightthickness=0)
        hbar = tk.Scrollbar(canvas_outer, orient="horizontal",
                            command=self.canvas.xview)
        self.canvas.configure(xscrollcommand=hbar.set)
        hbar.pack(side="bottom", fill="x")
        self.canvas.pack(side="left", fill="both", expand=True)

        self.img_container = tk.Frame(self.canvas, bg="#1e1e1e")
        self.canvas_window = self.canvas.create_window(
            (0, 0), window=self.img_container, anchor="nw")
        self.img_container.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.bind_all(
            "<MouseWheel>",
            lambda e: self.canvas.xview_scroll(-1 * (e.delta // 120), "units"))

        # Label buttons
        btn_frame = tk.Frame(self.root, bg="#1e1e1e")
        btn_frame.pack(fill="x", padx=10, pady=8)

        btn_cfg = dict(font=("Helvetica", 13, "bold"), width=14,
                       relief="flat", cursor="hand2", bd=0, pady=8)

        self.btn_male = tk.Button(
            btn_frame, text="Male", bg="#2563eb", fg="white",
            activebackground="#1d4ed8",
            command=lambda: self._label("male"), **btn_cfg)
        self.btn_male.pack(side="left", expand=True, fill="x", padx=(0, 4))

        self.btn_female = tk.Button(
            btn_frame, text="Female", bg="#db2777", fg="white",
            activebackground="#be185d",
            command=lambda: self._label("female"), **btn_cfg)
        self.btn_female.pack(side="left", expand=True, fill="x", padx=4)

        self.btn_unknown = tk.Button(
            btn_frame, text="Don't know", bg="#4b5563", fg="white",
            activebackground="#374151",
            command=lambda: self._label("unknown"), **btn_cfg)
        self.btn_unknown.pack(side="left", expand=True, fill="x", padx=(4, 0))

        # Navigation row
        nav_frame = tk.Frame(self.root, bg="#1e1e1e")
        nav_frame.pack(fill="x", padx=10, pady=(0, 8))

        nav_cfg = dict(font=("Helvetica", 10), relief="flat", cursor="hand2",
                       bg="#374151", fg="#d1d5db",
                       activebackground="#4b5563", padx=10, pady=4)

        tk.Button(nav_frame, text="Back",
                  command=self._go_back, **nav_cfg).pack(side="left")
        tk.Button(nav_frame, text="Skip",
                  command=self._skip, **nav_cfg).pack(side="left", padx=6)
        tk.Button(nav_frame, text="Jump to first unlabelled",
                  command=self._jump_to_first_unlabelled,
                  **nav_cfg).pack(side="left")
        tk.Button(nav_frame, text="Settings...",
                  command=self._open_settings,
                  **nav_cfg).pack(side="left", padx=6)

        tk.Label(nav_frame,
                 text="  m = Male  |  f = Female  |  u = Don't know  |  <- -> navigate",
                 bg="#1e1e1e", fg="#555555",
                 font=("Helvetica", 9)).pack(side="right")

        # Keyboard shortcuts
        self.root.bind("<m>", lambda _: self._label("male"))
        self.root.bind("<f>", lambda _: self._label("female"))
        self.root.bind("<u>", lambda _: self._label("unknown"))
        self.root.bind("<Left>",  lambda _: self._go_back())
        self.root.bind("<Right>", lambda _: self._skip())

    # -- Rendering -----------------------------------------------------------
    def _show_current(self):
        total    = len(self.persons)
        labelled = len(self.labels)

        if self.index >= total:
            self._show_done()
            return

        person = self.persons[self.index]
        pid    = person["person_id"]

        self.progress_var.set(f"{self.index + 1} / {total}")
        self.name_var.set(person["full_name"])
        self.id_var.set(f"(person id: {pid}  |  {len(person['crops'])} painting(s))")

        m  = sum(1 for r in self.labels.values() if r["gender"] == "male")
        fe = sum(1 for r in self.labels.values() if r["gender"] == "female")
        un = sum(1 for r in self.labels.values() if r["gender"] == "unknown")
        self.stats_var.set(f"Labelled: {labelled}  M:{m}  F:{fe}  ?:{un}")

        # Highlight existing label on buttons
        existing = self.labels.get(pid, {}).get("gender")
        self.btn_male.config(   relief="sunken" if existing == "male"    else "flat")
        self.btn_female.config( relief="sunken" if existing == "female"  else "flat")
        self.btn_unknown.config(relief="sunken" if existing == "unknown" else "flat")

        # Rebuild image grid
        for w in self.img_container.winfo_children():
            w.destroy()
        self._img_refs.clear()

        # One column per crop: painting stem label, face thumb, painting thumb
        for col, crop in enumerate(person["crops"]):
            col_frame = tk.Frame(self.img_container, bg="#1e1e1e")
            col_frame.grid(row=0, column=col, padx=6, pady=4)

            tk.Label(col_frame, text=crop["painting_stem"],
                     bg="#1e1e1e", fg="#666666", font=("Helvetica", 8),
                     wraplength=PAINTING_THUMB_SIZE[0]).pack()

            # Face crop
            face_panel = tk.Frame(col_frame, bg="#2a2a2a")
            face_panel.pack()
            tk.Label(face_panel, text="Face crop", bg="#2a2a2a",
                     fg="#888888", font=("Helvetica", 8)).pack(pady=(3, 0))
            face_img = load_thumb(crop["face"], FACE_THUMB_SIZE)
            self._img_refs.append(face_img)
            tk.Label(face_panel, image=face_img, bg="#2a2a2a").pack(padx=4, pady=4)

            # Painting with bounding box
            paint_panel = tk.Frame(col_frame, bg="#2a2a2a")
            paint_panel.pack(pady=(4, 0))
            tk.Label(paint_panel, text="Painting (bbox)", bg="#2a2a2a",
                     fg="#888888", font=("Helvetica", 8)).pack(pady=(3, 0))
            paint_img = load_thumb(crop["painting"], PAINTING_THUMB_SIZE)
            self._img_refs.append(paint_img)
            tk.Label(paint_panel, image=paint_img, bg="#2a2a2a").pack(padx=4, pady=4)

        self.canvas.xview_moveto(0)

    def _show_done(self):
        for w in self.img_container.winfo_children():
            w.destroy()
        self._img_refs.clear()
        self.progress_var.set(f"All {len(self.persons)} persons labelled!")
        self.name_var.set("Done!")
        self.id_var.set("")
        m  = sum(1 for r in self.labels.values() if r["gender"] == "male")
        fe = sum(1 for r in self.labels.values() if r["gender"] == "female")
        un = sum(1 for r in self.labels.values() if r["gender"] == "unknown")
        self.stats_var.set(f"Total: {len(self.labels)}  M:{m}  F:{fe}  ?:{un}")

    # -- Actions -------------------------------------------------------------
    def _label(self, gender: str):
        if self.index >= len(self.persons):
            return
        person = self.persons[self.index]
        pid    = person["person_id"]
        self.labels[pid] = {"person_id": pid,
                            "full_name": person["full_name"],
                            "gender":    gender}
        persist_labels(self.labels, self.person_lookup)
        self.index += 1
        self._show_current()

    def _go_back(self):
        if self.index > 0:
            self.index -= 1
            self._show_current()

    def _skip(self):
        if self.index < len(self.persons) - 1:
            self.index += 1
            self._show_current()

    def _jump_to_first_unlabelled(self):
        for i, p in enumerate(self.persons):
            if p["person_id"] not in self.labels:
                self.index = i
                self._show_current()
                return
        messagebox.showinfo("Done", "All persons have been labelled!")

    def _open_settings(self):
        dlg = SettingsDialog(self.root)
        self.root.wait_window(dlg.win)
        if dlg.result:
            messagebox.showinfo(
                "Settings saved",
                "Paths saved. Restart the tool to apply the new folders.")


# -- Entry point -------------------------------------------------------------
def main():
    global SITTERS_DIR, DETECTOR_DIR, XML_PATH, OUTPUT_CSV

    # Show settings dialog first
    pre = tk.Tk()
    pre.withdraw()
    dlg = SettingsDialog(pre)
    pre.wait_window(dlg.win)

    if dlg.result is None:
        # User closed the dialog without clicking Start
        pre.destroy()
        return

    # Apply chosen paths
    SITTERS_DIR = Path(dlg.result["sitters_dir"])
    DETECTOR_DIR = Path(dlg.result["detector_dir"])
    XML_PATH     = Path(dlg.result["xml_path"])
    OUTPUT_CSV   = Path(dlg.result["csv_path"])
    pre.destroy()

    print("Loading XML metadata...")
    person_names = load_metadata()
    print(f"  {len(person_names)} person names loaded")

    print("Scanning sitters/ directory...")
    persons = build_person_list(person_names)
    print(f"  {len(persons)} persons found")

    if not persons:
        messagebox.showerror("No faces found",
                             f"No person folders with images found in:\n{SITTERS_DIR}")
        return

    existing_labels = load_existing_labels()
    print(f"  Resuming: {len(existing_labels)} already labelled")

    # Backfill: re-persist existing labels in the new painting-centric format
    person_lookup = {p["person_id"]: p for p in persons}
    if existing_labels:
        persist_labels(existing_labels, person_lookup)
        print("  Saved existing labels in painting-centric format")

    root = tk.Tk()
    root.geometry("1000x780")
    GenderLabelerApp(root, persons, existing_labels)
    root.mainloop()


if __name__ == "__main__":
    main()
