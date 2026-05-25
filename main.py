"""
main.py — Entry point for the ExamGuard Proctoring System.

Usage:
    python main.py
    python main.py --student "Alice Johnson"
"""

import sys
import tkinter as tk
from tkinter import messagebox

# ── Dependency check (friendly error instead of traceback) ────────────────────
_MISSING = []
try:
    import cv2
except ImportError:
    _MISSING.append("opencv-python")
try:
    import numpy
except ImportError:
    _MISSING.append("numpy")
try:
    from PIL import Image
except ImportError:
    _MISSING.append("Pillow")

if _MISSING:
    root = tk.Tk()
    root.withdraw()
    messagebox.showerror(
        "Missing Dependencies",
        "Please install the required packages:\n\n"
        f"  pip install {' '.join(_MISSING)}\n\n"
        "Then restart the application.",
    )
    sys.exit(1)

from ui.dashboard import ProctoringDashboard


def main():
    # Parse optional --student flag
    student_name = "Student"
    args = sys.argv[1:]
    if "--student" in args:
        idx = args.index("--student")
        if idx + 1 < len(args):
            student_name = args[idx + 1]

    # ── Ask for student name BEFORE creating the main window ─────────────────
    # Using a separate temporary root so the dialog doesn't block the main loop
    if student_name == "Student":
        tmp = tk.Tk()
        tmp.withdraw()

        dialog = _NameDialog(tmp)
        tmp.wait_window(dialog.top)
        name = dialog.result

        tmp.destroy()
        if name and name.strip():
            student_name = name.strip()

    # ── Create main window ────────────────────────────────────────────────────
    root = tk.Tk()
    root.title("ExamGuard — Proctoring System")

    # Centre window on screen
    screen_w = root.winfo_screenwidth()
    screen_h = root.winfo_screenheight()
    win_w, win_h = 1060, 700
    x = (screen_w - win_w) // 2
    y = (screen_h - win_h) // 2
    root.geometry(f"{win_w}x{win_h}+{x}+{y}")
    root.minsize(win_w, win_h)

    app = ProctoringDashboard(root, student_name=student_name)
    root.protocol("WM_DELETE_WINDOW", app._on_quit)
    root.mainloop()


class _NameDialog:
    """Simple modal dialog to ask for the student name (no simpledialog dependency)."""

    def __init__(self, parent):
        self.result = ""
        self.top = tk.Toplevel(parent)
        self.top.title("ExamGuard — Session Setup")
        self.top.resizable(False, False)
        self.top.configure(bg="#0d1117")
        self.top.grab_set()

        # Centre it
        self.top.geometry("360x160")
        self.top.update_idletasks()
        x = (self.top.winfo_screenwidth() - 360) // 2
        y = (self.top.winfo_screenheight() - 160) // 2
        self.top.geometry(f"360x160+{x}+{y}")

        tk.Label(self.top, text="🎓 ExamGuard",
                 font=("Segoe UI", 14, "bold"),
                 bg="#0d1117", fg="#58a6ff").pack(pady=(18, 4))

        tk.Label(self.top, text="Enter student name (optional):",
                 font=("Segoe UI", 10),
                 bg="#0d1117", fg="#8b949e").pack()

        self._entry = tk.Entry(self.top,
                               font=("Segoe UI", 11),
                               bg="#21262d", fg="#f0f6fc",
                               insertbackground="#f0f6fc",
                               relief="flat", width=28)
        self._entry.pack(pady=10, ipady=6)
        self._entry.focus_set()
        self._entry.bind("<Return>", lambda e: self._ok())

        tk.Button(self.top, text="Start Session →",
                  command=self._ok,
                  font=("Segoe UI", 10, "bold"),
                  bg="#58a6ff", fg="#0d1117",
                  relief="flat", cursor="hand2",
                  padx=16, pady=6).pack()

    def _ok(self):
        self.result = self._entry.get()
        self.top.destroy()


if __name__ == "__main__":
    main()
