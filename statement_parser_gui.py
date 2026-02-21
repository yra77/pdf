#!/usr/bin/env python3
"""Tkinter GUI for the bank statement parser."""

from __future__ import annotations

import traceback
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from statement_parser import run_parser


class ParserGUI(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("PDF Bank Statement Parser")
        self.geometry("760x340")
        self.resizable(False, False)

        self.input_var = tk.StringVar()
        self.output_var = tk.StringVar()
        self.ocr_var = tk.BooleanVar(value=False)
        self.profile_config_var = tk.StringVar()
        self.profile_name_var = tk.StringVar(value="default")
        self.status_var = tk.StringVar(value="Готово до запуску")

        self._build_ui()

    def _build_ui(self) -> None:
        frame = ttk.Frame(self, padding=16)
        frame.pack(fill="both", expand=True)
        frame.columnconfigure(1, weight=1)

        ttk.Label(frame, text="Вхідний файл (PDF/JPG/PNG):").grid(row=0, column=0, sticky="w", pady=6)
        ttk.Entry(frame, textvariable=self.input_var).grid(row=0, column=1, sticky="ew", pady=6, padx=8)
        ttk.Button(frame, text="Обрати...", command=self._pick_input).grid(row=0, column=2, pady=6)

        ttk.Label(frame, text="Вихідний файл (.csv/.xlsx):").grid(row=1, column=0, sticky="w", pady=6)
        ttk.Entry(frame, textvariable=self.output_var).grid(row=1, column=1, sticky="ew", pady=6, padx=8)
        ttk.Button(frame, text="Зберегти як...", command=self._pick_output).grid(row=1, column=2, pady=6)

        ttk.Checkbutton(frame, text="Примусово OCR", variable=self.ocr_var).grid(
            row=2, column=1, sticky="w", pady=6
        )

        ttk.Label(frame, text="YAML профіль (необов'язково):").grid(row=3, column=0, sticky="w", pady=6)
        ttk.Entry(frame, textvariable=self.profile_config_var).grid(
            row=3, column=1, sticky="ew", pady=6, padx=8
        )
        ttk.Button(frame, text="Обрати...", command=self._pick_profile).grid(row=3, column=2, pady=6)

        ttk.Label(frame, text="Назва профілю:").grid(row=4, column=0, sticky="w", pady=6)
        ttk.Entry(frame, textvariable=self.profile_name_var).grid(row=4, column=1, sticky="ew", pady=6, padx=8)

        ttk.Button(frame, text="Запустити парсинг", command=self._run).grid(
            row=5, column=1, sticky="w", pady=16
        )

        ttk.Label(frame, textvariable=self.status_var).grid(row=6, column=0, columnspan=3, sticky="w")

    def _pick_input(self) -> None:
        selected = filedialog.askopenfilename(
            title="Оберіть PDF",
            filetypes=[
                ("Supported files", "*.pdf *.jpg *.jpeg *.png"),
                ("PDF files", "*.pdf"),
                ("Image files", "*.jpg *.jpeg *.png"),
                ("All files", "*.*"),
            ],
        )
        if selected:
            self.input_var.set(selected)
            if not self.output_var.get():
                suggested = Path(selected).with_suffix(".csv")
                self.output_var.set(str(suggested))

    def _pick_output(self) -> None:
        selected = filedialog.asksaveasfilename(
            title="Збереження результату",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv"), ("Excel", "*.xlsx")],
        )
        if selected:
            self.output_var.set(selected)

    def _pick_profile(self) -> None:
        selected = filedialog.askopenfilename(
            title="Оберіть YAML профіль",
            filetypes=[("YAML", "*.yaml *.yml"), ("All files", "*.*")],
        )
        if selected:
            self.profile_config_var.set(selected)

    def _run(self) -> None:
        input_path = Path(self.input_var.get().strip())
        output_path = Path(self.output_var.get().strip())
        profile_config = self.profile_config_var.get().strip()
        profile_name = self.profile_name_var.get().strip() or "default"

        if not input_path.exists():
            messagebox.showerror("Помилка", "Вкажіть коректний вхідний файл (PDF/JPG/PNG).")
            return

        if output_path.suffix.lower() not in {".csv", ".xlsx", ".xls"}:
            messagebox.showerror("Помилка", "Вихідний файл має бути .csv, .xlsx або .xls")
            return

        profile_path = Path(profile_config) if profile_config else None

        self.status_var.set("Виконується парсинг...")
        self.update_idletasks()

        try:
            parsed_count = run_parser(
                input_path=input_path,
                output_path=output_path,
                force_ocr=self.ocr_var.get(),
                profile_config=profile_path,
                profile_name=profile_name,
            )
        except Exception as exc:  # noqa: BLE001
            self.status_var.set("Помилка")
            messagebox.showerror(
                "Помилка парсингу",
                f"{exc}\n\nДеталі:\n{traceback.format_exc()}",
            )
            return

        self.status_var.set(f"Готово: знайдено {parsed_count} транзакцій")
        messagebox.showinfo("Успіх", f"Оброблено {parsed_count} транзакцій.\nФайл: {output_path}")


def main() -> None:
    app = ParserGUI()
    app.mainloop()


if __name__ == "__main__":
    main()
