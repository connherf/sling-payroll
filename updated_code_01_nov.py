#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Weekly Sling payroll workbook (Sat→Fri) with guardrails & schedule trim.

INCLUDED:
- UHL employees ARE included (not excluded).

EXCLUDED (by default, editable via --exclude):
- Rows where Position/Location matches: Solid Fire, Administration

Rules:
- Zero the day if raw OR trimmed duration > 13h
- Zero the day if NO clock-out and Auto != Yes OR text says "NO CLOCK OUT"
- No unpaid break deduction (0.0h)
- Schedule-trim by Position:
    Firestoppers: 07:30–16:00 Mon–Thu, 07:30–15:00 Fri
    Painters:     07:00–15:30 Mon–Thu, 07:00–14:00 Fri
- Daily policy: Mon–Thu min(Worked,8), Fri min(Worked,7), Sat full, Sun 0
- Leave/Training/Holiday: full weekday day (Mon–Thu 8h, Fri 7h)
- No-carry: blank weekly total if no valid punches AND no leave
- NEW: If a single day's effective worked > 24h, pay 6.00h (instead of 0)
- Outputs: Weekly Totals (Approved mirrors into Weekly Net), Audit (spaced + plain),
           QC – Trimmed, QC – Zeroed, QC – Excluded, Guardrails
- Overtime intentionally omitted.
"""

import argparse
import re
from datetime import datetime
from typing import Optional, Tuple

import numpy as np
import pandas as pd

# ---------------- Configuration defaults ----------------
THRESH_HOURS_13 = 13.0          # zero if raw/trimmed > 13h
GUARD_24H_PAYS = 6.0            # if effective worked > 24h → pay fixed 6.0h
BREAK_DEDUCT_HOURS = 0.0        # no break deduction
DEFAULT_EXCLUDE_REGEX = r"solid\s*fire|\badministration\b"  # UHL included

# ---------------- Helper regexes ----------------
VTH_REGEXES = [
    re.compile(r"\bannual\s*leave\b", re.I),
    re.compile(r"\bvacation\b", re.I),
    re.compile(r"\btraining\b", re.I),
    re.compile(r"\b(induction|course)\b", re.I),
    re.compile(
        r"\b(bank\s*holiday|public\s*holiday|company\s*holiday|stat(?:utory)?\s*holiday|holiday)\b", re.I),
    re.compile(r"(?<!\w)A/?L(?!\w)", re.I),
    re.compile(r"(?<!\w)B/?H(?!\w)", re.I),
    re.compile(r"(?<!\w)P/?H(?!\w)", re.I),
]

# ---------------- Helpers ----------------


def to_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce")


def latest_week_window(dates_series: pd.Series) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """Get Sat→Fri window that contains the latest date in the export."""
    dates = to_date(dates_series).dropna()
    if dates.empty:
        raise RuntimeError("Export has no valid dates.")
    latest = dates.max().normalize()
    # Saturday anchor (Mon=0..Sun=6 → Sat=5)
    shift = (latest.weekday() - 5) % 7
    week_start = (latest - pd.Timedelta(days=shift)).normalize()
    week_end = week_start + pd.Timedelta(days=6)
    return week_start, week_end


def parse_excel_time(val) -> Optional[int]:
    """Return minutes since midnight (0..1439) or None from Excel time / text."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    if isinstance(val, pd.Timestamp):
        return val.hour * 60 + val.minute
    if isinstance(val, (int, float)) and 0 <= val <= 1:
        mins = int(round(val * 24 * 60))               # Excel fraction of day
        return max(0, min(mins, 24 * 60 - 1))
    s = str(val).strip()
    if s == "" or s.lower() in {"nan", "none", "-"}:
        return None
    s = s.replace(".,", ":").replace(".", ":").upper()
    for fmt in ("%I:%M %p", "%I %p", "%H:%M", "%H.%M", "%H:%M:%S"):
        try:
            t = datetime.strptime(s, fmt)
            return t.hour * 60 + t.minute
        except Exception:
            pass
    if s.isdigit() and (3 <= len(s) <= 4):
        hh = int(s[:-2])
        mm = int(s[-2:])
        if 0 <= hh <= 24 and 0 <= mm < 60:
            return min(hh * 60 + mm, 24 * 60 - 1)
    return None


def parse_hours(x) -> Optional[float]:
    """Parse HH:MM or numeric hours to float hours."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    s = str(x).strip()
    if s == "" or s.lower() in {"nan", "none", "-"}:
        return None
    if ":" in s and not any(c.isalpha() for c in s):
        parts = s.split(":")
        hh = int(parts[0])
        mm = int(parts[1]) if len(parts) > 1 else 0
        return hh + mm/60.0
    try:
        return float(s)
    except Exception:
        return None


def ok_time(v) -> bool:
    return v is not None and 0 <= v < 24 * 60


def detect_vth(row: pd.Series) -> bool:
    text = " ".join(str(row.get(c, ""))
                    for c in ("POSITIONS", "LOCATIONS", "STATUS", "NOTES"))
    return any(rx.search(text or "") for rx in VTH_REGEXES)


def default_sched(row: pd.Series) -> Tuple[Optional[int], Optional[int]]:
    """
    Default schedule by Position:
      Firestoppers: 07:30–16:00 Mon–Thu, 07:30–15:00 Fri
      Painters:     07:00–15:30 Mon–Thu, 07:00–14:00 Fri
      Unknown → no trim.
    """
    pos = str(row.get("POSITIONS", "")).strip().lower()
    dow = to_date(pd.Series([row["DATE"]])).dt.weekday.iloc[0] if pd.notna(
        row.get("DATE")) else None
    if dow is None:
        return None, None
    if "firestop" in pos or "firestopp" in pos:
        start = 7*60 + 30
        end = 16*60 if dow in (0, 1, 2, 3) else (15*60 if dow == 4 else None)
        return start, end
    if "paint" in pos:
        start = 7*60
        end = 15*60 + 30 if dow in (0, 1, 2,
                                    3) else (14*60 if dow == 4 else None)
        return start, end
    return None, None


def full_day_for_dow_name(day_name: str) -> float:
    if day_name in ("Monday", "Tuesday", "Wednesday", "Thursday"):
        return 8.0
    if day_name == "Friday":
        return 7.0
    return 0.0

# ---------------- Core generator ----------------


def generate_weekly_workbook(
    export_path: str,
    output_path: Optional[str] = None,
    exclude_regex: str = DEFAULT_EXCLUDE_REGEX,
) -> str:
    # Load export
    df_raw = pd.read_excel(export_path, sheet_name=0)

    # Normalize headers
    expected = {
        "EMPLOYEE":  ["EMPLOYEE", "Employee", "Name"],
        "DATE":      ["DATE", "Date"],
        "CLOCK IN":  ["CLOCK IN", "CLOCK IN\nTIME", "Clock In", "IN", "Time In"],
        "CLOCK OUT": ["CLOCK OUT", "CLOCK OUT\nTIME", "Clock Out", "OUT", "Time Out"],
        "AUTO":      ["AUTO\nCLOCK-OUT", "Auto Clock-out", "AUTO CLOCK OUT", "Auto Clock Out", "Auto"],
        "SHIFT DUR": ["SHIFT\nDURATION", "Worked", "Duration", "Hours"],
        "STATUS":    ["STATUS", "Status"],
        "NOTES":     ["NOTES", "Notes", "Comments"],
        "POSITIONS": ["POSITIONS", "Position"],
        "LOCATIONS": ["LOCATIONS", "Location", "Site"],
    }
    col_map = {}
    for tgt, alts in expected.items():
        for a in alts:
            if a in df_raw.columns:
                col_map[a] = tgt
                break
    df = df_raw.rename(columns=col_map)
    # Ensure all expected columns exist
    for k in expected.keys():
        if k not in df.columns:
            df[k] = np.nan

    # Week window (Sat→Fri) based on latest date present
    week_start, week_end = latest_week_window(df["DATE"])
    in_week = (to_date(df["DATE"]) >= week_start) & (
        to_date(df["DATE"]) <= week_end)
    df = df[in_week].copy()

    # Derived / normalization
    df["Employee"] = df["EMPLOYEE"].astype(str).str.strip()
    df["Clock In"] = df["CLOCK IN"]
    df["Clock Out"] = df["CLOCK OUT"]
    df["Auto"] = df["AUTO"].astype(str).str.strip().str.lower().eq("yes")
    df["Worked_raw"] = df["SHIFT DUR"].apply(parse_hours)

    df["cin_min"] = df["Clock In"].apply(parse_excel_time)
    df["cout_min"] = df["Clock Out"].apply(parse_excel_time)
    df["has_clock_in"] = df["cin_min"].notna()
    df["has_clock_out"] = df["cout_min"].notna()

    text_flags = df[["STATUS", "NOTES"]].astype(
        str).agg(" ".join, axis=1).str.lower()
    df["no_clock_out_text"] = text_flags.str.contains("no clock out", na=False)
    df["valid_punch"] = df["has_clock_in"] & (
        df["has_clock_out"] | df["Auto"]) & (~df["no_clock_out_text"])

    df["Date"] = to_date(df["DATE"]).dt.date
    df["Day"] = to_date(df["DATE"]).dt.day_name()
    df["Position"] = df["POSITIONS"].astype(str)
    df["Location"] = df["LOCATIONS"].astype(str)

    # Exclusions (UHL INCLUDED; only apply your regex)
    excluded_rows = pd.DataFrame(columns=df_raw.columns)
    if exclude_regex is not None and exclude_regex != "":
        exc_re = re.compile(exclude_regex, re.I)
        mask_ex = df["Position"].str.contains(
            exc_re, na=False) | df["Location"].str.contains(exc_re, na=False)
        excluded_rows = df[mask_ex].copy()
        df = df[~mask_ex].copy()

    # Schedule-trim window
    sched = df.apply(default_sched, axis=1).tolist()
    df["SchedStart_min"] = [s[0] for s in sched]
    df["SchedEnd_min"] = [s[1] for s in sched]

    # Effective minutes within schedule (or worked_raw)
    def effective_minutes(row):
        if not row["valid_punch"]:
            return 0.0, "Invalid punch"
        ci, co, ss, se = row["cin_min"], row["cout_min"], row["SchedStart_min"], row["SchedEnd_min"]
        # If we have clock in/out, compute raw minutes
        if ok_time(ci) and ok_time(co):
            raw = co - ci
            if raw <= 0 or raw >= 24 * 60:
                return 0.0, "Invalid raw duration"
            if ok_time(ss) and ok_time(se):
                start = max(ci, int(ss))
                end = min(co, int(se))
                eff = max(0, end - start)
                return eff, "Trimmed to schedule" if eff < raw else "Within schedule"
            return raw, "No schedule"
        # Fall back to Worked_raw
        if pd.notna(row["Worked_raw"]):
            if ok_time(ss) and ok_time(se):
                sched_minutes = max(0, int(se) - int(ss))
                return min(int(round(row["Worked_raw"] * 60)), sched_minutes), "Capped by schedule duration"
            return int(round(row["Worked_raw"] * 60)), "From Worked_raw"
        return 0.0, "No times or worked"

    df[["eff_minutes", "eff_note"]] = df.apply(
        lambda r: pd.Series(effective_minutes(r)), axis=1)

    # Guardrails: 13h + no-clockout-without-auto (+ text)
    df["raw_minutes"] = ((df["cout_min"].fillna(0) - df["cin_min"].fillna(0))).where(
        df["cout_min"].notna() & df["cin_min"].notna(), np.nan
    )
    df["over13_raw"] = (df["raw_minutes"] / 60.0) > THRESH_HOURS_13
    df["over13_eff"] = (df["eff_minutes"] / 60.0) > THRESH_HOURS_13
    df["no_clockout_noauto"] = (~df["has_clock_out"]) & (~df["Auto"])
    df["guard_zero"] = df["over13_raw"] | df["over13_eff"] | df["no_clockout_noauto"] | df["no_clock_out_text"]

    # Worked (no unpaid break deduction)
    df["Break Applied (h)"] = 0.0
    df["Worked_final"] = np.where(
        df["guard_zero"], 0.0, df["eff_minutes"] / 60.0)
    df["Worked_payable"] = df["Worked_final"]  # alias

    # Leave / Training / Holiday
    df["VacTrainHoliday"] = df.apply(detect_vth, axis=1)

    # Daily policy with 24h guardrail → pay 6h (applies AFTER the 13h zero rule)
    def daily_policy_paid(row):
        day = row["Day"]
        # Leave/Training/Holiday: full weekday day
        if row["VacTrainHoliday"]:
            return full_day_for_dow_name(day)
        # Sundays excluded
        if day == "Sunday":
            return 0.0
        # Must be a valid punch to pay anything (except leave handled above)
        if not row["valid_punch"]:
            return 0.0

        w = row["Worked_payable"] if pd.notna(row["Worked_payable"]) else 0.0

        # NEW 24h guardrail → pay fixed 6h if somehow > 24
        if w > 24.0:
            return GUARD_24H_PAYS

        if day in ("Monday", "Tuesday", "Wednesday", "Thursday"):
            return min(w, 8.0)
        if day == "Friday":
            return min(w, 7.0)
        if day == "Saturday":
            return w
        return 0.0

    df["Policy Paid"] = df.apply(daily_policy_paid, axis=1).round(2)

    # ---------- Audit (plain) ----------
    audit_cols = [
        "Employee", "Date", "Day", "Position", "Location", "Clock In", "Clock Out", "Auto",
        "cin_min", "cout_min", "SchedStart_min", "SchedEnd_min",
        "Worked_raw", "eff_minutes", "Worked_final", "Worked_payable",
        "valid_punch", "VacTrainHoliday", "eff_note",
        "over13_raw", "over13_eff", "no_clockout_noauto", "guard_zero",
        "Policy Paid",
    ]
    audit = df[audit_cols].copy().sort_values(["Employee", "Date"])

    # ---------- Spaced Audit ----------
    spaced_rows = []
    prev_emp = None
    for _, r in audit.iterrows():
        if prev_emp is not None and r["Employee"] != prev_emp:
            spaced_rows.append({c: np.nan for c in audit.columns})
        spaced_rows.append(r.to_dict())
        prev_emp = r["Employee"]
    audit_spaced = pd.DataFrame(spaced_rows, columns=audit.columns)

    # ---------- Weekly totals (no OT) ----------
    holiday_by_emp = (
        audit.assign(_holiday=np.where(audit["VacTrainHoliday"], audit["Day"].map(
            full_day_for_dow_name).fillna(0.0), 0.0))
        .groupby("Employee")["_holiday"]
        .sum()
    )
    weekly = (
        audit.groupby("Employee", as_index=False)["Policy Paid"]
        .sum()
        .rename(columns={"Policy Paid": "Computed Policy Total"})
    )
    weekly = weekly.merge(holiday_by_emp.rename(
        "Holiday Hours"), on="Employee", how="left")

    flags = (
        audit.groupby("Employee")
        .agg(
            **{
                "Any Valid Punch": ("valid_punch", lambda s: bool((s == True).any())),
                "Any Vac/Train/Holiday": ("VacTrainHoliday", lambda s: bool((s == True).any())),
            }
        )
        .reset_index()
    )
    weekly = weekly.merge(flags, on="Employee", how="left")

    # No-carry unless leave present
    weekly.loc[~weekly["Any Valid Punch"] & ~
               weekly["Any Vac/Train/Holiday"], "Computed Policy Total"] = np.nan

    # Mirror Approved → Weekly Net (Approved column left blank initially)
    weekly["Approved Pay Hours"] = np.nan
    weekly["Weekly Net Total"] = weekly["Approved Pay Hours"].where(
        weekly["Approved Pay Hours"].notna(), weekly["Computed Policy Total"]
    )

    # QC tabs
    qc_zeroed = audit[audit["guard_zero"]][
        [
            "Employee", "Date", "Day", "Position", "Clock In", "Clock Out",
            "Worked_final", "Worked_payable", "Policy Paid",
            "over13_raw", "over13_eff", "no_clockout_noauto", "eff_note",
        ]
    ].sort_values(["Employee", "Date"])

    qc_trimmed = audit[audit["eff_note"].astype(str).str.contains("Trimmed", na=False)][
        [
            "Employee", "Date", "Day", "Position", "Clock In", "Clock Out",
            "Worked_final", "Worked_payable", "Policy Paid", "eff_note",
        ]
    ].sort_values(["Employee", "Date"])

    # Excluded view
    keep_cols = [c for c in ["EMPLOYEE", "DATE", "POSITIONS", "LOCATIONS", "CLOCK IN",
                             "CLOCK OUT", "SHIFT DUR", "STATUS", "NOTES"] if c in excluded_rows.columns]
    excluded_view = excluded_rows[keep_cols].copy(
    ) if not excluded_rows.empty and keep_cols else pd.DataFrame(columns=["(no exclusions)"])

    # Output
    label = f"{week_start.date():%Y-%m-%d}_to_{week_end.date():%Y-%m-%d}"
    if not output_path:
        output_path = f"WC_{label}_MASTER_13H+24H-6h_INCLUDE-UHL_NO-BREAK.xlsx"

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        weekly[
            [
                "Employee",
                "Weekly Net Total",
                "Approved Pay Hours",
                "Computed Policy Total",
                "Holiday Hours",
                "Any Valid Punch",
                "Any Vac/Train/Holiday",
            ]
        ].sort_values("Employee").to_excel(writer, sheet_name="Weekly Totals", index=False)

        audit_spaced.to_excel(writer, sheet_name="Audit", index=False)
        audit.to_excel(writer, sheet_name="Audit (No Spacing)", index=False)
        qc_trimmed.to_excel(writer, sheet_name="QC - Trimmed", index=False)
        qc_zeroed.to_excel(writer, sheet_name="QC - Zeroed", index=False)
        excluded_view.to_excel(writer, sheet_name="QC - Excluded", index=False)

        pd.DataFrame(
            {
                "Guardrails": [
                    "UHL employees are INCLUDED.",
                    f"Exclusions regex (editable): {exclude_regex or '(none)'}",
                    "Daily policy: Mon–Thu min(Worked,8), Fri min(Worked,7), Sat full, Sun 0.",
                    f"Zero if raw or trimmed > {THRESH_HOURS_13} hours.",
                    "Zero if no clock-out and Auto != Yes, or 'NO CLOCK OUT' flagged.",
                    "No unpaid break deduction.",
                    "Schedule-trim by Position (Firestoppers 07:30–16:00 / 15:00 Fri; Painters 07:00–15:30 / 14:00 Fri).",
                    "Leave/Training/Holiday = full weekday day.",
                    "No-carry if no valid punches and no leave.",
                    "Weekly Net mirrors Approved if present.",
                    f"24h guardrail: if effective worked > 24h on a day, pay {GUARD_24H_PAYS:.2f}h.",
                    "Overtime intentionally omitted.",
                ]
            }
        ).to_excel(writer, sheet_name="Guardrails", index=False)

    return output_path

# ---------------- CLI ----------------


def main():
    p = argparse.ArgumentParser(
        description="Weekly Sling payroll workbook (includes UHL) with 24h→6h guardrail.")
    p.add_argument(
        "export", help="Path to Sling report export (.xlsx or .xls).")
    p.add_argument("--out", dest="out", default=None,
                   help="Output .xlsx path (optional).")
    p.add_argument(
        "--exclude",
        dest="exclude",
        default=DEFAULT_EXCLUDE_REGEX,
        help="Regex for exclusions against Position/Location. Default: 'solid fire|administration'. Use --exclude \"\" for no exclusions.",
    )
    args = p.parse_args()
    path = generate_weekly_workbook(args.export, args.out, args.exclude)
    print(f"Created: {path}")


if __name__ == "__main__":
    main()
