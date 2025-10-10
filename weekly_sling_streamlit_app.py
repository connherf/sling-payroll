# app.py
# Streamlit app: Generate Weekly Sling Payroll Workbook with guardrails

import io
import re
from datetime import datetime
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# ---------------- Defaults (matches your latest guardrails) ----------------
THRESH_HOURS_DEFAULT = 13.0
DEFAULT_EXCLUDE_REGEX = r"uhl|solid\s*fire|\badministration\b"   # EXCLUSIONS
DEFAULT_UNPAID_BREAK_HOURS = 0.5
DEFAULT_UNPAID_BREAK_THRESHOLD = 6.0  # deduct when Worked_final > 6h


# ---------------- Helper functions ----------------
def to_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce")


def latest_week_window(dates_series: pd.Series) -> Tuple[pd.Timestamp, pd.Timestamp]:
    dates = to_date(dates_series).dropna()
    if dates.empty:
        raise RuntimeError("Export has no valid dates.")
    latest = dates.max().normalize()
    # Saturday anchor (Mon=0..Sun=6 -> Sat=5)
    shift = (latest.weekday() - 5) % 7
    week_start = (latest - pd.Timedelta(days=shift)).normalize()
    week_end = week_start + pd.Timedelta(days=6)
    return week_start, week_end


def parse_excel_time(val) -> Optional[int]:
    """Return minutes since midnight (0..1439) or None."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    if isinstance(val, pd.Timestamp):
        return val.hour * 60 + val.minute
    if isinstance(val, (int, float)) and 0 <= val <= 1:
        mins = int(round(val * 24 * 60))
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
        if 0 <= hh < 24 and 0 <= mm < 60:
            return hh * 60 + mm
    return None


def parse_hours(s) -> float:
    if pd.isna(s):
        return np.nan
    st = str(s).replace(",", "").strip()
    if st in {"", "-", "nan", "none", "None"}:
        return np.nan
    try:
        return float(st)
    except Exception:
        try:
            h, m = st.split(":")
            return float(h) + float(m) / 60.0
        except Exception:
            return np.nan


def full_day_for_dow(dow: str) -> float:
    if dow in ("Monday", "Tuesday", "Wednesday", "Thursday"):
        return 8.0
    if dow == "Friday":
        return 7.0
    return 0.0


def ok_time(x) -> bool:
    return (x is not None) and not (isinstance(x, float) and np.isnan(x))


# Leave / training / holiday detection
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


def detect_vth(row: pd.Series) -> bool:
    text = " ".join(str(row.get(c, ""))
                    for c in ("POSITIONS", "LOCATIONS", "STATUS", "NOTES"))
    return any(rx.search(text or "") for rx in VTH_REGEXES)


def default_sched(row: pd.Series) -> Tuple[Optional[int], Optional[int]]:
    """
    Position-based defaults:
      Firestoppers: 07:30–16:00 Mon–Thu, 07:30–15:00 Fri
      Painters:     07:00–15:30 Mon–Thu, 07:00–14:00 Fri
    """
    dow = row["Day"]
    pos = (row.get("Position") or "").lower()
    if "firestop" in pos or "fire stopper" in pos:
        start = 7 * 60 + 30
        if dow in ("Monday", "Tuesday", "Wednesday", "Thursday"):
            end = 16 * 60
        elif dow == "Friday":
            end = 15 * 60
        else:
            return (None, None)
        return (start, end)
    if "painter" in pos:
        start = 7 * 60
        if dow in ("Monday", "Tuesday", "Wednesday", "Thursday"):
            end = 15 * 60 + 30
        elif dow == "Friday":
            end = 14 * 60
        else:
            return (None, None)
        return (start, end)
    return (None, None)


def mm_to_hhmm(m: Optional[int]) -> str:
    if not ok_time(m):
        return ""
    m = int(m)
    return f"{m // 60:02d}:{m % 60:02d}"


# ---------------- Core generator ----------------
def generate_weekly_workbook(
    df_raw: pd.DataFrame,
    exclude_regex: str = DEFAULT_EXCLUDE_REGEX,
    thresh_hours: float = THRESH_HOURS_DEFAULT,
    unpaid_break_hours: float = DEFAULT_UNPAID_BREAK_HOURS,
    unpaid_break_threshold: float = DEFAULT_UNPAID_BREAK_THRESHOLD,
) -> Tuple[bytes, pd.DataFrame, Tuple[pd.Timestamp, pd.Timestamp]]:
    """
    Returns (excel_bytes, weekly_totals_df, (week_start, week_end))
    """

    # Normalize headers to canonical set used in pipeline
    expected = {
        "EMPLOYEE": ["EMPLOYEE", "Employee", "Name"],
        "DATE": ["DATE", "Date"],
        "CLOCK IN": ["CLOCK IN", "CLOCK IN\nTIME", "Clock In", "IN", "Time In"],
        "CLOCK OUT": ["CLOCK OUT", "CLOCK OUT\nTIME", "Clock Out", "OUT", "Time Out"],
        "AUTO": ["AUTO\nCLOCK-OUT", "Auto Clock-out", "AUTO CLOCK OUT", "Auto Clock Out", "Auto"],
        "SHIFT DUR": ["SHIFT\nDURATION", "Worked", "Duration", "Hours"],
        "STATUS": ["STATUS", "Status"],
        "NOTES": ["NOTES", "Notes", "Comments"],
        "POSITIONS": ["POSITIONS", "Position"],
        "LOCATIONS": ["LOCATIONS", "Location", "Site"],
    }
    col_map = {}
    for tgt, alts in expected.items():
        for a in alts:
            if a in df_raw.columns:
                col_map[a] = tgt
                break
    df = df_raw.rename(columns=col_map).copy()
    for k in expected.keys():
        if k not in df.columns:
            df[k] = np.nan

    # Week window filter
    week_start, week_end = latest_week_window(df["DATE"])
    in_week = (to_date(df["DATE"]) >= week_start) & (
        to_date(df["DATE"]) <= week_end)
    df = df[in_week].copy()

    # Derived columns
    df["Employee"] = df["EMPLOYEE"].astype(str).str.strip()
    df["Clock In"] = df["CLOCK IN"].astype(str)
    df["Clock Out"] = df["CLOCK OUT"].astype(str)
    df["Auto"] = df["AUTO"].astype(str).str.strip().str.lower().eq("yes")
    df["Worked_raw"] = df["SHIFT DUR"].apply(parse_hours)

    df["cin_min"] = df["CLOCK IN"].apply(parse_excel_time)
    df["cout_min"] = df["CLOCK OUT"].apply(parse_excel_time)
    df["has_clock_in"] = df["Clock In"].apply(lambda s: str(
        s).strip() not in {"", "-", "nan", "none", "None"})
    df["has_clock_out"] = df["Clock Out"].apply(lambda s: str(s).strip() not in {
                                                "", "-", "nan", "none", "None"})

    text_flags = df[["STATUS", "NOTES"]].astype(
        str).agg(" ".join, axis=1).str.lower()
    df["no_clock_out_text"] = text_flags.str.contains("no clock out", na=False)
    df["valid_punch"] = df["has_clock_in"] & (
        df["has_clock_out"] | df["Auto"]) & (~df["no_clock_out_text"])

    df["Date"] = to_date(df["DATE"]).dt.date
    df["Day"] = to_date(df["DATE"]).dt.day_name()
    df["Position"] = df["POSITIONS"].astype(str)
    df["Location"] = df["LOCATIONS"].astype(str)

    # Exclusions (Position/Location contains pattern)
    exc_re = re.compile(exclude_regex, re.I) if exclude_regex else None
    if exc_re:
        mask_ex = df["Position"].str.contains(
            exc_re, na=False) | df["Location"].str.contains(exc_re, na=False)
        excluded_rows = df[mask_ex].copy()
        df = df[~mask_ex].copy()
    else:
        excluded_rows = df.iloc[0:0].copy()

    # Schedule-trim
    df["SchedStart_min"], df["SchedEnd_min"] = zip(
        *df.apply(default_sched, axis=1))

    def effective_minutes(row):
        if not row["valid_punch"]:
            return 0.0, "Invalid punch"
        ci, co, ss, se = row["cin_min"], row["cout_min"], row["SchedStart_min"], row["SchedEnd_min"]
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
        if pd.notna(row["Worked_raw"]):
            if ok_time(ss) and ok_time(se):
                sched_minutes = max(0, int(se) - int(ss))
                return min(int(round(row["Worked_raw"] * 60)), sched_minutes), "Capped by schedule duration"
            return int(round(row["Worked_raw"] * 60)), "From Worked_raw"
        return 0.0, "No times or worked"

    df[["eff_minutes", "eff_note"]] = df.apply(
        lambda r: pd.Series(effective_minutes(r)), axis=1)

    # Guardrails: 13h and no-clockout-without-auto
    df["raw_minutes"] = ((df["cout_min"].fillna(0) - df["cin_min"].fillna(0))).where(
        df["cout_min"].notna() & df["cin_min"].notna(), np.nan
    )
    df["over13_raw"] = (df["raw_minutes"] / 60.0) > float(thresh_hours)
    df["over13_eff"] = (df["eff_minutes"] / 60.0) > float(thresh_hours)
    df["no_clockout_noauto"] = (~df["has_clock_out"]) & (~df["Auto"])
    df["guard_zero"] = df["over13_raw"] | df["over13_eff"] | df["no_clockout_noauto"] | df["no_clock_out_text"]

    # Worked (before unpaid break)
    df["Worked_final"] = np.where(
        df["guard_zero"], 0.0, df["eff_minutes"] / 60.0)

    # Leave / Training / Holiday
    df["VacTrainHoliday"] = df.apply(detect_vth, axis=1)

    # UNPAID BREAK: deduct (strict > threshold), before caps, not on leave/training/holiday
    if unpaid_break_hours and unpaid_break_hours > 0:
        df["Break Applied (h)"] = np.where(
            (df["Worked_final"] > float(unpaid_break_threshold)) & (
                ~df["VacTrainHoliday"]),
            float(unpaid_break_hours),
            0.0,
        )
    else:
        df["Break Applied (h)"] = 0.0

    df["Worked_payable"] = np.maximum(
        0.0, df["Worked_final"] - df["Break Applied (h)"])

    # Daily policy caps (after break applied)
    def policy_paid(row):
        dow = row["Day"]
        if row["VacTrainHoliday"]:
            return full_day_for_dow(dow)
        if dow == "Sunday":
            return 0.0
        if not row["valid_punch"]:
            return 0.0
        w = row["Worked_payable"] if pd.notna(row["Worked_payable"]) else 0.0
        if dow in ("Monday", "Tuesday", "Wednesday", "Thursday"):
            return min(w, 8.0)
        if dow == "Friday":
            return min(w, 7.0)
        if dow == "Saturday":
            return w
        return 0.0

    df["Policy Paid"] = df.apply(policy_paid, axis=1)

    # ---------- Audit ----------
    audit = df[
        [
            "Employee",
            "Date",
            "Day",
            "Position",
            "Location",
            "Clock In",
            "Clock Out",
            "Auto",
            "Worked_raw",
            "Worked_final",
            "Break Applied (h)",
            "Worked_payable",
            "Policy Paid",
            "VacTrainHoliday",
            "valid_punch",
            "eff_note",
            "guard_zero",
            "over13_raw",
            "over13_eff",
            "no_clockout_noauto",
            "SchedStart_min",
            "SchedEnd_min",
        ]
    ].copy()
    audit["Sched Start"] = audit["SchedStart_min"].apply(mm_to_hhmm)
    audit["Sched End"] = audit["SchedEnd_min"].apply(mm_to_hhmm)
    audit.drop(columns=["SchedStart_min", "SchedEnd_min"], inplace=True)
    audit = audit.sort_values(
        ["Employee", "Date", "Day"]).reset_index(drop=True)

    # Spaced Audit (blank row between employees)
    spaced = []
    prev_emp = None
    for _, r in audit.iterrows():
        if prev_emp is not None and r["Employee"] != prev_emp:
            spaced.append({c: np.nan for c in audit.columns})
        spaced.append(r.to_dict())
        prev_emp = r["Employee"]
    audit_spaced = pd.DataFrame(spaced, columns=audit.columns)

    # ---------- Weekly totals ----------
    holiday_by_emp = (
        audit.assign(
            _holiday=np.where(audit["VacTrainHoliday"], audit["Day"].map(
                full_day_for_dow).fillna(0.0), 0.0)
        )
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
    weekly["Approved Pay Hours"] = np.nan
    weekly["Weekly Net Total"] = weekly["Approved Pay Hours"].where(
        weekly["Approved Pay Hours"].notna(), weekly["Computed Policy Total"]
    )

    # QC tabs
    qc_zeroed = audit[audit["guard_zero"]][
        [
            "Employee",
            "Date",
            "Day",
            "Position",
            "Clock In",
            "Clock Out",
            "Worked_final",
            "Worked_payable",
            "Policy Paid",
            "over13_raw",
            "over13_eff",
            "no_clockout_noauto",
            "eff_note",
        ]
    ]
    qc_trimmed = audit[audit["eff_note"].astype(str).str.contains("Trimmed", na=False)][
        [
            "Employee",
            "Date",
            "Day",
            "Position",
            "Clock In",
            "Clock Out",
            "Worked_final",
            "Worked_payable",
            "Policy Paid",
            "eff_note",
        ]
    ]

    # Guardrails sheet (verbatim)
    guardrails_text = [
        "EXCLUSIONS: Remove rows where Position/Location contains 'UHL', 'Solid Fire', or 'Administration'.",
        "ZERO if daily raw or trimmed > 13 hours.",
        "ZERO if no clock-out and Auto != Yes, or 'NO CLOCK OUT' flagged.",
        "Schedule-trim by Position defaults (Firestoppers: 07:30–16:00 Mon–Thu / 15:00 Fri; Painters: 07:00–15:30 Mon–Thu / 14:00 Fri).",
        "Daily policy: Mon–Thu min(Worked,8), Fri min(Worked,7), Sat full, Sun 0.",
        "UNPAID BREAK: deduct 0.5h when Worked_final > 6h (strictly over 6h; before caps; not on leave/training/holiday).",
        "Leave/Training/Holiday: pay full weekday day (Mon–Thu 8h, Fri 7h).",
        "No-carry: blank weekly total if no valid punches and no leave.",
        "Weekly Net mirrors Approved if present; else Computed Policy Total.",
        "Overtime is intentionally omitted from outputs; calculate separately if needed.",
    ]

    # Build workbook in memory
    buffer = io.BytesIO()
    wk_label = f"{week_start.date():%Y-%m-%d}_to_{week_end.date():%Y-%m-%d}"
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
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
        # Excluded view: show raw if present
        keep_cols = [c for c in ["EMPLOYEE", "DATE", "POSITIONS", "LOCATIONS",
                                 "CLOCK IN", "CLOCK OUT", "SHIFT DUR", "STATUS", "NOTES"] if c in df_raw.columns]
        excluded_view = df_raw[keep_cols].copy(
        ) if keep_cols else df_raw.copy()
        # Only keep actual excluded rows if we computed them
        if exc_re is not None and not excluded_rows.empty:
            excluded_view = excluded_rows[keep_cols].copy(
            ) if keep_cols else excluded_rows.copy()
        excluded_view.to_excel(writer, sheet_name="QC - Excluded", index=False)
        pd.DataFrame({"Guardrails": guardrails_text}).to_excel(
            writer, sheet_name="Guardrails", index=False)

    buffer.seek(0)
    return buffer.read(), weekly.sort_values("Employee"), (week_start, week_end)


# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="Weekly Sling Payroll Generator", layout="wide")
st.title("Weekly Sling Payroll Generator")

with st.sidebar:
    st.header("Options")
    exclude_regex = st.text_input("Exclusion regex (Position/Location)", value=DEFAULT_EXCLUDE_REGEX,
                                  help="Rows matching this regex in Position OR Location will be removed.")
    thresh_hours = st.number_input(
        "Daily cap (hours): ZERO if raw/trimmed > cap", value=THRESH_HOURS_DEFAULT, step=0.5, min_value=1.0)
    use_unpaid_break = st.checkbox(
        "Apply unpaid break when Worked_final > 6h", value=True)
    unpaid_break_hours = DEFAULT_UNPAID_BREAK_HOURS if use_unpaid_break else 0.0
    if use_unpaid_break:
        unpaid_break_hours = st.number_input(
            "Unpaid break (hours)", value=DEFAULT_UNPAID_BREAK_HOURS, step=0.25, min_value=0.0)
        unpaid_break_threshold = st.number_input(
            "Unpaid break threshold (hours, strict >)", value=DEFAULT_UNPAID_BREAK_THRESHOLD, step=0.25, min_value=0.0)
    else:
        unpaid_break_threshold = DEFAULT_UNPAID_BREAK_THRESHOLD

uploaded = st.file_uploader(
    "Upload Sling export (.xls or .xlsx)", type=["xls", "xlsx"])

if uploaded is not None:
    try:
        # Use engine fallback so .xls works (needs xlrd)
        df_raw = pd.read_excel(uploaded, sheet_name=0)
        excel_bytes, weekly_df, (week_start, week_end) = generate_weekly_workbook(
            df_raw=df_raw,
            exclude_regex=exclude_regex,
            thresh_hours=thresh_hours,
            unpaid_break_hours=unpaid_break_hours,
            unpaid_break_threshold=unpaid_break_threshold,
        )

        st.success(
            f"Workbook created for week {week_start.date():%Y-%m-%d} to {week_end.date():%Y-%m-%d}")
        st.download_button(
            label="⬇️ Download Weekly Workbook (.xlsx)",
            data=excel_bytes,
            file_name=f"WC_{week_start.date():%Y-%m-%d}_to_{week_end.date():%Y-%m-%d}_MASTER.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

        st.subheader("Weekly Totals Preview")
        st.dataframe(weekly_df, use_container_width=True)

        with st.expander("Show guardrails applied"):
            st.markdown(
                """
**Guardrails**

- **EXCLUSIONS:** Remove rows where Position/Location contains 'UHL', 'Solid Fire', or 'Administration'.  
- **ZERO** if daily raw or trimmed > 13 hours.  
- **ZERO** if no clock-out and Auto != Yes, or 'NO CLOCK OUT' flagged.  
- **Schedule-trim** by Position defaults (Firestoppers: 07:30–16:00 Mon–Thu / 15:00 Fri; Painters: 07:00–15:30 Mon–Thu / 14:00 Fri).  
- **Daily policy:** Mon–Thu min(Worked,8), Fri min(Worked,7), Sat full, Sun 0.  
- **UNPAID BREAK:** deduct 0.5h when Worked_final > 6h (strictly over 6h; before caps; not on leave/training/holiday).  
- **Leave/Training/Holiday:** pay full weekday day (Mon–Thu 8h, Fri 7h).  
- **No-carry:** blank weekly total if no valid punches and no leave.  
- **Weekly Net** mirrors Approved if present; else Computed Policy Total.  
- **Overtime** intentionally omitted.
                """
            )
    except Exception as e:
        st.error(f"Failed to process file: {e}")
else:
    st.info("Upload a Sling export to begin.")
