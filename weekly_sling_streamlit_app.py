import io
import os
import re
from datetime import datetime
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# ---------------- UI ----------------
st.set_page_config(page_title="Weekly Sling Payroll – Streamlit", layout="wide")
st.title("Weekly Sling Payroll (Sat→Fri)")
st.caption(
    "Upload a raw Sling XLS/XLSX export. The app will apply guardrails, build Weekly Totals, "
    "and produce a multi‑sheet Excel workbook for download. Overtime is intentionally omitted."
)

with st.expander("Settings", expanded=False):
    THRESH_HOURS = st.number_input("Zero if raw/trimmed exceeds (hours)", min_value=1.0, max_value=24.0, value=13.0, step=0.5)
    BREAK_DEDUCT_HOURS = st.number_input("Unpaid break deducted (hours)", min_value=0.0, max_value=2.0, value=0.5, step=0.25)
    BREAK_MIN_SHIFT_HOURS = st.number_input("Apply break only if Worked_final > (hours)", min_value=0.0, max_value=12.0, value=6.0, step=0.5)
    EMPLOYEE_WIDE_EXCLUSION = st.checkbox("Exclude entire employee for week if any excluded row matches", value=False)
    excl_pattern_text = st.text_input("Exclude rows where Position/Location matches (regex)", r"\bUHL\b|solid\s*fire|\badministration\b")
    try:
        EXCLUDE_PATTERN = re.compile(excl_pattern_text, re.I)
        excl_ok = True
    except re.error as e:
        st.error(f"Invalid regex for exclusions: {e}")
        EXCLUDE_PATTERN = re.compile(r"$^")  # match nothing
        excl_ok = False
    DISALLOW_WORKED_RAW_FALLBACK = st.checkbox("Disallow fallback to 'SHIFT DURATION' when times are missing", value=False)


uploaded = st.file_uploader("Upload Sling export (.xls or .xlsx)", type=["xls", "xlsx"])
go = st.button("Process")

# ---------------- Helpers ----------------
def to_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce")

def latest_week_window(dates_series: pd.Series):
    dates = to_date(dates_series).dropna()
    if dates.empty:
        return None, None
    latest = dates.max().normalize()
    shift = (latest.weekday() - 5) % 7  # Sat=5
    week_start = (latest - pd.Timedelta(days=shift)).normalize()
    week_end = week_start + pd.Timedelta(days=6)
    return week_start, week_end

def parse_excel_time(val) -> Optional[int]:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    if isinstance(val, pd.Timestamp):
        return val.hour * 60 + val.minute
    if isinstance(val, (int, float)) and 0 <= val <= 1:
        mins = int(round(val * 24 * 60))
        return max(0, min(mins, 24*60 - 1))
    s = str(val).strip()
    if s == "" or s.lower() in {"nan", "none", "-"}:
        return None
    parts = s.replace(".", ":").upper()
    for fmt in ("%I:%M %p", "%I %p", "%H:%M", "%H.%M", "%H:%M:%S"):
        try:
            t = datetime.strptime(parts, fmt)
            return t.hour * 60 + t.minute
        except Exception:
            pass
    if parts.isdigit() and (3 <= len(parts) <= 4):
        hh = int(parts[:-2]); mm = int(parts[-2:])
        if 0 <= hh < 24 and 0 <= mm < 60:
            return hh * 60 + mm
    return None

def parse_hours(s) -> float:
    if pd.isna(s):
        return np.nan
    stv = str(s).replace(",", "").strip()
    if stv in {"", "-", "nan", "none", "None"}:
        return np.nan
    try:
        return float(stv)
    except Exception:
        try:
            h, m = stv.split(":")
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

VTH_REGEXES = [
    re.compile(r"\bannual\s*leave\b", re.I),
    re.compile(r"\bvacation\b", re.I),
    re.compile(r"\btraining\b", re.I),
    re.compile(r"\b(induction|course)\b", re.I),
    re.compile(r"\b(bank\s*holiday|public\s*holiday|company\s*holiday|stat(?:utory)?\s*holiday|holiday)\b", re.I),
    re.compile(r"(?<!\w)A/?L(?!\w)", re.I),
    re.compile(r"(?<!\w)B/?H(?!\w)", re.I),
    re.compile(r"(?<!\w)P/?H(?!\w)", re.I),
]

def detect_vth(row: pd.Series) -> bool:
    text = " ".join(str(row.get(c, "")) for c in ("POSITIONS", "LOCATIONS", "STATUS", "NOTES"))
    return any(rx.search(text or "") for rx in VTH_REGEXES)

def default_sched(row: pd.Series):
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

def generate_weekly_workbook_from_bytes(excel_bytes: bytes) -> tuple[bytes, dict]:
    # Read first sheet from Bytes
    df_raw = pd.read_excel(io.BytesIO(excel_bytes), sheet_name=0)
    # Normalize columns
    expected = {
        "EMPLOYEE": ["EMPLOYEE", "Employee", "Name"],
        "DATE": ["DATE", "Date"],
        "CLOCK IN": ["CLOCK IN", "CLOCK IN\\nTIME", "Clock In", "IN", "Time In"],
        "CLOCK OUT": ["CLOCK OUT", "CLOCK OUT\\nTIME", "Clock Out", "OUT", "Time Out"],
        "AUTO": ["AUTO\\nCLOCK-OUT", "Auto Clock-out", "AUTO CLOCK OUT", "Auto Clock Out", "Auto"],
        "SHIFT DUR": ["SHIFT\\nDURATION", "Worked", "Duration", "Hours"],
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
    df = df_raw.rename(columns=col_map)
    for k in expected.keys():
        if k not in df.columns:
            df[k] = np.nan

    # Week window
    week_start, week_end = latest_week_window(df["DATE"])
    if week_start is None:
        raise RuntimeError("Export has no valid dates.")
    in_week = (to_date(df["DATE"]) >= week_start) & (to_date(df["DATE"]) <= week_end)
    df = df[in_week].copy()

    # Derived fields
    df["Employee"] = df["EMPLOYEE"].astype(str).str.strip()
    df["Clock In"] = df["CLOCK IN"].astype(str)
    df["Clock Out"] = df["CLOCK OUT"].astype(str)
    df["Auto"] = df["AUTO"].astype(str).str.strip().str.lower().eq("yes")
    df["Worked_raw"] = df["SHIFT DUR"].apply(parse_hours)

    df["cin_min"] = df["CLOCK IN"].apply(parse_excel_time)
    df["cout_min"] = df["CLOCK OUT"].apply(parse_excel_time)
    df["has_clock_in"] = df["Clock In"].apply(lambda s: s.strip() not in {"", "-", "nan", "none", "None"})
    df["has_clock_out"] = df["Clock Out"].apply(lambda s: s.strip() not in {"", "-", "nan", "none", "None"})

    text_flags = df[["STATUS", "NOTES"]].astype(str).agg(" ".join, axis=1).str.lower()
    df["no_clock_out_text"] = text_flags.str.contains("no clock out", na=False)
    df["valid_punch"] = df["has_clock_in"] & (df["has_clock_out"] | df["Auto"]) & (~df["no_clock_out_text"])

    df["Date"] = to_date(df["DATE"]).dt.date
    df["Day"] = to_date(df["DATE"]).dt.day_name()
    df["Position"] = df["POSITIONS"].astype(str)
    df["Location"] = df["LOCATIONS"].astype(str)

    # Exclusions
    ex_mask = df[["Position", "Location"]].astype(str).apply(lambda c: c.str.contains(EXCLUDE_PATTERN, na=False))
    excluded_rows = df[ex_mask.any(axis=1)].copy()
    if EMPLOYEE_WIDE_EXCLUSION:
        bad_emps = set(excluded_rows["Employee"].unique())
        df = df[~df["Employee"].isin(bad_emps)].copy()
    else:
        df = df[~ex_mask.any(axis=1)].copy()

    # Default schedule windows by Position
    df["SchedStart_min"], df["SchedEnd_min"] = zip(*df.apply(default_sched, axis=1))

    # Effective minutes (schedule-trim)
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
        if (not DISALLOW_WORKED_RAW_FALLBACK) and pd.notna(row["Worked_raw"]):
            if ok_time(ss) and ok_time(se):
                sched_minutes = max(0, int(se) - int(ss))
                return min(int(round(row["Worked_raw"] * 60)), sched_minutes), "Capped by schedule duration"
            return int(round(row["Worked_raw"] * 60)), "From Worked_raw"
        return 0.0, "No times or worked"

    df[["eff_minutes", "eff_note"]] = df.apply(lambda r: pd.Series(effective_minutes(r)), axis=1)

    # Guardrails
    df["raw_minutes"] = ((df["cout_min"].fillna(0) - df["cin_min"].fillna(0))).where(
        df["cout_min"].notna() & df["cin_min"].notna(), np.nan
    )
    df["over13_raw"] = (df["raw_minutes"] / 60.0) > THRESH_HOURS
    df["over13_eff"] = (df["eff_minutes"] / 60.0) > THRESH_HOURS
    df["no_clockout_noauto"] = (~df["has_clock_out"]) & (~df["Auto"])
    df["guard_zero"] = df["over13_raw"] | df["over13_eff"] | df["no_clockout_noauto"] | df["no_clock_out_text"]

    # Worked (before break) then payable
    df["Worked_final"] = np.where(df["guard_zero"], 0.0, df["eff_minutes"] / 60.0)
    df["Break Applied (h)"] = np.where(df["Worked_final"] > BREAK_MIN_SHIFT_HOURS, BREAK_DEDUCT_HOURS, 0.0)
    df["Worked_payable"] = np.maximum(0.0, df["Worked_final"] - df["Break Applied (h)"])

    # Leave detection
    df["VacTrainHoliday"] = df.apply(detect_vth, axis=1)

    # Policy Paid per day
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

    # ---------- Build Audit ----------
    audit = df[[
        "Employee","Date","Day","Position","Location","Clock In","Clock Out","Auto",
        "Worked_raw","Worked_final","Break Applied (h)","Worked_payable","Policy Paid",
        "VacTrainHoliday","valid_punch","eff_note","guard_zero","over13_raw","over13_eff",
        "no_clockout_noauto","SchedStart_min","SchedEnd_min",
    ]].copy()
    audit["Sched Start"] = audit["SchedStart_min"].apply(mm_to_hhmm)
    audit["Sched End"] = audit["SchedEnd_min"].apply(mm_to_hhmm)
    audit.drop(columns=["SchedStart_min","SchedEnd_min"], inplace=True)
    audit = audit.sort_values(["Employee","Date","Day"]).reset_index(drop=True)

    # Spaced audit
    spaced_rows = []
    prev_emp = None
    for _, r in audit.iterrows():
        if prev_emp is not None and r["Employee"] != prev_emp:
            spaced_rows.append({c: np.nan for c in audit.columns})
        spaced_rows.append(r.to_dict())
        prev_emp = r["Employee"]
    audit_spaced = pd.DataFrame(spaced_rows, columns=audit.columns)

    # Weekly Totals
    holiday_by_emp = (
        audit.assign(_holiday=np.where(audit["VacTrainHoliday"], audit["Day"].map(full_day_for_dow).fillna(0.0), 0.0))
        .groupby("Employee")["_holiday"].sum()
    )
    weekly = (
        audit.groupby("Employee", as_index=False)["Policy Paid"].sum().rename(columns={"Policy Paid": "Computed Policy Total"})
    )
    weekly = weekly.merge(holiday_by_emp.rename("Holiday Hours"), on="Employee", how="left")
    flags = (
        audit.groupby("Employee")
        .agg(**{
            "Any Valid Punch": ("valid_punch", lambda s: bool((s == True).any())),
            "Any Vac/Train/Holiday": ("VacTrainHoliday", lambda s: bool((s == True).any())),
        })
        .reset_index()
    )
    weekly = weekly.merge(flags, on="Employee", how="left")
    weekly.loc[~weekly["Any Valid Punch"] & ~weekly["Any Vac/Train/Holiday"], "Computed Policy Total"] = np.nan
    weekly["Approved Pay Hours"] = np.nan
    # Leave Weekly Net Total blank by default (mirrors manual approval flow)
    weekly["Weekly Net Total"] = np.nan

    # QC tabs
    qc_trimmed = audit[audit["eff_note"].astype(str).str.contains("Trimmed", na=False)][
        ["Employee","Date","Day","Position","Clock In","Clock Out","Sched Start","Sched End","Worked_final","Worked_payable","Policy Paid","eff_note"]
    ]
    qc_zeroed = audit[audit["guard_zero"]][
        ["Employee","Date","Day","Position","Clock In","Clock Out","Worked_final","Worked_payable","Policy Paid","over13_raw","over13_eff","no_clockout_noauto"]
    ]
    qc_breaks = audit[audit["Break Applied (h)"] > 0][
        ["Employee","Date","Day","Position","Worked_final","Break Applied (h)","Worked_payable","Policy Paid"]
    ]
    qc_vth_nopunch = audit[(audit["VacTrainHoliday"]) & (~audit["valid_punch"])][
        ["Employee","Date","Day","Position","Policy Paid"]
    ]

    # Exclusions view
    excluded_cols = [c for c in ["EMPLOYEE","DATE","POSITIONS","LOCATIONS","CLOCK IN","CLOCK OUT","SHIFT DUR","STATUS","NOTES"] if c in df_raw.columns]
    excluded_view = pd.DataFrame(columns=excluded_cols)
    # note: we're only able to show what would have been excluded by mask selection earlier
    # build excluded_view from df_raw by recomputing mask on the in-week subset
    raw_in_week = df_raw[in_week].copy()
    raw_in_week["POSITIONS"] = raw_in_week.get("POSITIONS", np.nan)
    raw_in_week["LOCATIONS"] = raw_in_week.get("LOCATIONS", np.nan)
    ex_mask_raw = raw_in_week[["POSITIONS","LOCATIONS"]].astype(str).apply(lambda c: c.str.contains(EXCLUDE_PATTERN, na=False))
    excluded_view = raw_in_week[ex_mask_raw.any(axis=1)][excluded_cols].copy()

    # Output buffer
    label = f"{week_start.date():%Y-%m-%d}_to_{week_end.date():%Y-%m-%d}"
    out_name = f"WC_{label}_MASTER_13H_EXCL-UHL-SOLIDFIRE-ADMIN_BREAK30gt6_NO-OT.xlsx"
    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        weekly[[
            "Employee","Weekly Net Total","Approved Pay Hours","Computed Policy Total",
            "Holiday Hours","Any Valid Punch","Any Vac/Train/Holiday"
        ]].sort_values("Employee").to_excel(writer, sheet_name="Weekly Totals", index=False)

        audit_spaced.to_excel(writer, sheet_name="Audit", index=False)
        audit.to_excel(writer, sheet_name="Audit (No Spacing)", index=False)
        qc_trimmed.to_excel(writer, sheet_name="QC - Trimmed", index=False)
        qc_zeroed.to_excel(writer, sheet_name="QC - Zeroed", index=False)
        qc_breaks.to_excel(writer, sheet_name="QC - Breaks", index=False)
        qc_vth_nopunch.to_excel(writer, sheet_name="QC - Leave no punches", index=False)
        excluded_view.to_excel(writer, sheet_name="QC - Excluded UHL+SF+Admin", index=False)

        pd.DataFrame({
            "Guardrails": [
                "EXCLUSIONS: Remove rows where Position/Location contains 'UHL', 'Solid Fire', or 'Administration'.",
                f"ZERO if daily raw or trimmed > {THRESH_HOURS} hours.",
                "ZERO if no clock-out and Auto != Yes, or 'NO CLOCK OUT' flagged.",
                "Schedule-trim by Position defaults (Firestoppers: 07:30–16:00 Mon–Thu / 15:00 Fri; Painters: 07:00–15:30 Mon–Thu / 14:00 Fri).",
                "Daily policy: Mon–Thu min(Worked,8), Fri min(Worked,7), Sat full, Sun 0.",
                f"UNPAID BREAK: deduct {BREAK_DEDUCT_HOURS}h when Worked_final > {BREAK_MIN_SHIFT_HOURS}h.",
                "Leave/Training/Holiday: pay full weekday day (Mon–Thu 8h, Fri 7h).",
                "No-carry: blank weekly total if no valid punches and no leave.",
                "Weekly Net mirrors Approved if present; else Computed Policy Total.",
                "Overtime is intentionally omitted from outputs; calculate separately.",
            ]
        }).to_excel(writer, sheet_name="Guardrails", index=False)
    bio.seek(0)
    return bio.read(), {
        "weekly": weekly,
        "audit": audit,
        "label": label,
        "out_name": out_name,
    }

# ---------------- Run ----------------
if go:
    if not uploaded:
        st.warning("Please upload an Excel file first.")
    elif not excl_ok:
        st.warning("Fix the exclusion regex before running.")
    else:
        try:
            out_bytes, meta = generate_weekly_workbook_from_bytes(uploaded.read())
            st.success(f"Processed week: {meta['label']}")
            # Preview Weekly Totals
            st.subheader("Preview – Weekly Totals")
            st.dataframe(meta["weekly"].sort_values("Employee"), use_container_width=True)
            st.download_button(
                label="⬇️ Download Excel Workbook",
                data=out_bytes,
                file_name=meta["out_name"],
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

            # Optional: compare against a reference workbook to validate matching
            st.subheader("Optional: Compare against a reference workbook")
            ref = st.file_uploader("Upload an expected/reference workbook (xlsx) to compare 'Weekly Totals'", type=["xlsx"], key="ref")
            if ref is not None:
                try:
                    exp_weekly = pd.read_excel(ref, sheet_name="Weekly Totals")
                    act_weekly = meta["weekly"].copy()
                    # Normalise names
                    exp_weekly["Employee"] = exp_weekly["Employee"].astype(str).str.strip()
                    act_weekly["Employee"] = act_weekly["Employee"].astype(str).str.strip()
                    # Pick common columns
                    common_cols = [c for c in ["Weekly Net Total","Approved Pay Hours","Computed Policy Total","Holiday Hours"] if c in exp_weekly.columns and c in act_weekly.columns]
                    merged = exp_weekly[["Employee"] + common_cols].merge(
                        act_weekly[["Employee"] + common_cols], on="Employee", how="outer", suffixes=("_exp","_act")
                    )
                    for c in common_cols:
                        merged[f"{c} diff"] = merged[f"{c}_act"] - merged[f"{c}_exp"]
                    # Filter mismatches
                    def row_mismatch(row):
                        for c in common_cols:
                            a = row.get(f"{c} (act)", row.get(f"{c}_act"))
                            e = row.get(f"{c} (exp)", row.get(f"{c}_exp"))
                            if pd.isna(a) != pd.isna(e):
                                return True
                            if pd.notna(a) and pd.notna(e) and abs(a - e) > 1e-6:
                                return True
                        return False
                    mask = merged.apply(lambda r: row_mismatch(r), axis=1)
                    st.write(f"Employees compared: {len(merged)}")
                    st.write(f"Rows with any mismatch: {int(mask.sum())}")
                    st.dataframe(merged[mask], use_container_width=True)
                except Exception as ee:
                    st.error(f"Comparison failed: {ee}")

        except Exception as e:
            st.error(f"Failed to process file: {e}")
else:
    st.info("Upload an XLS/XLSX and click **Process**.")
