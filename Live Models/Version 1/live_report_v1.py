import os
import sys
import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from fpdf import FPDF, XPos, YPos  
import config

DUTCH_TZ = "Europe/Amsterdam"

# --- CONFIGURATION ---
LIVE_DIR = "voltcast_ipi_live_v1"
REPORT_DIR = "voltcast_ipi_reports_v1"
os.makedirs(REPORT_DIR, exist_ok=True)

class VoltCastReport(FPDF):
    def header(self):
        self.set_font("helvetica", "B", 16)
        self.set_text_color(40, 70, 140)
        self.cell(0, 10, "VoltCast IPI - Daily Market Intelligence", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="L")
        
        self.set_font("helvetica", "I", 10)
        self.set_text_color(128, 128, 128)
        
        # ── FIX: Enforce Dutch Time in the PDF Header ──
        now_dutch = pd.Timestamp.now(tz="Europe/Amsterdam")
        self.cell(0, 5, f"Generated: {now_dutch.strftime('%Y-%m-%d %H:%M')} CET", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="L")
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font("helvetica", "I", 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f"Page {self.page_no()} | Confidential Business Intelligence | VoltCast", align="C")

def create_visuals(data, date_str):
    """Generates the main Price/Signal chart for the PDF."""
    df = pd.DataFrame(data['ptus'])
    df['timestamp_local'] = pd.to_datetime(df['timestamp_local'])
    
    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1]})
    
    # 1. Price Curve with Conformal Bands
    ax1.fill_between(df['timestamp_local'], df['forecast_p10'], df['forecast_p90'], color='royalblue', alpha=0.2, label='80% Confidence Band')
    ax1.plot(df['timestamp_local'], df['forecast_p50'], color='cyan', linewidth=2, label='Forecast (P50)')
    ax1.set_ylabel("EUR / MWh", fontsize=12, color='white')
    ax1.set_title(f"Day-Ahead Price Forecast: {data['delivery_date']}", fontsize=14, pad=15)
    ax1.legend(loc='upper left')
    ax1.grid(alpha=0.2)

    # 2. Signal Ribbon
    colors = []
    for sig in df['signal_strength']:
        if "Strong BUY" in sig: colors.append('green')
        elif "Moderate BUY" in sig: colors.append('lime')
        elif "Strong AVOID" in sig: colors.append('red')
        elif "Moderate AVOID" in sig: colors.append('orange')
        else: colors.append('gray')
    
    ax2.bar(df['timestamp_local'], [1]*len(df), width=0.01, color=colors)
    ax2.set_yticks([])
    ax2.set_xlabel("Local Time (Hour)", fontsize=12)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    
    plt.tight_layout()
    chart_path = os.path.join(REPORT_DIR, f"chart_{date_str}.png")
    plt.savefig(chart_path, dpi=150)
    plt.close()
    return chart_path

def sanitize_text(text):
    """Safety wrapper to convert fancy dashes/quotes and math symbols into standard ASCII."""
    if not isinstance(text, str):
        return str(text)
    
    # Replaces em-dashes (—), en-dashes (–), smart quotes, and unicode math symbols (≥, ≤)
    return text.replace("—", "-").replace("–", "-").replace("”", '"').replace("“", '"').replace("’", "'").replace("≥", ">=").replace("≤", "<=")

def generate_pdf(date_str):
    # ── THE CIRCUIT BREAKER: Check Pipeline Health First ──
    status_path = os.path.join(LIVE_DIR, "prediction_status.json")
    if os.path.exists(status_path):
        with open(status_path) as f:
            status = json.load(f)
            
        # Abort if the pipeline failed its quality gates
        if not status.get("overall_pass", False):
            print("\n[CRITICAL ABORT] Pipeline failure detected in prediction_status.json!")
            print(f"Alerts: {status.get('alerts', ['Unknown error'])}")
            print("Aborting PDF generation to prevent delivering corrupted intelligence.\n")
            from alerts import send_pipeline_alert
            send_pipeline_alert("REPORT", "Report generation aborted due to prediction pipeline failure", 
                                f"Delivery {date_str}\nAlerts: {status.get('alerts', ['Unknown error'])}")
            return
    else:
        print("Warning: prediction_status.json not found. Proceeding blindly...")

    # ── PROCEED WITH GENERATION ──
    json_path = os.path.join(LIVE_DIR, f"predictions_{date_str}.json")
    
    if not os.path.exists(json_path):
        print(f"Error: Could not find prediction file {json_path}")
        return

    with open(json_path) as f:
        data = json.load(f)['voltcast_ipi']
    
    summary = data['summary']
    chart_path = create_visuals(data, date_str)
    
    pdf = VoltCastReport()
    pdf.add_page()
    
    # --- SECTION 1: EXECUTIVE VERDICT ---
    verdict = summary['day_verdict']
    color = (0, 150, 0) if verdict == "BUY" else (200, 0, 0) if verdict == "AVOID" else (100, 100, 100)
    
    pdf.set_font("helvetica", "B", 14)
    pdf.cell(40, 10, "Daily Verdict:")
    pdf.set_text_color(*color)
    
    pdf.cell(0, 10, f"{verdict}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_text_color(0, 0, 0)
    
    pdf.set_font("helvetica", "", 10)
    pdf.multi_cell(0, 5, sanitize_text(summary['day_verdict_rationale']))
    pdf.ln(5)

    # --- SECTION 2: PRICE OUTLOOK TABLE ---
    pdf.set_font("helvetica", "B", 12)
    pdf.cell(0, 10, "Market Outlook Summary", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    
    pdf.set_font("helvetica", "", 10)
    po = summary['price_outlook']
    pdf.cell(60, 7, f"Expected Avg: EUR {po['expected_avg_eur_mwh']}/MWh", border=1)
    pdf.cell(60, 7, f"Daily Low: EUR {po['expected_low_eur_mwh']}/MWh", border=1)
    pdf.cell(60, 7, f"Daily High: EUR {po['expected_high_eur_mwh']}/MWh", border=1, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(5)

    # --- SECTION 3: THE CHART ---
    pdf.image(chart_path, x=10, y=None, w=190)
    pdf.ln(5)

    # --- SECTION 4: PROCUREMENT WINDOWS ---
    pdf.set_font("helvetica", "B", 12)
    pdf.cell(0, 10, "Optimal Dispatch Windows", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("helvetica", "", 9)
    
    # Render Buy Windows
    if summary.get('buy_windows'):
        for window in summary['buy_windows']:
            pdf.set_text_color(0, 100, 0) # Green
            pdf.cell(0, 5, f"- BUY WINDOW: {window['start_local']} to {window['end_local']} ({window['duration_hours']} hours | Avg: EUR {window['avg_forecast_eur_mwh']})", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    
    # Render Avoid Windows 
    if summary.get('avoid_windows'):
        for window in summary['avoid_windows']:
            pdf.set_text_color(200, 0, 0) # Red
            pdf.cell(0, 5, f"- AVOID WINDOW: {window['start_local']} to {window['end_local']} ({window['duration_hours']} hours | Avg: EUR {window['avg_forecast_eur_mwh']})", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    if not summary.get('buy_windows') and not summary.get('avoid_windows'):
        pdf.set_text_color(100, 100, 100)
        pdf.cell(0, 5, "- No high-conviction windows identified for this delivery date.", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf.set_text_color(0, 0, 0)
    pdf.ln(5)

    # --- SECTION 5: RISK FLAGS ---
    pdf.set_font("helvetica", "B", 12)
    pdf.cell(0, 10, "Risk Flags", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("helvetica", "", 9)
    
    rf = summary['risk_flags']
    if rf['spike_risk']:
        pdf.cell(0, 5, sanitize_text(f"- SPIKE RISK: {rf['spike_risk_note']}"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    if rf['negative_price_risk']:
        pdf.cell(0, 5, sanitize_text(f"- NEGATIVE RISK: {rf['negative_price_note']}"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(5)


    # --- SECTION 6: DISCLAIMER ---
    pdf.set_y(-30)
    pdf.set_font("helvetica", "I", 8)
    pdf.set_text_color(150, 150, 150)
    pdf.multi_cell(0, 4, sanitize_text("DISCLAIMER: " + data['disclaimer']))

    pdf_output_path = os.path.join(REPORT_DIR, f"VoltCast_Report_{date_str}.pdf")
    pdf.output(pdf_output_path)
    print(f"Report Generated: {pdf_output_path}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        target_date = sys.argv[1]
    else:
        # ── FIX: Enforce Dutch Time for the CLI fallback ──
        target_date = pd.Timestamp.now(tz="Europe/Amsterdam").strftime("%Y%m%d")
        
    generate_pdf(target_date)