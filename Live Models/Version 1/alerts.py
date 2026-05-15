# alerts.py — new file
import os, smtplib, logging
from email.mime.text import MIMEText
log = logging.getLogger("VoltCast.Alerts")

def send_pipeline_alert(stage: str, subject: str, body: str):
    host = os.environ.get("VOLTCAST_SMTP_HOST", "")
    user = os.environ.get("VOLTCAST_SMTP_USER", "")
    to   = os.environ.get("VOLTCAST_ALERT_TO", "")
    if not host or not to:
        log.warning(f"[{stage}] No SMTP configured — alert is log-only: {subject}")
        return
    msg = MIMEText(body)
    msg["Subject"] = f"[VoltCast {stage}] {subject}"
    msg["From"]    = user or "voltcast@localhost"
    msg["To"]      = to
    try:
        port = int(os.environ.get("VOLTCAST_SMTP_PORT", "25"))
        passwd = os.environ.get("VOLTCAST_SMTP_PASS", "")
        with smtplib.SMTP(host, port, timeout=15) as s:
            if user and passwd:
                s.starttls(); s.login(user, passwd)
            s.send_message(msg)
        log.info(f"[{stage}] Alert sent: {subject}")
    except Exception as e:
        log.error(f"[{stage}] Alert send failed: {e}")