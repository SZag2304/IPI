# alerts.py
import os
import smtplib
import logging
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication

log = logging.getLogger("VoltCast.Alerts")

def send_pipeline_alert(stage: str, subject: str, body: str, attachments: list = None):
    """Sends pipeline emails (failures or successful payload deliveries)."""
    # Use your environment variables (e.g. from .env)
    host = os.environ.get("VOLTCAST_SMTP_HOST", "")
    user = os.environ.get("VOLTCAST_SMTP_USER", "")
    to   = os.environ.get("VOLTCAST_ALERT_TO", "")
    
    if not host or not to:
        log.warning(f"[{stage}] No SMTP configured — alert is log-only: {subject}")
        return

    msg = MIMEMultipart()
    msg["Subject"] = f"[VoltCast {stage}] {subject}"
    msg["From"]    = user or "voltcast@localhost"
    msg["To"]      = to

    # Attach the email body text
    msg.attach(MIMEText(body, "plain"))

    # Process any attachments (PDFs, JSONs)
    if attachments:
        for filepath in attachments:
            if os.path.exists(filepath):
                with open(filepath, "rb") as f:
                    part = MIMEApplication(f.read(), Name=os.path.basename(filepath))
                part['Content-Disposition'] = f'attachment; filename="{os.path.basename(filepath)}"'
                msg.attach(part)
            else:
                log.error(f"[{stage}] Could not attach file (not found): {filepath}")

    try:
        port = int(os.environ.get("VOLTCAST_SMTP_PORT", "25"))
        passwd = os.environ.get("VOLTCAST_SMTP_PASS", "")
        with smtplib.SMTP(host, port, timeout=15) as s:
            if user and passwd:
                s.starttls()
                s.login(user, passwd)
            s.send_message(msg)
        log.info(f"[{stage}] Alert sent: {subject}")
    except Exception as e:
        log.error(f"[{stage}] Alert send failed: {e}")