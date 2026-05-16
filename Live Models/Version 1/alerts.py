# alerts.py
import os
import smtplib
import logging
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger("VoltCast.Alerts")

def send_pipeline_alert(stage: str, subject: str, body: str, attachments: list = None):
    """Sends pipeline emails with role-based routing (Owner vs Clients)."""
    host = os.environ.get("VOLTCAST_SMTP_HOST", "")
    user = os.environ.get("VOLTCAST_SMTP_USER", "")
    
    # Fetch roles from .env (fallback to old variable name just in case)
    owner_email = os.environ.get("VOLTCAST_OWNER_EMAIL") or os.environ.get("VOLTCAST_ALERT_TO", "")
    client_emails_str = os.environ.get("VOLTCAST_CLIENT_EMAILS", "")
    
    if not host or not owner_email:
        log.warning(f"[{stage}] No SMTP or Owner email configured — alert is log-only: {subject}")
        return

    # ── ROUTING LOGIC ──
    # If this is the final report, send to the Owner AND all Clients
    if stage == "DELIVERY":
        recipients = [owner_email]
        if client_emails_str:
            # Clean up the comma-separated list and add to recipients
            recipients.extend([e.strip() for e in client_emails_str.split(",") if e.strip()])
    # Otherwise (Fetch, Feature, Model, Errors), send ONLY to Owner
    else:
        recipients = [owner_email]

    msg = MIMEMultipart()
    msg["Subject"] = f"[VoltCast {stage}] {subject}"
    msg["From"]    = user or "voltcast@localhost"
    # smtplib expects multiple recipients as a comma-separated string in the "To" header
    msg["To"]      = ", ".join(recipients) 

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
        log.info(f"[{stage}] Alert sent to {len(recipients)} recipient(s): {subject}")
    except Exception as e:
        log.error(f"[{stage}] Alert send failed: {e}")