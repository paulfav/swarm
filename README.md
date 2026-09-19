# China Access

Hard-goods factory sourcing with generative deal rooms, live China marketplace search, multi-channel supplier outreach, and reply retranscription.

## Product rules

- Hard goods only · no sampling · client never chats with the factory
- Agent sources + talks to suppliers; deal room shows retranscription

## Live loop

1. **Source** — Made-in-China + AliExpress (Alibaba often CAPTCHA-blocked)
2. **Contact** — Made-in-China inquiry (live) + WhatsApp (Twilio if configured) + WeChat/WeCom (if configured)
3. **Replies** — paste into Supplier thread, poll IMAP inbox, or email/WhatsApp webhooks → parse FOB/MOQ/lead time → refresh landed quote

## Configure (optional)

```bash
# Agent identity (MIC inquiry “from”)
export CHINA_ACCESS_AGENT_EMAIL=you@example.com
export CHINA_ACCESS_AGENT_NAME="China Access Agent"
export CHINA_ACCESS_AGENT_COMPANY="China Access"
export CHINA_ACCESS_AGENT_MOBILE=5550100123

# Inbound email (IMAP poll)
export CHINA_ACCESS_IMAP_HOST=imap.gmail.com
export CHINA_ACCESS_IMAP_USER=you@example.com
export CHINA_ACCESS_IMAP_PASS=app-password

# Email webhook secret
export CHINA_ACCESS_WEBHOOK_SECRET=...

# WhatsApp via Twilio
export TWILIO_ACCOUNT_SID=...
export TWILIO_AUTH_TOKEN=...
export TWILIO_WHATSAPP_FROM=whatsapp:+14155238886

# WeChat Work (WeCom)
export WECOM_CORP_ID=...
export WECOM_SECRET=...
export WECOM_AGENT_ID=...
```

## Run

```bash
npm install
npx playwright install chromium
npm run dev
```

## API

- `POST /api/inquiries` — create + live source
- `POST /api/deals/:id/contact` — MIC + WhatsApp/WeChat outreach
- `POST /api/deals/:id/replies` — `{ text, channel? }` ingest supplier reply
- `POST /api/deals/:id/poll-replies` — IMAP poll + ingest
- `POST /api/webhooks/email` — inbound email JSON
- `POST /api/webhooks/whatsapp` — Twilio WhatsApp inbound
