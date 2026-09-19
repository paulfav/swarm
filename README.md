# China Access

Hard-goods factory sourcing with generative deal rooms, live China marketplace search, multi-channel supplier outreach, and reply retranscription.

## Product rules

- Hard goods only · no sampling · client never chats with the factory
- Agent sources + talks to suppliers; deal room shows retranscription

## Live loop

1. **Source** — Made-in-China + AliExpress (Alibaba often CAPTCHA-blocked)
2. **Contact** — Made-in-China inquiry (live) + WhatsApp (Green-API self-provisioned, Twilio optional) + WeChat/WeCom when available
3. **Replies** — paste into Supplier thread, poll mail.tm / IMAP / Green-API, or email/WhatsApp webhooks → parse FOB/MOQ/lead time → refresh landed quote

## Self-setup (agent does this)

```bash
npm run inbox:setup      # provisions mail.tm inbox → .env.local
npm run whatsapp:setup   # provisions free Green-API WA instance → .env.local
npm run dev              # both run automatically before next dev
```

WhatsApp still needs **one** Linked-Devices QR scan (phone). Open `GET /api/inbox/status` — when `whatsapp.state` is `notAuthorized`, `whatsapp.qr.base64` is a PNG to scan.

WeChat personal accounts have no public send API. WeCom requires a verified Chinese company corp account (cannot be created from this cloud environment).

## Optional overrides

```bash
# Agent identity (MIC inquiry “from”) — defaults come from mail.tm setup
export CHINA_ACCESS_AGENT_EMAIL=you@example.com
export CHINA_ACCESS_AGENT_NAME="China Access Agent"
export CHINA_ACCESS_AGENT_COMPANY="China Access"
export CHINA_ACCESS_AGENT_MOBILE=5550100123

# IMAP fallback (if not using mail.tm)
export CHINA_ACCESS_IMAP_HOST=imap.gmail.com
export CHINA_ACCESS_IMAP_USER=you@example.com
export CHINA_ACCESS_IMAP_PASS=app-password

# Twilio WhatsApp (optional fallback)
export TWILIO_ACCOUNT_SID=...
export TWILIO_AUTH_TOKEN=...
export TWILIO_WHATSAPP_FROM=whatsapp:+14155238886

# WeChat Work (WeCom) — needs China entity
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
- `POST /api/deals/:id/poll-replies` — mail.tm / IMAP / Green-API poll + ingest
- `GET /api/inbox/status` — inbox + WhatsApp auth/QR status
- `POST /api/webhooks/email` — inbound email JSON
- `POST /api/webhooks/whatsapp` — Twilio or Green-API inbound
