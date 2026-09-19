# China Access

Hard-goods factory sourcing with **generative deal rooms** and **live China marketplace search**.

## Product rules (v1)

- Hard goods only
- No sampling loop
- No direct client ↔ factory chat
- **Live scrape** of Made-in-China + AliExpress on each inquiry (Alibaba attempted; often CAPTCHA-blocked from cloud IPs)
- Deal room retranscribes sourcing + (later) factory negotiation
- Each product gets a tailored UI blueprint

## Live sourcing

`src/lib/china-source.ts` launches Playwright and queries:

1. **Made-in-China.com** — factory storefront listings (primary)
2. **AliExpress** — export retail price signal
3. **Alibaba.com** — attempted; datacenter IPs usually hit CAPTCHA

Results are attached to the deal (`sourcing.listings`) and rendered in the deal room.

WhatsApp/WeChat bridge is still future work. **Made-in-China inquiry outreach is live**: the deal room **Contact supplier** button sends a real inquiry to the factory contact (e.g. Ms. He) via MIC's form.

Configure agent identity (optional):

```bash
export CHINA_ACCESS_AGENT_EMAIL=you@example.com
export CHINA_ACCESS_AGENT_NAME="China Access Agent"
export CHINA_ACCESS_AGENT_COMPANY="China Access"
export CHINA_ACCESS_AGENT_MOBILE=5550100123
```

## API

- `POST /api/deals/:id/contact` — send real Made-in-China inquiry to top MIC listing
- `POST /api/deals/:id/actions` — `{ action: "approve" | "request_change" | "reject", note? }`
- `POST /api/inquiries` — create inquiry → live source + generative deal room
