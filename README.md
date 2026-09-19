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

WhatsApp/WeChat factory chat bridge is **not** connected yet — timeline is honest about that.

## Stack

- Next.js 15 (App Router) + TypeScript + Tailwind CSS v4
- Playwright Chromium for marketplace browse
- In-memory deal store

## Run

```bash
npm install
npx playwright install chromium
npm run dev
```

Opening a deal takes ~15–40s while marketplaces are scraped.

Demo deals: `/deals/deal_sofa_demo`, `/deals/deal_lamp_demo`, `/deals/deal_table_demo`

## API

- `GET /api/inquiries` — list deals
- `POST /api/inquiries` — create inquiry → live source + generative deal room
- `GET /api/deals/:id` — fetch deal
- `POST /api/deals/:id/actions` — `{ action: "approve" | "request_change" | "reject", note? }`
