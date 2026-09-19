# China Access

Hard-goods factory sourcing with **generative deal rooms**.

## Product rules (v1)

- Hard goods only
- No sampling loop
- No direct client ↔ factory chat
- An AI agent negotiates with Chinese producers in the background
- The client sees a **visual deal room** that retranscribes that negotiation
- Each product gets a **tailored UI blueprint** coded on the spot (sofa atelier vs lamp gallery vs table workshop, etc.)

## Stack

- Next.js 15 (App Router) + TypeScript + Tailwind CSS v4
- In-memory deal store (seeded demos) for the MVP
- Rule-based generative blueprint engine (`src/lib/generate-deal-room.ts`) — ready to swap/augment with an LLM

## Run

```bash
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

Demo deals:

- `/deals/deal_sofa_demo`
- `/deals/deal_lamp_demo`
- `/deals/deal_table_demo`

## API

- `GET /api/inquiries` — list deals
- `POST /api/inquiries` — create inquiry → generative deal room
- `GET /api/deals/:id` — fetch deal
- `POST /api/deals/:id/actions` — `{ action: "approve" | "request_change" | "reject", note? }`
