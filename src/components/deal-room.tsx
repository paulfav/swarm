"use client";

import type {
  Deal,
  DealRoomModule,
  TimelineEvent,
  Quote,
} from "@/lib/types";
import Link from "next/link";
import { useMemo, useState, useTransition } from "react";

function money(n: number) {
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 0,
  }).format(n);
}

function timeLabel(iso: string) {
  return new Intl.DateTimeFormat("en-US", {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  }).format(new Date(iso));
}

function ModuleShell({
  title,
  children,
  wide,
}: {
  title?: string;
  children: React.ReactNode;
  wide?: boolean;
}) {
  return (
    <section className={`module ${wide ? "wide" : ""}`}>
      {title ? <h2 className="module-title">{title}</h2> : null}
      {children}
    </section>
  );
}

function HeroStage({
  module,
  accent,
  imageDataUrl,
}: {
  module: Extract<DealRoomModule, { type: "hero_stage" }>;
  accent: string;
  imageDataUrl?: string;
}) {
  return (
    <section
      className={`hero-stage mood-${module.mood}`}
      style={{ ["--deal-accent" as string]: accent }}
    >
      <div className="hero-copy">
        <p className="eyebrow">{module.eyebrow}</p>
        <h1>{module.title}</h1>
        <p className="hero-sub">{module.subtitle}</p>
        <div className="badge-row">
          {module.badges.map((b) => (
            <span key={b}>{b}</span>
          ))}
        </div>
      </div>
      <div className="hero-visual" aria-hidden>
        {imageDataUrl ? (
          // eslint-disable-next-line @next/next/no-img-element
          <img src={imageDataUrl} alt="" />
        ) : (
          <div className="hero-abstract" />
        )}
      </div>
    </section>
  );
}

function DimensionFigure({
  module,
}: {
  module: Extract<DealRoomModule, { type: "dimension_figure" }>;
}) {
  return (
    <ModuleShell title={module.title}>
      <div className="dim-figure">
        <div className="dim-box">
          <span className="dim-w">
            {module.width}
            {module.unit}
          </span>
          <span className="dim-d">
            {module.depth}
            {module.unit}
          </span>
          <span className="dim-h">
            {module.height}
            {module.unit}
          </span>
        </div>
        <ul>
          {module.callouts.map((c) => (
            <li key={c}>{c}</li>
          ))}
        </ul>
      </div>
    </ModuleShell>
  );
}

function Timeline({
  title,
  events,
}: {
  title: string;
  events: TimelineEvent[];
}) {
  const sorted = useMemo(
    () => [...events].sort((a, b) => a.at.localeCompare(b.at)),
    [events],
  );

  return (
    <ModuleShell title={title} wide>
      <ol className="timeline">
        {sorted.map((evt, i) => (
          <li
            key={evt.id}
            className={`tl-item actor-${evt.actor}`}
            style={{ animationDelay: `${i * 60}ms` }}
          >
            <div className="tl-meta">
              <span className="tl-actor">{evt.actor}</span>
              <span className="tl-time">{timeLabel(evt.at)}</span>
            </div>
            <h3>{evt.title}</h3>
            <p>{evt.body}</p>
            {evt.facts ? (
              <dl className="tl-facts">
                {Object.entries(evt.facts).map(([k, v]) => (
                  <div key={k}>
                    <dt>{k.replaceAll("_", " ")}</dt>
                    <dd>{v}</dd>
                  </div>
                ))}
              </dl>
            ) : null}
          </li>
        ))}
      </ol>
      <p className="timeline-note">
        Verbatim Chinese factory messages stay internal. This is the commercial
        retranscription.
      </p>
    </ModuleShell>
  );
}

function LandedQuote({ title, quote }: { title: string; quote: Quote }) {
  return (
    <ModuleShell title={`${title} · v${quote.version}`} wide>
      <div className="quote-panel">
        <div className="quote-total">
          <span>Landed total</span>
          <strong>{money(quote.landedTotalUsd)}</strong>
          <em>
            {quote.incoterm} · MOQ {quote.moq} · {quote.leadTimeDays} days lead
          </em>
        </div>
        <ul className="quote-lines">
          {quote.lines.map((line) => (
            <li key={line.label}>
              <div>
                <span>{line.label}</span>
                {line.note ? <small>{line.note}</small> : null}
              </div>
              <strong>{money(line.amountUsd)}</strong>
            </li>
          ))}
        </ul>
        <p className="guarantee">{quote.guaranteeSummary}</p>
      </div>
    </ModuleShell>
  );
}

function SupplierThread({
  deal,
  onUpdated,
}: {
  deal: Deal;
  onUpdated: (deal: Deal) => void;
}) {
  const [paste, setPaste] = useState("");
  const [busy, setBusy] = useState(false);
  const [info, setInfo] = useState<string>();
  const [error, setError] = useState<string>();
  const messages = deal.thread ?? [];

  async function pasteReply() {
    if (!paste.trim()) return;
    setBusy(true);
    setError(undefined);
    setInfo(undefined);
    try {
      const res = await fetch(`/api/deals/${deal.id}/replies`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: paste, channel: "email" }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Failed");
      onUpdated(data.deal);
      setPaste("");
      setInfo("Reply ingested — quote refreshed if FOB/MOQ/lead time found.");
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed");
    } finally {
      setBusy(false);
    }
  }

  async function pollInbox() {
    setBusy(true);
    setError(undefined);
    setInfo(undefined);
    try {
      const res = await fetch(`/api/deals/${deal.id}/poll-replies`, {
        method: "POST",
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Poll failed");
      onUpdated(data.deal);
      if (!data.inboxConfigured) {
        setInfo(data.error || "IMAP not configured — paste a reply instead.");
      } else if (data.error) {
        setError(data.error);
      } else {
        setInfo(`Matched ${data.matched} inbox message(s).`);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : "Poll failed");
    } finally {
      setBusy(false);
    }
  }

  const waLink = deal.outreach?.find((o) => o.channel === "whatsapp")?.deepLink;

  return (
    <ModuleShell title="Supplier thread" wide>
      <div className="thread">
        <div className="thread-meta">
          <span>
            Contacts:{" "}
            {deal.contacts?.contactPerson || "—"}
            {deal.contacts?.whatsapp
              ? ` · WA +${deal.contacts.whatsapp}`
              : " · no WhatsApp"}
            {deal.contacts?.wechat
              ? ` · WeChat ${deal.contacts.wechat}`
              : " · no WeChat"}
          </span>
          {waLink ? (
            <a href={waLink} target="_blank" rel="noreferrer">
              Open WhatsApp deep link
            </a>
          ) : null}
        </div>

        <ol className="thread-list">
          {messages.length === 0 ? (
            <li className="thread-empty">
              No messages yet. Contact supplier to start the thread.
            </li>
          ) : (
            messages.map((m) => (
              <li
                key={m.id}
                className={`thread-msg dir-${m.direction} author-${m.author}`}
              >
                <div className="thread-msg-meta">
                  <span>
                    {m.direction} · {m.channel} · {m.author}
                  </span>
                  <span>{timeLabel(m.at)}</span>
                </div>
                <p className="thread-body">
                  {m.retranscription || m.body}
                </p>
                {m.retranscription && m.body !== m.retranscription ? (
                  <details>
                    <summary>Original</summary>
                    <pre>{m.body}</pre>
                  </details>
                ) : null}
              </li>
            ))
          )}
        </ol>

        <textarea
          rows={4}
          placeholder="Paste supplier reply (email/WhatsApp/WeChat text). We'll retranscribe + parse FOB / MOQ / lead time into the quote."
          value={paste}
          onChange={(e) => setPaste(e.target.value)}
          disabled={busy}
        />
        <div className="action-row">
          <button
            type="button"
            className="btn-primary"
            disabled={busy || !paste.trim()}
            onClick={() => pasteReply()}
          >
            Ingest reply
          </button>
          <button
            type="button"
            className="btn-secondary"
            disabled={busy}
            onClick={() => pollInbox()}
          >
            Poll agent inbox
          </button>
        </div>
        {info ? <p className="module-note">{info}</p> : null}
        {error ? <p className="form-error">{error}</p> : null}
      </div>
    </ModuleShell>
  );
}

function Actions({
  dealId,
  status,
  onUpdated,
  canContact,
}: {
  dealId: string;
  status: string;
  onUpdated: (deal: Deal) => void;
  canContact: boolean;
}) {
  const [note, setNote] = useState("");
  const [pending, startTransition] = useTransition();
  const [contacting, setContacting] = useState(false);
  const [error, setError] = useState<string>();

  function run(action: "approve" | "request_change" | "reject") {
    setError(undefined);
    startTransition(async () => {
      const res = await fetch(`/api/deals/${dealId}/actions`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ action, note: note || undefined }),
      });
      const data = await res.json();
      if (!res.ok) {
        setError(data.error || "Action failed");
        return;
      }
      onUpdated(data.deal);
      setNote("");
    });
  }

  async function contactSupplier() {
    setError(undefined);
    setContacting(true);
    try {
      const res = await fetch(`/api/deals/${dealId}/contact`, {
        method: "POST",
      });
      const data = await res.json();
      if (!res.ok) {
        setError(data.error || "Contact failed");
        return;
      }
      onUpdated(data.deal);
      const failed = (data.deal.outreach || []).filter(
        (o: { ok: boolean }) => !o.ok,
      );
      if (failed.length && !(data.deal.outreach || []).some((o: { ok: boolean }) => o.ok)) {
        setError(failed[failed.length - 1]?.error || "Outreach did not confirm");
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : "Contact failed");
    } finally {
      setContacting(false);
    }
  }

  const locked = status === "rejected" || status === "deposit_due";
  const busy = pending || contacting;

  return (
    <ModuleShell title="Your move" wide>
      <div className="actions">
        <p className="actions-lead">
          Status: <strong>{status.replaceAll("_", " ")}</strong>. Contact
          supplier hits Made-in-China (+ WhatsApp/WeChat when found). You never
          chat with the factory.
        </p>
        <textarea
          placeholder="Optional note for the agent (e.g. push MOQ, ask for darker linen)"
          value={note}
          onChange={(e) => setNote(e.target.value)}
          disabled={locked || busy}
          rows={3}
        />
        <div className="action-row">
          <button
            type="button"
            className="btn-primary"
            disabled={!canContact || busy}
            onClick={() => contactSupplier()}
          >
            {contacting
              ? "Contacting supplier channels…"
              : "Contact supplier"}
          </button>
          <button
            type="button"
            className="btn-secondary"
            disabled={locked || busy}
            onClick={() => run("approve")}
          >
            Approve quote
          </button>
          <button
            type="button"
            className="btn-secondary"
            disabled={status === "rejected" || busy}
            onClick={() => run("request_change")}
          >
            Request change
          </button>
          <button
            type="button"
            className="btn-ghost"
            disabled={status === "rejected" || busy}
            onClick={() => run("reject")}
          >
            Reject match
          </button>
        </div>
        <p className="module-note">
          Replies: paste into Supplier thread, poll agent inbox (mail.tm auto-setup),
          or email/WhatsApp webhooks. Agent email is set in env.
        </p>
        {error ? <p className="form-error">{error}</p> : null}
      </div>
    </ModuleShell>
  );
}

function renderModule(
  module: DealRoomModule,
  deal: Deal,
  onUpdated: (deal: Deal) => void,
) {
  switch (module.type) {
    case "hero_stage":
      return (
        <HeroStage
          key="hero"
          module={module}
          accent={deal.blueprint.accent}
          imageDataUrl={deal.imageDataUrl}
        />
      );
    case "spec_grid":
      return (
        <ModuleShell key="spec" title={module.title}>
          <dl className="spec-grid">
            {module.items.map((item) => (
              <div key={item.label}>
                <dt>{item.label}</dt>
                <dd>
                  {item.value}
                  {item.hint ? <small>{item.hint}</small> : null}
                </dd>
              </div>
            ))}
          </dl>
        </ModuleShell>
      );
    case "dimension_figure":
      return <DimensionFigure key="dims" module={module} />;
    case "finish_strip":
      return (
        <ModuleShell key="finish" title={module.title}>
          <div className="finish-strip">
            {module.options.map((opt) => (
              <button
                key={opt.name}
                type="button"
                className={opt.selected ? "selected" : ""}
                style={{ ["--swatch" as string]: opt.hex }}
              >
                <i />
                {opt.name}
              </button>
            ))}
          </div>
          {module.note ? <p className="module-note">{module.note}</p> : null}
        </ModuleShell>
      );
    case "material_callouts":
      return (
        <ModuleShell key="materials" title={module.title}>
          <ul className="material-list">
            {module.materials.map((m) => (
              <li key={m.name}>
                <strong>{m.name}</strong>
                <span>{m.detail}</span>
              </li>
            ))}
          </ul>
        </ModuleShell>
      );
    case "freight_volume":
      return (
        <ModuleShell key="freight" title={module.title}>
          <div className="freight">
            <div className="cbm">
              <strong>{module.cbm.toFixed(2)}</strong>
              <span>CBM</span>
            </div>
            <div>
              <p>{module.packNote}</p>
              <p className="route">{module.route}</p>
            </div>
          </div>
        </ModuleShell>
      );
    case "electrical_notes":
      return (
        <ModuleShell key="elec" title={module.title}>
          <p className="voltage">{module.voltage}</p>
          <ul className="cert-list">
            {module.certs.map((c) => (
              <li key={c}>{c}</li>
            ))}
          </ul>
          <ul className="note-list">
            {module.notes.map((n) => (
              <li key={n}>{n}</li>
            ))}
          </ul>
        </ModuleShell>
      );
    case "risk_flags":
      return (
        <ModuleShell key="risks" title={module.title}>
          <ul className="risk-list">
            {module.flags.map((f) => (
              <li key={f.text} className={`risk-${f.level}`}>
                {f.text}
              </li>
            ))}
          </ul>
        </ModuleShell>
      );
    case "match_evidence":
      return (
        <ModuleShell key="match" title={module.title}>
          <div className="confidence">
            <div
              className="confidence-bar"
              style={{
                width: `${Math.round(module.confidence * 100)}%`,
              }}
            />
            <span>{Math.round(module.confidence * 100)}% confidence</span>
          </div>
          <ul className="note-list">
            {module.points.map((p) => (
              <li key={p}>{p}</li>
            ))}
          </ul>
        </ModuleShell>
      );
    case "live_listings":
      return (
        <ModuleShell key="live" title={module.title} wide>
          <div className="live-source-meta">
            {deal.sourcing ? (
              <>
                <p>
                  Query: <strong>{deal.sourcing.query}</strong> ·{" "}
                  {deal.sourcing.live ? "live hits" : "no live hits"}
                </p>
                <ul className="source-attempts">
                  {deal.sourcing.attempts.map((a) => (
                    <li key={a.source} className={`attempt-${a.status}`}>
                      <strong>{a.source}</strong> — {a.status}: {a.detail}
                    </li>
                  ))}
                </ul>
              </>
            ) : (
              <p>No sourcing payload on this deal.</p>
            )}
          </div>
          <div className="live-listings">
            {(deal.sourcing?.listings ?? []).map((listing) => (
              <a
                key={listing.url}
                className="live-card"
                href={listing.url}
                target="_blank"
                rel="noreferrer"
              >
                {listing.imageUrl ? (
                  // eslint-disable-next-line @next/next/no-img-element
                  <img src={listing.imageUrl} alt="" />
                ) : (
                  <div className="live-card-ph" />
                )}
                <div>
                  <span className="live-source">{listing.source}</span>
                  <strong>{listing.title}</strong>
                  <em>
                    {listing.supplierName || "Supplier"}
                    {listing.priceUsd
                      ? ` · $${listing.priceUsd}`
                      : listing.rawPriceText
                        ? ` · ${listing.rawPriceText}`
                        : ""}
                  </em>
                </div>
              </a>
            ))}
          </div>
        </ModuleShell>
      );
    case "negotiation_timeline":
      return (
        <Timeline key="timeline" title={module.title} events={deal.timeline} />
      );
    case "supplier_thread":
      return (
        <SupplierThread key="thread" deal={deal} onUpdated={onUpdated} />
      );
    case "landed_quote":
      return (
        <LandedQuote key="quote" title={module.title} quote={deal.quote} />
      );
    case "actions":
      return (
        <Actions
          key="actions"
          dealId={deal.id}
          status={deal.status}
          canContact={Boolean(
            deal.sourcing?.listings?.some((l) => l.source === "made-in-china"),
          )}
          onUpdated={onUpdated}
        />
      );
    default:
      return null;
  }
}

export function DealRoom({ initialDeal }: { initialDeal: Deal }) {
  const [deal, setDeal] = useState(initialDeal);

  return (
    <div
      className={`deal-room layout-${deal.blueprint.layout}`}
      style={{ ["--deal-accent" as string]: deal.blueprint.accent }}
    >
      <header className="deal-topbar">
        <Link href="/">China Access</Link>
        <span>
          Generative deal room · {deal.blueprint.generator} ·{" "}
          {deal.providerName}
        </span>
      </header>
      <div className="deal-modules">
        {deal.blueprint.modules.map((module, index) => (
          <div
            key={`${module.type}-${index}`}
            className="module-enter"
            style={{ animationDelay: `${index * 45}ms` }}
          >
            {renderModule(module, deal, setDeal)}
          </div>
        ))}
      </div>
    </div>
  );
}
