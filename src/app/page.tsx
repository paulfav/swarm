import { IntakeForm } from "@/components/intake-form";
import { listDealsAsync } from "@/lib/store";
import Link from "next/link";

export const dynamic = "force-dynamic";
export const maxDuration = 120;

export default async function HomePage() {
  const deals = await listDealsAsync();
  const demos = deals.filter((d) => d.id.includes("_demo"));
  const agentEmail =
    process.env.CHINA_ACCESS_AGENT_EMAIL ||
    process.env.CHINA_ACCESS_MAILTM_ADDRESS ||
    null;
  const inboxReady = Boolean(
    process.env.CHINA_ACCESS_MAILTM_ADDRESS &&
      process.env.CHINA_ACCESS_MAILTM_PASSWORD,
  );

  return (
    <div className="landing">
      <nav className="site-nav">
        <div className="brand-mark">China Access</div>
        <span>
          {inboxReady
            ? `Inbox live · ${agentEmail}`
            : "Hard goods · live China sourcing"}
        </span>
      </nav>

      <header className="hero-landing">
        <div>
          <h1 className="hero-brand">China Access</h1>
          <p className="hero-line">
            Show us the product. We search Made-in-China and AliExpress live,
            attempt Alibaba, then code a deal room on the spot — duties,
            guarantees, and a clean retranscription. You never chat with the
            factory.
          </p>
          <div className="hero-cta-row">
            <a className="btn-primary" href="#open">
              Open a deal
            </a>
            <a className="btn-secondary" href="#demos">
              See tailored rooms
            </a>
          </div>
        </div>

        <div className="panel" id="open">
          <IntakeForm />
        </div>
      </header>

      <section className="section" id="demos">
        <h2>Each product gets its own room</h2>
        <div className="demo-grid">
          {demos.map((deal) => (
            <Link
              key={deal.id}
              className="demo-link"
              href={`/deals/${deal.id}`}
            >
              <strong>{deal.title}</strong>
              <span>
                {deal.sourcing?.live
                  ? `Live · ${deal.sourcing.listings.length} listings`
                  : deal.blueprint.layout.replaceAll("_", " ")}{" "}
                · landed{" "}
                {new Intl.NumberFormat("en-US", {
                  style: "currency",
                  currency: "USD",
                  maximumFractionDigits: 0,
                }).format(deal.quote.landedTotalUsd)}
              </span>
            </Link>
          ))}
        </div>
      </section>
    </div>
  );
}
