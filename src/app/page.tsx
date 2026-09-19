import { IntakeForm } from "@/components/intake-form";
import { listDeals } from "@/lib/store";
import Link from "next/link";

export default function HomePage() {
  const demos = listDeals().filter((d) => d.id.includes("_demo"));

  return (
    <div className="landing">
      <nav className="site-nav">
        <div className="brand-mark">China Access</div>
        <span>Hard goods · factory-direct</span>
      </nav>

      <header className="hero-landing">
        <div>
          <h1 className="hero-brand">China Access</h1>
          <p className="hero-line">
            Show us the product. An agent finds the producer, negotiates in
            Chinese, and codes a deal room on the spot — duties, guarantees, and
            a clean retranscription. You never chat with the factory.
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
                {deal.blueprint.layout.replaceAll("_", " ")} · landed{" "}
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
