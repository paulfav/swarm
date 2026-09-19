"use client";

import { useEffect, useState } from "react";

type Status = {
  agentEmail: string | null;
  inboxConfigured: boolean;
  whatsapp: {
    greenApiConfigured: boolean;
    authorized: boolean;
    state: string | null;
    qr: { mime: string; base64: string } | null;
    error: string | null;
  };
  wechat: { wecomConfigured: boolean; note?: string };
};

export function ChannelStatus() {
  const [status, setStatus] = useState<Status | null>(null);

  useEffect(() => {
    let cancelled = false;
    const load = () => {
      fetch("/api/inbox/status")
        .then((r) => r.json())
        .then((j) => {
          if (!cancelled) setStatus(j);
        })
        .catch(() => undefined);
    };
    load();
    const t = setInterval(load, 8000);
    return () => {
      cancelled = true;
      clearInterval(t);
    };
  }, []);

  if (!status) return null;

  const wa = status.whatsapp;
  const needsQr = wa.greenApiConfigured && !wa.authorized && wa.qr?.base64;

  return (
    <section className="section channel-status" id="channels">
      <h2>Agent channels</h2>
      <p className="hero-line" style={{ maxWidth: "36rem" }}>
        Inbox {status.inboxConfigured ? "live" : "pending"}
        {status.agentEmail ? ` · ${status.agentEmail}` : ""}. WhatsApp{" "}
        {wa.authorized
          ? "authorized"
          : wa.greenApiConfigured
            ? `instance ready (${wa.state || "pending"})`
            : "not provisioned"}
        . WeChat{" "}
        {status.wechat.wecomConfigured ? "WeCom ready" : "needs China corp WeCom"}.
      </p>
      {needsQr ? (
        <div className="wa-qr-block">
          <p>
            Scan once with WhatsApp → Linked devices to finish self-setup.
          </p>
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={`data:${wa.qr!.mime};base64,${wa.qr!.base64}`}
            alt="WhatsApp link QR"
            width={220}
            height={220}
          />
        </div>
      ) : null}
    </section>
  );
}
