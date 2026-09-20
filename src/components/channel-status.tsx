"use client";

import { useEffect, useState, useTransition } from "react";

type Status = {
  agentEmail: string | null;
  inboxConfigured: boolean;
  whatsapp: {
    greenApiConfigured: boolean;
    authorized: boolean;
    state: string | null;
    error: string | null;
  };
  wechat: { wecomConfigured: boolean; note?: string };
};

type QrPayload = {
  ok: boolean;
  authorized?: boolean;
  state?: string | null;
  qr?: { mime: string; base64: string } | null;
  error?: string;
  fetchedAt?: string;
};

export function ChannelStatus() {
  const [status, setStatus] = useState<Status | null>(null);
  const [qr, setQr] = useState<QrPayload | null>(null);
  const [phone, setPhone] = useState("");
  const [linkCode, setLinkCode] = useState<string | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);
  const [pending, startTransition] = useTransition();

  useEffect(() => {
    let cancelled = false;
    const loadStatus = () => {
      fetch("/api/inbox/status")
        .then((r) => r.json())
        .then((j) => {
          if (!cancelled) setStatus(j);
        })
        .catch(() => undefined);
    };
    loadStatus();
    const t = setInterval(loadStatus, 10000);
    return () => {
      cancelled = true;
      clearInterval(t);
    };
  }, []);

  const needsLink =
    Boolean(status?.whatsapp.greenApiConfigured) &&
    !status?.whatsapp.authorized;

  useEffect(() => {
    if (!needsLink) {
      setQr(null);
      return;
    }
    let cancelled = false;
    const loadQr = () => {
      fetch("/api/whatsapp/link", { cache: "no-store" })
        .then((r) => r.json())
        .then((j: QrPayload) => {
          if (cancelled) return;
          setQr(j);
          if (j.authorized) {
            fetch("/api/inbox/status")
              .then((r) => r.json())
              .then(setStatus)
              .catch(() => undefined);
          }
        })
        .catch(() => undefined);
    };
    loadQr();
    // Green-API QR rotates ~every 20s — refresh before it expires
    const t = setInterval(loadQr, 4000);
    return () => {
      cancelled = true;
      clearInterval(t);
    };
  }, [needsLink]);

  function reboot() {
    setActionError(null);
    setLinkCode(null);
    startTransition(async () => {
      const res = await fetch("/api/whatsapp/link", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ action: "reboot" }),
      });
      const j = await res.json();
      if (!j.ok) {
        setActionError(j.error || "Reboot failed");
        return;
      }
      setActionError("Rebooted — wait ~30s for a fresh QR");
      setQr(null);
    });
  }

  function requestPhoneCode() {
    setActionError(null);
    setLinkCode(null);
    startTransition(async () => {
      const res = await fetch("/api/whatsapp/link", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ action: "phone_code", phone }),
      });
      const j = await res.json();
      if (!j.ok) {
        setActionError(j.error || "Could not get link code");
        return;
      }
      setLinkCode(j.code);
    });
  }

  if (!status) return null;

  const wa = status.whatsapp;
  const showQr = needsLink && qr?.qr?.base64;

  return (
    <section className="section channel-status" id="channels">
      <h2>Agent channels</h2>
      <p className="hero-line" style={{ maxWidth: "40rem" }}>
        Inbox {status.inboxConfigured ? "live" : "pending"}
        {status.agentEmail ? ` · ${status.agentEmail}` : ""}. WhatsApp{" "}
        {wa.authorized
          ? "authorized"
          : wa.greenApiConfigured
            ? `instance ready (${wa.state || qr?.state || "pending"})`
            : "not provisioned"}
        . WeChat{" "}
        {status.wechat.wecomConfigured ? "WeCom ready" : "needs China corp WeCom"}
        .
      </p>

      {needsLink ? (
        <div className="wa-link-panel">
          <div className="wa-link-col">
            <h3>Link with phone number (recommended)</h3>
            <ol className="wa-steps">
              <li>WhatsApp → Linked devices → Link a device</li>
              <li>Choose “Link with phone number instead”</li>
              <li>Enter your WhatsApp number below, then type the code into WhatsApp</li>
            </ol>
            <div className="wa-phone-row">
              <input
                type="tel"
                inputMode="tel"
                placeholder="+15551234567"
                value={phone}
                onChange={(e) => setPhone(e.target.value)}
                aria-label="WhatsApp phone number"
              />
              <button
                type="button"
                className="btn-primary"
                disabled={pending || phone.replace(/\D/g, "").length < 8}
                onClick={requestPhoneCode}
              >
                Get code
              </button>
            </div>
            {linkCode ? (
              <p className="wa-link-code">
                Enter in WhatsApp: <strong>{linkCode}</strong>
                <span> · valid ~2 minutes</span>
              </p>
            ) : null}
          </div>

          <div className="wa-link-col">
            <h3>Or scan live QR</h3>
            <p className="wa-hint">
              Scan only this live code (not a screenshot). It refreshes every few
              seconds — fit the whole square in the camera.
            </p>
            {showQr ? (
              // eslint-disable-next-line @next/next/no-img-element
              <img
                className="wa-qr-img"
                src={`data:${qr!.qr!.mime};base64,${qr!.qr!.base64}`}
                alt="WhatsApp link QR — refreshes automatically"
                width={280}
                height={280}
              />
            ) : (
              <p className="wa-hint">
                {qr?.error ||
                  actionError ||
                  "Waiting for a fresh QR from Green-API…"}
              </p>
            )}
            <div className="wa-phone-row">
              <button
                type="button"
                className="btn-secondary"
                disabled={pending}
                onClick={reboot}
              >
                Reboot instance
              </button>
            </div>
          </div>
        </div>
      ) : null}

      {actionError && needsLink ? (
        <p className="wa-action-error">{actionError}</p>
      ) : null}
    </section>
  );
}
