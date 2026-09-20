"use client";

import { useRouter } from "next/navigation";
import { FormEvent, useState } from "react";

export function IntakeForm() {
  const router = useRouter();
  const [description, setDescription] = useState("");
  const [sourceUrl, setSourceUrl] = useState("");
  const [quantity, setQuantity] = useState(1);
  const [destinationCountry, setDestinationCountry] = useState("USA");
  const [budgetUsd, setBudgetUsd] = useState("");
  const [imageDataUrl, setImageDataUrl] = useState<string>();
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string>();

  async function onSubmit(e: FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError(undefined);
    try {
      const res = await fetch("/api/inquiries", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          description,
          sourceUrl: sourceUrl || undefined,
          quantity,
          destinationCountry,
          budgetUsd: budgetUsd ? Number(budgetUsd) : undefined,
          imageDataUrl,
        }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Failed to open deal");
      router.push(`/deals/${data.deal.id}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong");
      setBusy(false);
    }
  }

  function onFile(file: File | null) {
    if (!file) {
      setImageDataUrl(undefined);
      return;
    }
    const reader = new FileReader();
    reader.onload = () => setImageDataUrl(String(reader.result));
    reader.readAsDataURL(file);
  }

  return (
    <form className="intake" onSubmit={onSubmit}>
      <label className="field">
        <span>What do you want sourced?</span>
        <textarea
          required
          rows={4}
          placeholder="e.g. Deep linen sofa like a $4k US cloud couch, 220cm, ship to NYC"
          value={description}
          onChange={(e) => setDescription(e.target.value)}
        />
      </label>

      <div className="intake-grid">
        <label className="field">
          <span>Retail / reference URL</span>
          <input
            type="url"
            placeholder="https://"
            value={sourceUrl}
            onChange={(e) => setSourceUrl(e.target.value)}
          />
        </label>
        <label className="field">
          <span>Photo</span>
          <input
            type="file"
            accept="image/*"
            onChange={(e) => onFile(e.target.files?.[0] ?? null)}
          />
        </label>
      </div>

      <div className="intake-grid three">
        <label className="field">
          <span>Qty</span>
          <input
            type="number"
            min={1}
            value={quantity}
            onChange={(e) => setQuantity(Number(e.target.value) || 1)}
          />
        </label>
        <label className="field">
          <span>Destination</span>
          <input
            value={destinationCountry}
            onChange={(e) => setDestinationCountry(e.target.value)}
          />
        </label>
        <label className="field">
          <span>Budget (USD)</span>
          <input
            type="number"
            min={0}
            placeholder="optional"
            value={budgetUsd}
            onChange={(e) => setBudgetUsd(e.target.value)}
          />
        </label>
      </div>

      {imageDataUrl ? (
        // eslint-disable-next-line @next/next/no-img-element
        <img className="intake-preview" src={imageDataUrl} alt="Reference" />
      ) : null}

      {error ? <p className="form-error">{error}</p> : null}

      <button className="btn-primary" type="submit" disabled={busy}>
        {busy ? "Searching China marketplaces…" : "Open deal room"}
      </button>
      <p className="fineprint">
        Live scrape of Made-in-China + AliExpress (Alibaba often CAPTCHA-blocked
        from cloud IPs) · hard goods only · no sampling · you never chat with
        the factory.
      </p>
    </form>
  );
}
