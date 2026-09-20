import { NextResponse } from "next/server";
import {
  getGreenApiConfig,
  greenApiGetAuthorizationCode,
  greenApiGetQr,
  greenApiGetState,
  greenApiReboot,
} from "@/lib/channels/green-api";

export const dynamic = "force-dynamic";

/** Fresh WhatsApp link QR (expires ~20s — poll every few seconds). */
export async function GET() {
  if (!getGreenApiConfig()) {
    return NextResponse.json(
      { ok: false, error: "Green-API not configured" },
      { status: 503 },
    );
  }

  const state = await greenApiGetState();
  if (state.stateInstance === "authorized") {
    return NextResponse.json({
      ok: true,
      authorized: true,
      state: state.stateInstance,
      qr: null,
    });
  }

  if (state.stateInstance === "starting") {
    return NextResponse.json({
      ok: false,
      authorized: false,
      state: state.stateInstance,
      error:
        "Instance is starting after a timed-out scan. Use Reboot, wait ~30s, then try again.",
      qr: null,
    });
  }

  const qr = await greenApiGetQr();
  if (!qr.ok || qr.type !== "qrCode" || !qr.message) {
    return NextResponse.json({
      ok: false,
      authorized: false,
      state: state.stateInstance || null,
      error: qr.error || "QR not ready yet — wait a few seconds or reboot",
      qr: null,
    });
  }

  return NextResponse.json({
    ok: true,
    authorized: false,
    state: state.stateInstance || null,
    qr: { mime: "image/png", base64: qr.message },
    fetchedAt: new Date().toISOString(),
  });
}

/**
 * POST actions:
 * - { action: "reboot" }
 * - { action: "phone_code", phone: "+15551234567" }
 */
export async function POST(request: Request) {
  if (!getGreenApiConfig()) {
    return NextResponse.json(
      { ok: false, error: "Green-API not configured" },
      { status: 503 },
    );
  }

  const body = (await request.json().catch(() => ({}))) as {
    action?: string;
    phone?: string;
  };

  if (body.action === "reboot") {
    const result = await greenApiReboot();
    return NextResponse.json(result, { status: result.ok ? 200 : 400 });
  }

  if (body.action === "phone_code") {
    const result = await greenApiGetAuthorizationCode(body.phone || "");
    return NextResponse.json(result, { status: result.ok ? 200 : 400 });
  }

  return NextResponse.json(
    { ok: false, error: "Unknown action" },
    { status: 400 },
  );
}
