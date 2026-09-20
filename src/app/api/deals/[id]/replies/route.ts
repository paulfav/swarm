import { NextResponse } from "next/server";
import { ingestSupplierReply } from "@/lib/replies";
import { getDealAsync, saveDeal } from "@/lib/store";

export const maxDuration = 60;

export async function POST(
  request: Request,
  context: { params: Promise<{ id: string }> },
) {
  const { id } = await context.params;
  const deal = await getDealAsync(id);
  if (!deal) {
    return NextResponse.json({ error: "Deal not found" }, { status: 404 });
  }

  const body = (await request.json()) as {
    text?: string;
    channel?: "email" | "whatsapp" | "wechat" | "made-in-china-inquiry";
    from?: string;
    subject?: string;
  };

  if (!body.text?.trim()) {
    return NextResponse.json({ error: "text is required" }, { status: 400 });
  }

  const updated = ingestSupplierReply(deal, {
    text: body.text,
    channel: body.channel || "email",
    from: body.from,
    subject: body.subject,
  });
  saveDeal(updated);
  return NextResponse.json({ deal: updated });
}
