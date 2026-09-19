import { NextResponse } from "next/server";
import { applyDealAction } from "@/lib/deals";
import { getDeal, saveDeal } from "@/lib/store";

export async function POST(
  request: Request,
  context: { params: Promise<{ id: string }> },
) {
  const { id } = await context.params;
  const deal = getDeal(id);
  if (!deal) {
    return NextResponse.json({ error: "Deal not found" }, { status: 404 });
  }

  const body = (await request.json()) as {
    action?: "approve" | "request_change" | "reject";
    note?: string;
  };

  if (
    !body.action ||
    !["approve", "request_change", "reject"].includes(body.action)
  ) {
    return NextResponse.json({ error: "Invalid action" }, { status: 400 });
  }

  const updated = applyDealAction(deal, body.action, body.note);
  saveDeal(updated);
  return NextResponse.json({ deal: updated });
}
