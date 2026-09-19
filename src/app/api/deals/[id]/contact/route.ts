import { NextResponse } from "next/server";
import { contactSupplierOnDeal } from "@/lib/deals";
import { getDealAsync, saveDeal } from "@/lib/store";

export const maxDuration = 120;

export async function POST(
  _request: Request,
  context: { params: Promise<{ id: string }> },
) {
  const { id } = await context.params;
  const deal = await getDealAsync(id);
  if (!deal) {
    return NextResponse.json({ error: "Deal not found" }, { status: 404 });
  }

  const updated = await contactSupplierOnDeal(deal);
  saveDeal(updated);

  const last = updated.outreach?.[updated.outreach.length - 1];
  return NextResponse.json({
    deal: updated,
    outreach: last,
  });
}
