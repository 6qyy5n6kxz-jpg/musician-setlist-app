// Payment links for tips: open Venmo / Cash App / PayPal with the amount (and a note) filled in.
// The app never handles money — the patron pays in their own payment app.

export type PayMethod = "venmo" | "cashapp" | "paypal";

export interface PayHandles { venmo?: string | null; cashapp?: string | null; paypal?: string | null }

export interface RequestOptions extends PayHandles {
  /** Suggested tip buttons, in dollars. */
  amounts?: number[];
  /** Tipped requests go to the top of the queue. */
  priority?: boolean;
  shoutouts?: boolean;
  email_list?: boolean;
  booking?: boolean;
}

export const PAY_LABELS: Record<PayMethod, string> = { venmo: "Venmo", cashapp: "Cash App", paypal: "PayPal" };

/** "@Devin-Frank-5", "venmo.com/u/Devin-Frank-5" → "Devin-Frank-5"; "$devinfrank" → "devinfrank". */
export function cleanHandle(method: PayMethod, raw: string | null | undefined): string {
  let h = (raw ?? "").trim();
  h = h.replace(/^https?:\/\//i, "").replace(/^(www\.)?(account\.)?(venmo\.com\/(u\/)?|cash\.app\/|paypal\.me\/|paypal\.com\/paypalme\/)/i, "");
  h = h.split(/[/?#\s]/)[0];
  h = method === "cashapp" ? h.replace(/^\$/, "") : h.replace(/^@/, "");
  return h.replace(/[^A-Za-z0-9_.-]/g, "");
}

/** Link that opens the payment app (or site) ready to send `amount` with `note`. */
export function payLink(method: PayMethod, handle: string, amount?: number | null, note?: string): string {
  const h = encodeURIComponent(cleanHandle(method, handle));
  const amt = amount && amount > 0 ? String(Math.round(amount * 100) / 100) : "";
  if (method === "venmo") {
    const q = new URLSearchParams({ txn: "pay", ...(amt ? { amount: amt } : {}), ...(note ? { note } : {}) });
    return `https://venmo.com/${h}?${q.toString()}`;
  }
  if (method === "cashapp") return `https://cash.app/$${h}${amt ? `/${amt}` : ""}`;
  return `https://paypal.me/${h}${amt ? `/${amt}USD` : ""}`;
}

export function availableMethods(p: PayHandles): PayMethod[] {
  return (["venmo", "cashapp", "paypal"] as PayMethod[]).filter((m) => cleanHandle(m, p[m]));
}

export const formatMoney = (n: number) => `$${Number.isInteger(n) ? n : n.toFixed(2)}`;
