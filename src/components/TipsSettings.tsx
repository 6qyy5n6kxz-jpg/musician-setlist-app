import { useEffect, useState } from "react";
import { saveProfile, type Profile } from "../lib/db";
import { cleanHandle, type PayMethod, type RequestOptions } from "../lib/pay";
import { supabase } from "../lib/supabase";

/** Venmo / Cash App / PayPal handles, suggested amounts and which audience-page extras are on. */
export function TipsSettings({ profile }: { profile: Profile }) {
  const opts: RequestOptions = profile.request_options ?? {};
  const [venmo, setVenmo] = useState(opts.venmo ?? "");
  const [cashapp, setCashapp] = useState(opts.cashapp ?? "");
  const [paypal, setPaypal] = useState(opts.paypal ?? "");
  const [amounts, setAmounts] = useState((opts.amounts ?? [5, 10, 20]).join(", "));
  useEffect(() => {
    setVenmo(opts.venmo ?? ""); setCashapp(opts.cashapp ?? ""); setPaypal(opts.paypal ?? "");
    setAmounts((opts.amounts ?? [5, 10, 20]).join(", "));
  }, [profile.id]); // eslint-disable-line react-hooks/exhaustive-deps

  const save = (patch: Partial<RequestOptions>) => saveProfile({ request_options: { ...opts, ...patch } });
  const saveHandle = (m: PayMethod, v: string) => save({ [m]: cleanHandle(m, v) || null });
  const on = (k: "priority" | "shoutouts" | "email_list" | "booking") => opts[k] !== false;

  return (
    <div className="stack">
      <strong>💵 Tips & extras</strong>
      <div className="small dim">
        Patrons can add a tip to a request; their payment app opens with the amount filled in. The app records what they
        said they sent — check your Venmo / Cash App / PayPal to confirm.
      </div>
      <div className="meta-grid" style={{ gridTemplateColumns: "1fr 1fr 1fr" }}>
        <label className="field"><span>Venmo</span><input className="input" value={venmo} placeholder="@your-name" onChange={(e) => setVenmo(e.target.value)} onBlur={() => saveHandle("venmo", venmo)} /></label>
        <label className="field"><span>Cash App</span><input className="input" value={cashapp} placeholder="$yourname" onChange={(e) => setCashapp(e.target.value)} onBlur={() => saveHandle("cashapp", cashapp)} /></label>
        <label className="field"><span>PayPal.me</span><input className="input" value={paypal} placeholder="yourname" onChange={(e) => setPaypal(e.target.value)} onBlur={() => saveHandle("paypal", paypal)} /></label>
      </div>
      <label className="field"><span>Suggested tip buttons ($)</span>
        <input className="input" value={amounts} inputMode="decimal" onChange={(e) => setAmounts(e.target.value)}
          onBlur={() => {
            const list = amounts.split(/[,\s]+/).map(Number).filter((n) => n > 0 && n <= 1000).slice(0, 4);
            setAmounts((list.length ? list : [5, 10, 20]).join(", "));
            void save({ amounts: list.length ? list : [5, 10, 20] });
          }} />
      </label>
      <label className="check small"><input type="checkbox" checked={on("priority")} onChange={(e) => save({ priority: e.target.checked })} /> Tipped requests jump the queue (bigger tips first)</label>
      <label className="check small"><input type="checkbox" checked={on("shoutouts")} onChange={(e) => save({ shoutouts: e.target.checked })} /> Shoutouts (birthdays, anniversaries…)</label>
      <label className="check small"><input type="checkbox" checked={on("email_list")} onChange={(e) => save({ email_list: e.target.checked })} /> “Where are you playing next?” email sign-up</label>
      <label className="check small"><input type="checkbox" checked={on("booking")} onChange={(e) => save({ booking: e.target.checked })} /> “Book us for your event” form</label>
      <div className="small dim">Each act can use its own handles — set them under Acts below.</div>
    </div>
  );
}

interface Fan { id: string; email: string; name: string | null; created_at: string }

/** People who joined the email list from the request page; export as CSV for Mailchimp etc. */
export function FanList() {
  const [fans, setFans] = useState<Fan[] | null>(null);
  const [open, setOpen] = useState(false);
  const load = async () => {
    const { data } = await supabase.from("fans").select("id,email,name,created_at").order("created_at", { ascending: false }).limit(5000);
    setFans((data as Fan[]) ?? []);
  };
  useEffect(() => { void load(); }, []);

  const exportCsv = () => {
    if (!fans?.length) return;
    const esc = (v: string) => `"${v.replace(/"/g, '""')}"`;
    const csv = ["email,name,joined", ...fans.map((f) => [f.email, f.name ?? "", f.created_at.slice(0, 10)].map(esc).join(","))].join("\n");
    const a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([csv], { type: "text/csv" }));
    a.download = `email-list-${new Date().toISOString().slice(0, 10)}.csv`;
    a.click();
    URL.revokeObjectURL(a.href);
  };
  const remove = async (id: string) => {
    setFans((f) => f?.filter((x) => x.id !== id) ?? null);
    await supabase.from("fans").delete().eq("id", id);
  };

  const recent = fans?.filter((f) => Date.now() - Date.parse(f.created_at) < 1000 * 60 * 60 * 18).length ?? 0;
  return (
    <div className="stack">
      <div className="row">
        <strong className="grow">📬 Email list {fans ? `(${fans.length})` : ""}</strong>
        {recent > 0 && <span className="chip accent">+{recent} tonight</span>}
        <button className="btn small" onClick={exportCsv} disabled={!fans?.length}>Export CSV</button>
        <button className="btn small ghost" onClick={() => { setOpen(!open); if (!open) void load(); }}>{open ? "Hide" : "Show"}</button>
      </div>
      {fans?.length === 0 && <div className="small dim">Nobody yet — the sign-up box is at the bottom of your request page.</div>}
      {open && fans && fans.length > 0 && (
        <div style={{ maxHeight: 260, overflowY: "auto" }}>
          {fans.map((f) => (
            <div key={f.id} className="row small" style={{ padding: "4px 0", borderBottom: "1px solid var(--line)" }}>
              <span className="grow truncate">{f.email}{f.name ? <span className="dim"> · {f.name}</span> : null}</span>
              <span className="dim">{new Date(f.created_at).toLocaleDateString(undefined, { month: "short", day: "numeric" })}</span>
              <button className="btn small ghost" onClick={() => remove(f.id)} aria-label={`Remove ${f.email}`}>✕</button>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
