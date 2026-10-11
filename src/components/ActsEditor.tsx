import { useEffect, useState } from "react";
import { newId, saveProfile, type Act, type Profile } from "../lib/db";
import { cleanHandle } from "../lib/pay";

/** Edit acts (solo, duo, …) and choose which one is performing. */
export function ActsEditor({ profile }: { profile: Profile }) {
  const [acts, setActs] = useState<Act[]>(profile.acts ?? []);
  useEffect(() => setActs(profile.acts ?? []), [profile.id]); // eslint-disable-line react-hooks/exhaustive-deps

  const commit = (next: Act[]) => {
    setActs(next);
    void saveProfile({ acts: next, active_act: next.some((a) => a.id === profile.active_act) ? profile.active_act : next[0]?.id ?? null });
  };
  const change = (id: string, patch: Partial<Act>) => setActs((all) => all.map((a) => (a.id === id ? { ...a, ...patch } : a)));
  const save = () => commit(acts);

  return (
    <div className="stack">
      <div className="row wrap">
        <strong className="grow">Performing as</strong>
        {acts.length > 0 ? (
          <div className="seg">
            {acts.map((a) => (
              <button key={a.id} className={profile.active_act === a.id ? "on" : ""} onClick={() => saveProfile({ active_act: a.id })}>
                {a.name || "Untitled"}
              </button>
            ))}
          </div>
        ) : <span className="small dim">Add your acts below</span>}
      </div>
      <p className="small dim" style={{ margin: 0 }}>
        The request page shows the performing act's name, message and tip handles (or the defaults under Tips & extras). Starting Perform with a setlist switches to that set's act automatically.
      </p>
      {acts.map((a) => (
        <div key={a.id} className="card stack" style={{ background: "var(--bg-sunken)", padding: 12 }}>
          <div className="row">
            <input className="input grow" style={{ fontWeight: 700 }} value={a.name} placeholder="Act name (e.g. A Change of Plans)"
              onChange={(e) => change(a.id, { name: e.target.value })} onBlur={save} />
            <button className="btn small ghost danger" onClick={() => { if (confirm(`Remove “${a.name}”?`)) commit(acts.filter((x) => x.id !== a.id)); }}>Remove</button>
          </div>
          <input className="input" value={a.message} placeholder="Message to the audience (Request a song! Tips appreciated 🎸)"
            onChange={(e) => change(a.id, { message: e.target.value })} onBlur={save} />
          <div className="meta-grid" style={{ gridTemplateColumns: "1fr 1fr 1fr" }}>
            <input className="input" value={a.venmo ?? ""} placeholder="Venmo @handle" aria-label="Venmo for this act"
              onChange={(e) => change(a.id, { venmo: cleanHandle("venmo", e.target.value) })} onBlur={save} />
            <input className="input" value={a.cashapp ?? ""} placeholder="Cash App $tag" aria-label="Cash App for this act"
              onChange={(e) => change(a.id, { cashapp: cleanHandle("cashapp", e.target.value) })} onBlur={save} />
            <input className="input" value={a.paypal ?? ""} placeholder="PayPal.me name" aria-label="PayPal for this act"
              onChange={(e) => change(a.id, { paypal: cleanHandle("paypal", e.target.value) })} onBlur={save} />
          </div>
          {!a.venmo && !a.cashapp && !a.paypal && (
            <input className="input" value={a.tip_url} placeholder="Or any tip link: https://…"
              onChange={(e) => change(a.id, { tip_url: e.target.value })} onBlur={save} />
          )}
        </div>
      ))}
      <div className="row wrap">
        <button className="btn small" onClick={() => commit([...acts, { id: newId(), name: "", message: "", tip_url: "" }])}>+ Add act</button>
        {acts.length === 0 && (
          <button className="btn small primary" onClick={() => commit([
            { id: newId(), name: "A Change of Plans", message: "", tip_url: "" },
            { id: newId(), name: "Devin Frank", message: "", tip_url: "" },
          ])}>Add A Change of Plans + Devin Frank</button>
        )}
      </div>
    </div>
  );
}

export function activeAct(profile: Profile | undefined): Act | undefined {
  return profile?.acts?.find((a) => a.id === profile.active_act);
}
