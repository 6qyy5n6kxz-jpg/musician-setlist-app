import { useEffect, type ReactNode } from "react";
import { HashRouter, NavLink, Navigate, Outlet, Route, Routes, useNavigate } from "react-router-dom";
import { useRegisterSW } from "virtual:pwa-register/react";
import { IconGear, IconInbox, IconLibrary, IconSets } from "./components/Icons";
import { SignedOutBanner } from "./components/SyncBadge";
import { describeRequest, RequestsProvider, useRequests } from "./lib/requests";
import { startSync } from "./lib/sync";
import { Library } from "./pages/Library";
import { SongPage } from "./pages/SongPage";
import { SongEditor } from "./pages/SongEditor";
import { ImportPage } from "./pages/ImportPage";
import { Setlists } from "./pages/Setlists";
import { SetlistEditor } from "./pages/SetlistEditor";
import { Perform } from "./pages/Perform";
import { RequestsPage } from "./pages/RequestsPage";
import { SettingsPage } from "./pages/SettingsPage";
import { GigCheck } from "./pages/GigCheck";
import { TuneUp } from "./pages/TuneUp";
import { GigDetail, GigLog } from "./pages/GigLog";
import { PublicRequest } from "./pages/public/PublicRequest";
import { LyricsDisplay } from "./pages/public/LyricsDisplay";
import { BandFollow } from "./pages/public/BandFollow";

export function App() {
  return (
    <HashRouter>
      <Routes>
        {/* Public pages: no login, reached by secret links / QR codes */}
        <Route element={<PublicShell />}>
          <Route path="/r/:token" element={<PublicRequest />} />
          <Route path="/display/:token" element={<LyricsDisplay />} />
          <Route path="/band/:token" element={<BandFollow />} />
        </Route>

        <Route element={<Private />}>
          <Route path="/perform/:setlistId" element={<Perform />} />
          <Route path="/perform/song/:songId" element={<Perform />} />
          <Route element={<Shell />}>
            <Route path="/" element={<Library />} />
            <Route path="/song/:id" element={<SongPage />} />
            <Route path="/song/:id/edit" element={<SongEditor />} />
            <Route path="/import" element={<ImportPage />} />
            <Route path="/sets" element={<Setlists />} />
            <Route path="/sets/:id" element={<SetlistEditor />} />
            <Route path="/requests" element={<RequestsPage />} />
            <Route path="/settings" element={<SettingsPage />} />
            <Route path="/gigcheck" element={<GigCheck />} />
            <Route path="/tuneup" element={<TuneUp />} />
            <Route path="/gigs" element={<GigLog />} />
            <Route path="/gigs/:id" element={<GigDetail />} />
          </Route>
          <Route path="*" element={<Navigate to="/" replace />} />
        </Route>
      </Routes>
    </HashRouter>
  );
}

/** Everything that belongs to the performer: local data, sync, request queue. */
function Private() {
  useEffect(() => startSync(), []);
  return (
    <RequestsProvider>
      <Outlet />
      <RequestToasts />
      <UpdatePrompt />
    </RequestsProvider>
  );
}

function Shell() {
  const { newCount } = useRequests();
  return (
    <div className="shell">
      <main className="main">
        <SignedOutBanner />
        <Outlet />
      </main>
      <nav className="tabbar no-print">
        <Tab to="/" icon={<IconLibrary />} label="Songs" end />
        <Tab to="/sets" icon={<IconSets />} label="Setlists" />
        <Tab to="/requests" icon={<IconInbox />} label="Requests" badge={newCount} />
        <Tab to="/settings" icon={<IconGear />} label="Settings" />
      </nav>
    </div>
  );
}

function Tab({ to, icon, label, badge, end }: { to: string; icon: ReactNode; label: string; badge?: number; end?: boolean }) {
  return (
    <NavLink to={to} end={end} className={({ isActive }) => (isActive ? "active" : "")}>
      {icon}
      <span>{label}</span>
      {!!badge && <span className="badge">{badge}</span>}
    </NavLink>
  );
}

function RequestToasts() {
  const { toasts, dismissToast } = useRequests();
  const navigate = useNavigate();
  const inPerform = location.hash.startsWith("#/perform");
  if (!toasts.length || inPerform) return null; // Perform mode shows its own pop-up
  return (
    <div className="toast-stack">
      {toasts.map((t) => (
        <div key={t.id} className="toast" onClick={() => { dismissToast(t.id); navigate("/requests"); }}>
          <span className="badge">!</span>
          <div className="grow">
            <div style={{ fontWeight: 700 }}>{describeRequest(t).title}</div>
            <div className="small dim">{describeRequest(t).detail}</div>
          </div>
        </div>
      ))}
    </div>
  );
}

/**
 * Public screens (TV display, band phones, audience page) update themselves: nobody is there to
 * tap "Update", and a TV left on the display page would otherwise run old code indefinitely.
 */
function PublicShell() {
  const { needRefresh: [needRefresh], updateServiceWorker } = useRegisterSW({
    onRegisteredSW(_url, reg) {
      if (reg) setInterval(() => void reg.update(), 30 * 60_000); // check for new versions every 30 min
    },
  });
  useEffect(() => {
    if (needRefresh) void updateServiceWorker(true);
  }, [needRefresh, updateServiceWorker]);
  return <Outlet />;
}

/** New version available: ask before reloading (never mid-song). */
function UpdatePrompt() {
  const { needRefresh: [needRefresh], updateServiceWorker } = useRegisterSW({
    onRegisteredSW(_url, reg) {
      if (reg) setInterval(() => void reg.update(), 60 * 60_000);
    },
  });
  if (!needRefresh || location.hash.startsWith("#/perform")) return null;
  return (
    <div className="toast-stack">
      <div className="toast">
        <div className="grow">A new version of the app is ready.</div>
        <button className="btn small primary" onClick={() => updateServiceWorker(true)}>Update</button>
      </div>
    </div>
  );
}
