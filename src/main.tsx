import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { App } from "./App";
import { applyTheme, getSettings } from "./lib/settings";
import "./styles.css";

applyTheme(getSettings().theme);

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);

// Dev-only hook for seeding/inspecting local data from the browser console.
if (import.meta.env.DEV) void import("./lib/db").then(({ db }) => Object.assign(window, { stageDb: db }));
