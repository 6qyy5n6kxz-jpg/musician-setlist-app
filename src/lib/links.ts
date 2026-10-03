// Public links (audience request page, lyrics display, band follow) point at the deployed app.
export const PUBLIC_APP_URL = "https://stage.achangeofplansmusic.com/";

/** Absolute URL for an in-app route. Uses the deployed app when running on localhost, so QR codes work for patrons. */
export function appUrl(route: string): string {
  const local = /^(localhost|127\.|192\.168\.|10\.)/.test(location.hostname);
  const base = local ? PUBLIC_APP_URL : `${location.origin}${import.meta.env.BASE_URL}`;
  return `${base}#${route}`;
}
