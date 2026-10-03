// Minimal stroke icons (24x24), so the app needs no icon font or network.
type P = { size?: number };
const S = ({ size = 22, d }: P & { d: string }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
    <path d={d} />
  </svg>
);
export const IconLibrary = (p: P) => <S {...p} d="M4 19V5a2 2 0 0 1 2-2h12v18H6a2 2 0 0 1-2-2zm0 0a2 2 0 0 1 2-2h12M9 7h5" />;
export const IconSets = (p: P) => <S {...p} d="M8 6h13M8 12h13M8 18h13M3.5 6h.01M3.5 12h.01M3.5 18h.01" />;
export const IconInbox = (p: P) => <S {...p} d="M22 12h-6l-2 3h-4l-2-3H2m3.5-7h13L22 12v6a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2v-6z" />;
export const IconGear = (p: P) => <S {...p} d="M12 15a3 3 0 1 0 0-6 3 3 0 0 0 0 6zm7.4-3a7.4 7.4 0 0 0-.1-1.2l2-1.6-2-3.4-2.4 1a7.5 7.5 0 0 0-2-1.2L14.5 3h-5l-.4 2.6a7.5 7.5 0 0 0-2 1.2l-2.4-1-2 3.4 2 1.6a7.4 7.4 0 0 0 0 2.4l-2 1.6 2 3.4 2.4-1a7.5 7.5 0 0 0 2 1.2l.4 2.6h5l.4-2.6a7.5 7.5 0 0 0 2-1.2l2.4 1 2-3.4-2-1.6c.1-.4.1-.8.1-1.2z" />;
export const IconPlay = (p: P) => <S {...p} d="M6 4l14 8-14 8V4z" />;
export const IconPause = (p: P) => <S {...p} d="M7 4v16M17 4v16" />;
export const IconPlus = (p: P) => <S {...p} d="M12 5v14M5 12h14" />;
export const IconBack = (p: P) => <S {...p} d="M15 18l-6-6 6-6" />;
export const IconNext = (p: P) => <S {...p} d="M9 18l6-6-6-6" />;
export const IconEdit = (p: P) => <S {...p} d="M12 20h9M16.5 3.5a2.1 2.1 0 0 1 3 3L7 19l-4 1 1-4 12.5-12.5z" />;
export const IconTrash = (p: P) => <S {...p} d="M3 6h18M8 6V4h8v2m-9 0 1 14h8l1-14" />;
export const IconDrag = (p: P) => <S {...p} d="M9 5h.01M15 5h.01M9 12h.01M15 12h.01M9 19h.01M15 19h.01" />;
export const IconClose = (p: P) => <S {...p} d="M18 6 6 18M6 6l12 12" />;
export const IconSearch = (p: P) => <S {...p} d="M11 19a8 8 0 1 0 0-16 8 8 0 0 0 0 16zm10 2-4.3-4.3" />;
export const IconMetronome = (p: P) => <S {...p} d="M9 3h6l4 18H5L9 3zm3 13 5-9M7.5 16h9" />;
export const IconScroll = (p: P) => <S {...p} d="M12 3v14m0 0-5-5m5 5 5-5M5 21h14" />;
export const IconMusic = (p: P) => <S {...p} d="M9 18V5l12-2v13M9 18a3 3 0 1 1-6 0 3 3 0 0 1 6 0zm12-2a3 3 0 1 1-6 0 3 3 0 0 1 6 0z" />;
export const IconTv = (p: P) => <S {...p} d="M3 5h18v12H3zM8 21h8M12 17v4" />;
export const IconList = (p: P) => <S {...p} d="M3 12h18M3 6h18M3 18h18" />;
export const IconImport = (p: P) => <S {...p} d="M12 3v12m0 0 4-4m-4 4-4-4M4 17v2a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-2" />;
export const IconFile = (p: P) => <S {...p} d="M14 3H6a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9l-6-6zm0 0v6h6" />;
export const IconUsers = (p: P) => <S {...p} d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2M9 11a4 4 0 1 0 0-8 4 4 0 0 0 0 8zm14 10v-2a4 4 0 0 0-3-3.9M16 3.1a4 4 0 0 1 0 7.8" />;
export const IconMic = (p: P) => <S {...p} d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3zm7 10a7 7 0 0 1-14 0m7 7v3m-4 0h8" />;
