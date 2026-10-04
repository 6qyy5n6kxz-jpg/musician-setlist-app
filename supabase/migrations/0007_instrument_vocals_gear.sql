-- Duo categories and gear: who sings lead, which instrument Devin plays, and per-song settings for
-- the Numa X Piano 73, Neural DSP Nano Cortex and BeatBuddy (switched from the MIDI Captain).

alter table public.songs
  add column instrument text check (instrument is null or instrument in ('piano', 'electric', 'acoustic')),
  add column lead_vocal text check (lead_vocal is null or lead_vocal in ('kendra', 'devin', 'both')),
  add column gear jsonb not null default '{}'::jsonb;

-- The performer's own preset lists (names, numbers, sound categories) used for suggestions.
alter table public.profiles
  add column gear_library jsonb not null default '{}'::jsonb;
