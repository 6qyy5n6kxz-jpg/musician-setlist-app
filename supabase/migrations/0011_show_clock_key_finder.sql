-- Show clock: when a gig starts and has to stop, and how long breaks run.
alter table public.setlists
  add column start_time text check (start_time is null or start_time ~ '^\d{2}:\d{2}$'),
  add column end_time text check (end_time is null or end_time ~ '^\d{2}:\d{2}$'),
  add column break_minutes smallint check (break_minutes is null or break_minutes between 0 and 120);

-- Key finder: each song's melody range (MIDI note numbers, in the written key) and each singer's range.
alter table public.songs
  add column melody_low smallint check (melody_low is null or melody_low between 24 and 108),
  add column melody_high smallint check (melody_high is null or melody_high between 24 and 108);

alter table public.profiles
  add column vocal_ranges jsonb not null default '{}'::jsonb;
