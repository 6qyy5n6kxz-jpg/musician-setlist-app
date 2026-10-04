-- Key per singer, and a gig log (what was played, when and where) that syncs offline-first
-- like songs/setlists (client UUIDs, last-write-wins via sync_row()).

alter table public.songs
  add column key_kendra text,
  add column key_devin text;

create table public.gigs (
  id uuid primary key default gen_random_uuid(),
  owner_id uuid not null default auth.uid() references auth.users (id) on delete cascade,
  setlist_id uuid references public.setlists (id) on delete set null,
  act_id text,
  name text not null default '',
  venue text,
  gig_date date not null default current_date,
  notes text,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  server_updated_at timestamptz not null default now(),
  deleted_at timestamptz
);
create index gigs_owner_sync on public.gigs (owner_id, server_updated_at);

create table public.gig_songs (
  id uuid primary key default gen_random_uuid(),
  owner_id uuid not null default auth.uid() references auth.users (id) on delete cascade,
  gig_id uuid not null references public.gigs (id) on delete cascade,
  song_id uuid references public.songs (id) on delete set null,
  played_at timestamptz not null default now(),
  from_request boolean not null default false,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  server_updated_at timestamptz not null default now(),
  deleted_at timestamptz
);
create index gig_songs_owner_sync on public.gig_songs (owner_id, server_updated_at);
create index gig_songs_gig on public.gig_songs (gig_id);
create index gig_songs_song on public.gig_songs (song_id);

create trigger gigs_sync before insert or update on public.gigs
  for each row execute function public.sync_row();
create trigger gig_songs_sync before insert or update on public.gig_songs
  for each row execute function public.sync_row();

alter table public.gigs enable row level security;
alter table public.gig_songs enable row level security;
create policy "own gigs" on public.gigs for all to authenticated
  using (owner_id = (select auth.uid())) with check (owner_id = (select auth.uid()));
create policy "own gig songs" on public.gig_songs for all to authenticated
  using (owner_id = (select auth.uid())) with check (owner_id = (select auth.uid()));

alter publication supabase_realtime add table public.gigs, public.gig_songs;
