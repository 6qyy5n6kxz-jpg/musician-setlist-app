-- Setlist Stage schema: offline-first sync tables, audience requests, live sessions.
--
-- Sync model: clients generate UUIDs and stamp `updated_at` with their own clock.
-- The sync_row trigger keeps the newest `updated_at` (last write wins per row) and
-- stamps `server_updated_at`, which clients use as their pull cursor.

create extension if not exists pgcrypto with schema extensions;

-- ---------------------------------------------------------------- helpers
create or replace function public.sync_row()
returns trigger
language plpgsql
set search_path = ''
as $$
begin
  if tg_op = 'UPDATE' then
    if new.updated_at < old.updated_at then
      return old;  -- stale write from a device that was offline; keep the newer row
    end if;
    new.owner_id := old.owner_id;
  end if;
  new.server_updated_at := clock_timestamp();
  return new;
end;
$$;

create or replace function public.random_token(len int default 20)
returns text
language sql
volatile
set search_path = ''
as $$
  select string_agg(substr('abcdefghjkmnpqrstuvwxyz23456789', 1 + floor(random() * 31)::int, 1), '')
  from generate_series(1, len);
$$;

-- ---------------------------------------------------------------- profiles
create table public.profiles (
  id uuid primary key references auth.users (id) on delete cascade,
  display_name text,
  request_token text not null unique default public.random_token(10),
  requests_open boolean not null default false,
  request_message text,
  tip_url text,
  live_token text not null unique default public.random_token(20),
  settings jsonb not null default '{}'::jsonb,
  updated_at timestamptz not null default now(),
  server_updated_at timestamptz not null default now()
);

create or replace function public.profiles_sync_row()
returns trigger
language plpgsql
set search_path = ''
as $$
begin
  if tg_op = 'UPDATE' and new.updated_at < old.updated_at then
    return old;
  end if;
  new.server_updated_at := clock_timestamp();
  return new;
end;
$$;

create trigger profiles_sync before insert or update on public.profiles
  for each row execute function public.profiles_sync_row();

-- ---------------------------------------------------------------- songs
create table public.songs (
  id uuid primary key default gen_random_uuid(),
  owner_id uuid not null default auth.uid() references auth.users (id) on delete cascade,
  title text not null default '',
  artist text not null default '',
  song_key text,
  tempo int check (tempo is null or tempo between 20 and 400),
  time_signature text,
  duration_sec int check (duration_sec is null or duration_sec between 0 and 7200),
  capo int not null default 0 check (capo between 0 and 12),
  tags text[] not null default '{}',
  genre text,
  year int,
  ccli text,
  content text not null default '',
  notes text,
  flow text,
  requestable boolean not null default true,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  server_updated_at timestamptz not null default now(),
  deleted_at timestamptz
);
create index songs_owner_sync on public.songs (owner_id, server_updated_at);

create table public.song_files (
  id uuid primary key default gen_random_uuid(),
  owner_id uuid not null default auth.uid() references auth.users (id) on delete cascade,
  song_id uuid not null references public.songs (id) on delete cascade,
  kind text not null check (kind in ('pdf', 'audio', 'image')),
  name text not null default '',
  mime text,
  size int,
  storage_path text,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  server_updated_at timestamptz not null default now(),
  deleted_at timestamptz
);
create index song_files_owner_sync on public.song_files (owner_id, server_updated_at);
create index song_files_song on public.song_files (song_id);

-- ---------------------------------------------------------------- setlists
create table public.setlists (
  id uuid primary key default gen_random_uuid(),
  owner_id uuid not null default auth.uid() references auth.users (id) on delete cascade,
  name text not null default '',
  event_date date,
  venue text,
  notes text,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  server_updated_at timestamptz not null default now(),
  deleted_at timestamptz
);
create index setlists_owner_sync on public.setlists (owner_id, server_updated_at);

create table public.setlist_items (
  id uuid primary key default gen_random_uuid(),
  owner_id uuid not null default auth.uid() references auth.users (id) on delete cascade,
  setlist_id uuid not null references public.setlists (id) on delete cascade,
  song_id uuid references public.songs (id) on delete set null,
  kind text not null default 'song' check (kind in ('song', 'break')),
  label text,
  position double precision not null default 0,
  key_override text,
  capo_override int check (capo_override is null or capo_override between 0 and 12),
  notes text,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  server_updated_at timestamptz not null default now(),
  deleted_at timestamptz
);
create index setlist_items_owner_sync on public.setlist_items (owner_id, server_updated_at);
create index setlist_items_setlist on public.setlist_items (setlist_id);
create index setlist_items_song on public.setlist_items (song_id);

create trigger songs_sync before insert or update on public.songs
  for each row execute function public.sync_row();
create trigger song_files_sync before insert or update on public.song_files
  for each row execute function public.sync_row();
create trigger setlists_sync before insert or update on public.setlists
  for each row execute function public.sync_row();
create trigger setlist_items_sync before insert or update on public.setlist_items
  for each row execute function public.sync_row();

-- ---------------------------------------------------------------- requests
create table public.song_requests (
  id uuid primary key default gen_random_uuid(),
  owner_id uuid not null references auth.users (id) on delete cascade,
  song_id uuid references public.songs (id) on delete set null,
  title text not null,
  artist text,
  patron_name text,
  message text,
  status text not null default 'new' check (status in ('new', 'queued', 'played', 'declined')),
  ip_hash text,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);
create index song_requests_owner on public.song_requests (owner_id, created_at desc);
create index song_requests_ip on public.song_requests (ip_hash, created_at desc);
create index song_requests_song on public.song_requests (song_id);

-- ---------------------------------------------------------------- live sessions
create table public.live_sessions (
  owner_id uuid primary key default auth.uid() references auth.users (id) on delete cascade,
  state jsonb not null default '{}'::jsonb,
  updated_at timestamptz not null default now()
);

-- ---------------------------------------------------------------- RLS
alter table public.profiles enable row level security;
alter table public.songs enable row level security;
alter table public.song_files enable row level security;
alter table public.setlists enable row level security;
alter table public.setlist_items enable row level security;
alter table public.song_requests enable row level security;
alter table public.live_sessions enable row level security;

create policy "own profile" on public.profiles for all to authenticated
  using (id = (select auth.uid())) with check (id = (select auth.uid()));
create policy "own songs" on public.songs for all to authenticated
  using (owner_id = (select auth.uid())) with check (owner_id = (select auth.uid()));
create policy "own song files" on public.song_files for all to authenticated
  using (owner_id = (select auth.uid())) with check (owner_id = (select auth.uid()));
create policy "own setlists" on public.setlists for all to authenticated
  using (owner_id = (select auth.uid())) with check (owner_id = (select auth.uid()));
create policy "own setlist items" on public.setlist_items for all to authenticated
  using (owner_id = (select auth.uid())) with check (owner_id = (select auth.uid()));
-- Requests are inserted only through submit_request(); the owner reads and manages them.
create policy "own requests read" on public.song_requests for select to authenticated
  using (owner_id = (select auth.uid()));
create policy "own requests update" on public.song_requests for update to authenticated
  using (owner_id = (select auth.uid())) with check (owner_id = (select auth.uid()));
create policy "own requests delete" on public.song_requests for delete to authenticated
  using (owner_id = (select auth.uid()));
create policy "own live session" on public.live_sessions for all to authenticated
  using (owner_id = (select auth.uid())) with check (owner_id = (select auth.uid()));

-- ---------------------------------------------------------------- accounts
-- Personal app: only the first account may register. Profile row is created automatically.
create or replace function public.handle_new_user()
returns trigger
language plpgsql
security definer
set search_path = ''
as $$
begin
  if exists (select 1 from auth.users where id <> new.id) then
    raise exception 'Registration is closed';
  end if;
  return new;
end;
$$;

create or replace function public.create_profile_for_user()
returns trigger
language plpgsql
security definer
set search_path = ''
as $$
begin
  insert into public.profiles (id) values (new.id) on conflict do nothing;
  return new;
end;
$$;

create trigger on_auth_user_guard before insert on auth.users
  for each row execute function public.handle_new_user();
create trigger on_auth_user_created after insert on auth.users
  for each row execute function public.create_profile_for_user();

-- ---------------------------------------------------------------- public RPCs
-- Audience request page: performer info + requestable songs, looked up by the QR token.
create or replace function public.request_catalog(p_token text)
returns jsonb
language sql
stable
security definer
set search_path = ''
as $$
  select jsonb_build_object(
    'open', p.requests_open,
    'performer', coalesce(p.display_name, ''),
    'message', p.request_message,
    'tip_url', p.tip_url,
    'songs', coalesce((
      select jsonb_agg(jsonb_build_object('id', s.id, 'title', s.title, 'artist', s.artist)
                       order by lower(s.title))
      from public.songs s
      where s.owner_id = p.id and s.deleted_at is null and s.requestable
    ), '[]'::jsonb)
  )
  from public.profiles p
  where p.request_token = p_token;
$$;

create or replace function public.submit_request(
  p_token text,
  p_song_id uuid,
  p_title text,
  p_artist text,
  p_name text,
  p_message text
)
returns jsonb
language plpgsql
volatile
security definer
set search_path = ''
as $$
declare
  v_owner uuid;
  v_open boolean;
  v_title text := left(btrim(coalesce(p_title, '')), 200);
  v_artist text := nullif(left(btrim(coalesce(p_artist, '')), 200), '');
  v_ip text;
  v_ip_hash text;
begin
  select id, requests_open into v_owner, v_open from public.profiles where request_token = p_token;
  if v_owner is null then
    return jsonb_build_object('ok', false, 'error', 'This request link is not valid.');
  end if;
  if not v_open then
    return jsonb_build_object('ok', false, 'error', 'Requests are closed right now.');
  end if;

  if p_song_id is not null then
    select s.title, s.artist into v_title, v_artist
    from public.songs s
    where s.id = p_song_id and s.owner_id = v_owner and s.deleted_at is null and s.requestable;
    if not found then
      return jsonb_build_object('ok', false, 'error', 'That song is not available.');
    end if;
  elsif v_title = '' then
    return jsonb_build_object('ok', false, 'error', 'Pick a song or type one in.');
  end if;

  v_ip := split_part(coalesce(current_setting('request.headers', true)::jsonb ->> 'x-forwarded-for', ''), ',', 1);
  v_ip_hash := encode(extensions.digest(v_owner::text || ':' || v_ip, 'sha256'), 'hex');

  if (select count(*) from public.song_requests
      where ip_hash = v_ip_hash and created_at > now() - interval '2 minutes') >= 3 then
    return jsonb_build_object('ok', false, 'error', 'Slow down a little - try again in a minute.');
  end if;
  if (select count(*) from public.song_requests
      where owner_id = v_owner and created_at > now() - interval '1 hour') >= 300 then
    return jsonb_build_object('ok', false, 'error', 'Requests are full right now.');
  end if;

  insert into public.song_requests (owner_id, song_id, title, artist, patron_name, message, ip_hash)
  values (
    v_owner, p_song_id, v_title, v_artist,
    nullif(left(btrim(coalesce(p_name, '')), 80), ''),
    nullif(left(btrim(coalesce(p_message, '')), 280), ''),
    v_ip_hash
  );
  return jsonb_build_object('ok', true);
end;
$$;

-- Band follow / lyrics display: current live state for a secret live token.
create or replace function public.get_live_state(p_token text)
returns jsonb
language sql
stable
security definer
set search_path = ''
as $$
  select coalesce(ls.state, '{}'::jsonb)
  from public.profiles p
  left join public.live_sessions ls on ls.owner_id = p.id
  where p.live_token = p_token;
$$;

-- Keep-alive target for the scheduled ping (free projects pause after a week idle).
create or replace function public.ping()
returns int
language sql
stable
set search_path = ''
as $$ select 1 $$;

revoke execute on function public.request_catalog(text) from public;
revoke execute on function public.submit_request(text, uuid, text, text, text, text) from public;
revoke execute on function public.get_live_state(text) from public;
revoke execute on function public.handle_new_user() from public, anon, authenticated;
revoke execute on function public.create_profile_for_user() from public, anon, authenticated;
grant execute on function public.request_catalog(text) to anon, authenticated;
grant execute on function public.submit_request(text, uuid, text, text, text, text) to anon, authenticated;
grant execute on function public.get_live_state(text) to anon, authenticated;
grant execute on function public.ping() to anon, authenticated;

-- ---------------------------------------------------------------- realtime
alter publication supabase_realtime add table
  public.song_requests, public.songs, public.song_files, public.setlists,
  public.setlist_items, public.profiles;

-- ---------------------------------------------------------------- storage
insert into storage.buckets (id, name, public, file_size_limit)
values ('song-files', 'song-files', false, 104857600)
on conflict (id) do nothing;

create policy "own song files read" on storage.objects for select to authenticated
  using (bucket_id = 'song-files' and (storage.foldername(name))[1] = (select auth.uid())::text);
create policy "own song files write" on storage.objects for insert to authenticated
  with check (bucket_id = 'song-files' and (storage.foldername(name))[1] = (select auth.uid())::text);
create policy "own song files update" on storage.objects for update to authenticated
  using (bucket_id = 'song-files' and (storage.foldername(name))[1] = (select auth.uid())::text);
create policy "own song files delete" on storage.objects for delete to authenticated
  using (bucket_id = 'song-files' and (storage.foldername(name))[1] = (select auth.uid())::text);
