-- Live band karaoke: singer sign-ups (a second kind of audience request) and
-- per-song section timings recorded against a backing track.

alter table public.songs
  add column karaoke boolean not null default false,
  add column timings jsonb;

alter table public.profiles
  add column karaoke_open boolean not null default false;

alter table public.song_requests
  add column kind text not null default 'request' check (kind in ('request', 'karaoke')),
  add column position double precision;

-- Catalog now tells the audience page whether karaoke sign-up is open and which songs can be sung.
create or replace function public.request_catalog(p_token text)
returns jsonb
language sql
stable
security definer
set search_path = ''
as $$
  select jsonb_build_object(
    'open', p.requests_open,
    'karaoke_open', p.karaoke_open,
    'performer', coalesce(p.display_name, ''),
    'message', p.request_message,
    'tip_url', p.tip_url,
    'songs', coalesce((
      select jsonb_agg(jsonb_build_object('id', s.id, 'title', s.title, 'artist', s.artist, 'karaoke', s.karaoke)
                       order by lower(s.title))
      from public.songs s
      where s.owner_id = p.id and s.deleted_at is null and (s.requestable or s.karaoke)
    ), '[]'::jsonb)
  )
  from public.profiles p
  where p.request_token = p_token;
$$;

-- Singers waiting, in order (first names + songs are shown on the TV anyway).
create or replace function public.karaoke_lineup(p_token text)
returns jsonb
language sql
stable
security definer
set search_path = ''
as $$
  select coalesce(jsonb_agg(jsonb_build_object('name', r.patron_name, 'title', r.title)
                            order by r.position nulls last, r.created_at), '[]'::jsonb)
  from public.profiles p
  join public.song_requests r on r.owner_id = p.id
  where p.request_token = p_token and r.kind = 'karaoke' and r.status in ('new', 'queued')
    and r.created_at > now() - interval '18 hours';
$$;

drop function if exists public.submit_request(text, uuid, text, text, text, text);

create or replace function public.submit_request(
  p_token text,
  p_song_id uuid,
  p_title text,
  p_artist text,
  p_name text,
  p_message text,
  p_kind text default 'request'
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
  v_karaoke_open boolean;
  v_kind text := coalesce(p_kind, 'request');
  v_title text := left(btrim(coalesce(p_title, '')), 200);
  v_artist text := nullif(left(btrim(coalesce(p_artist, '')), 200), '');
  v_name text := nullif(left(btrim(coalesce(p_name, '')), 80), '');
  v_ip text;
  v_ip_hash text;
begin
  if v_kind not in ('request', 'karaoke') then
    return jsonb_build_object('ok', false, 'error', 'Unknown request type.');
  end if;
  select id, requests_open, karaoke_open into v_owner, v_open, v_karaoke_open
  from public.profiles where request_token = p_token;
  if v_owner is null then
    return jsonb_build_object('ok', false, 'error', 'This request link is not valid.');
  end if;
  if v_kind = 'request' and not v_open then
    return jsonb_build_object('ok', false, 'error', 'Requests are closed right now.');
  end if;
  if v_kind = 'karaoke' and not v_karaoke_open then
    return jsonb_build_object('ok', false, 'error', 'Karaoke sign-up is closed right now.');
  end if;
  if v_kind = 'karaoke' and v_name is null then
    return jsonb_build_object('ok', false, 'error', 'Add your name so we can call you up!');
  end if;

  if p_song_id is not null then
    select s.title, s.artist into v_title, v_artist
    from public.songs s
    where s.id = p_song_id and s.owner_id = v_owner and s.deleted_at is null
      and (case when v_kind = 'karaoke' then s.karaoke else s.requestable end);
    if not found then
      return jsonb_build_object('ok', false, 'error', 'That song is not available.');
    end if;
  elsif v_kind = 'karaoke' then
    return jsonb_build_object('ok', false, 'error', 'Pick a song from the karaoke list.');
  elsif v_title = '' then
    return jsonb_build_object('ok', false, 'error', 'Pick a song or type one in.');
  end if;

  v_ip := split_part(coalesce(current_setting('request.headers', true)::jsonb ->> 'x-forwarded-for', ''), ',', 1);
  v_ip_hash := encode(extensions.digest(v_owner::text || ':' || v_ip, 'sha256'), 'hex');

  if (select count(*) from public.song_requests
      where ip_hash = v_ip_hash and created_at > now() - interval '2 minutes') >= 10 then
    return jsonb_build_object('ok', false, 'error', 'Slow down a little - try again in a minute.');
  end if;
  if (select count(*) from public.song_requests
      where owner_id = v_owner and created_at > now() - interval '1 hour') >= 300 then
    return jsonb_build_object('ok', false, 'error', 'Requests are full right now.');
  end if;

  insert into public.song_requests (owner_id, song_id, title, artist, patron_name, message, ip_hash, kind, position)
  values (
    v_owner, p_song_id, v_title, v_artist, v_name,
    nullif(left(btrim(coalesce(p_message, '')), 280), ''),
    v_ip_hash, v_kind,
    case when v_kind = 'karaoke' then extract(epoch from clock_timestamp()) end
  );
  return jsonb_build_object('ok', true);
end;
$$;

revoke execute on function public.submit_request(text, uuid, text, text, text, text, text) from public;
revoke execute on function public.karaoke_lineup(text) from public;
grant execute on function public.submit_request(text, uuid, text, text, text, text, text) to anon, authenticated;
grant execute on function public.karaoke_lineup(text) to anon, authenticated;
