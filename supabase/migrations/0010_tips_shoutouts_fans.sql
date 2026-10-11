-- Tip-to-request: Venmo / Cash App / PayPal links with suggested amounts, tipped requests that
-- jump the queue, shoutouts, booking inquiries and an email list from the audience page.
-- Tips happen in the payment app; the request records what the patron said they sent.

alter table public.profiles
  add column request_options jsonb not null default '{}'::jsonb;

alter table public.song_requests drop constraint if exists song_requests_kind_check;
alter table public.song_requests
  add constraint song_requests_kind_check check (kind in ('request', 'karaoke', 'shoutout', 'booking')),
  add column tip_amount numeric(8, 2) check (tip_amount is null or (tip_amount > 0 and tip_amount <= 10000)),
  add column tip_method text check (tip_method is null or tip_method in ('venmo', 'cashapp', 'paypal', 'cash', 'other')),
  add column contact text,
  add column event_date date;

create table public.fans (
  id uuid primary key default gen_random_uuid(),
  owner_id uuid not null references auth.users (id) on delete cascade,
  email text not null,
  name text,
  source text,
  ip_hash text,
  created_at timestamptz not null default now()
);
create unique index fans_owner_email on public.fans (owner_id, lower(email));
create index fans_ip on public.fans (ip_hash, created_at desc);
alter table public.fans enable row level security;
create policy "own fans read" on public.fans for select to authenticated using (owner_id = (select auth.uid()));
create policy "own fans delete" on public.fans for delete to authenticated using (owner_id = (select auth.uid()));

-- Catalog: payment handles and which extras are switched on (act values win over the defaults)
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
    'performer', coalesce(nullif(a.act ->> 'name', ''), p.display_name, ''),
    'message', coalesce(nullif(a.act ->> 'message', ''), p.request_message),
    'tip_url', coalesce(nullif(a.act ->> 'tip_url', ''), p.tip_url),
    'pay', jsonb_build_object(
      'venmo', coalesce(nullif(a.act ->> 'venmo', ''), nullif(p.request_options ->> 'venmo', '')),
      'cashapp', coalesce(nullif(a.act ->> 'cashapp', ''), nullif(p.request_options ->> 'cashapp', '')),
      'paypal', coalesce(nullif(a.act ->> 'paypal', ''), nullif(p.request_options ->> 'paypal', ''))
    ),
    'amounts', coalesce(p.request_options -> 'amounts', '[5, 10, 20]'::jsonb),
    'priority', coalesce((p.request_options ->> 'priority')::boolean, true),
    'shoutouts', coalesce((p.request_options ->> 'shoutouts')::boolean, true),
    'email_list', coalesce((p.request_options ->> 'email_list')::boolean, true),
    'booking', coalesce((p.request_options ->> 'booking')::boolean, true),
    'songs', coalesce((
      select jsonb_agg(jsonb_build_object('id', s.id, 'title', s.title, 'artist', s.artist,
                                          'karaoke', s.karaoke, 'requestable', s.requestable)
                       order by lower(s.title))
      from public.songs s
      where s.owner_id = p.id and s.deleted_at is null and (s.requestable or s.karaoke)
    ), '[]'::jsonb)
  )
  from public.profiles p
  left join lateral (
    select x as act from jsonb_array_elements(p.acts) x where x ->> 'id' = p.active_act limit 1
  ) a on true
  where p.request_token = p_token;
$$;

drop function if exists public.submit_request(text, uuid, text, text, text, text, text);

create or replace function public.submit_request(
  p_token text,
  p_song_id uuid,
  p_title text,
  p_artist text,
  p_name text,
  p_message text,
  p_kind text default 'request',
  p_tip numeric default null,
  p_tip_method text default null,
  p_contact text default null,
  p_event_date date default null
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
  v_opts jsonb;
  v_kind text := coalesce(p_kind, 'request');
  v_title text := left(btrim(coalesce(p_title, '')), 200);
  v_artist text := nullif(left(btrim(coalesce(p_artist, '')), 200), '');
  v_name text := nullif(left(btrim(coalesce(p_name, '')), 80), '');
  v_message text := nullif(left(btrim(coalesce(p_message, '')), 600), '');
  v_contact text := nullif(left(btrim(coalesce(p_contact, '')), 200), '');
  v_tip numeric := case when p_tip > 0 and p_tip <= 10000 then round(p_tip, 2) end;
  v_method text := case when p_tip_method in ('venmo', 'cashapp', 'paypal', 'cash', 'other') then p_tip_method end;
  v_ip text;
  v_ip_hash text;
begin
  if v_kind not in ('request', 'karaoke', 'shoutout', 'booking') then
    return jsonb_build_object('ok', false, 'error', 'Unknown request type.');
  end if;
  select id, requests_open, karaoke_open, request_options into v_owner, v_open, v_karaoke_open, v_opts
  from public.profiles where request_token = p_token;
  if v_owner is null then
    return jsonb_build_object('ok', false, 'error', 'This request link is not valid.');
  end if;
  if v_kind in ('request', 'shoutout') and not v_open then
    return jsonb_build_object('ok', false, 'error', 'Requests are closed right now.');
  end if;
  if v_kind = 'karaoke' and not v_karaoke_open then
    return jsonb_build_object('ok', false, 'error', 'Karaoke sign-up is closed right now.');
  end if;
  if v_kind = 'shoutout' and not coalesce((v_opts ->> 'shoutouts')::boolean, true) then
    return jsonb_build_object('ok', false, 'error', 'Shoutouts are off tonight.');
  end if;
  if v_kind = 'booking' and not coalesce((v_opts ->> 'booking')::boolean, true) then
    return jsonb_build_object('ok', false, 'error', 'Booking requests are off.');
  end if;
  if v_kind = 'karaoke' and v_name is null then
    return jsonb_build_object('ok', false, 'error', 'Add your name so we can call you up!');
  end if;

  if v_kind = 'shoutout' then
    if v_message is null then
      return jsonb_build_object('ok', false, 'error', 'What should we say?');
    end if;
    v_title := 'Shoutout';
  elsif v_kind = 'booking' then
    if v_contact is null then
      return jsonb_build_object('ok', false, 'error', 'Add an email or phone number so we can reply.');
    end if;
    v_title := 'Booking inquiry';
    v_tip := null; v_method := null;
  elsif p_song_id is not null then
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

  insert into public.song_requests (owner_id, song_id, title, artist, patron_name, message, ip_hash, kind, position,
                                    tip_amount, tip_method, contact, event_date)
  values (
    v_owner, case when v_kind in ('request', 'karaoke') then p_song_id end, v_title, v_artist, v_name, v_message,
    v_ip_hash, v_kind,
    case when v_kind = 'karaoke' then extract(epoch from clock_timestamp()) end,
    v_tip, case when v_tip is not null then coalesce(v_method, 'other') end, v_contact, p_event_date
  );
  return jsonb_build_object('ok', true);
end;
$$;

-- Email list sign-up from the audience page (one row per address per performer)
create or replace function public.join_list(p_token text, p_email text, p_name text default null)
returns jsonb
language plpgsql
volatile
security definer
set search_path = ''
as $$
declare
  v_owner uuid;
  v_opts jsonb;
  v_email text := lower(btrim(coalesce(p_email, '')));
  v_ip text;
  v_ip_hash text;
begin
  select id, request_options into v_owner, v_opts from public.profiles where request_token = p_token;
  if v_owner is null then
    return jsonb_build_object('ok', false, 'error', 'This link is not valid.');
  end if;
  if not coalesce((v_opts ->> 'email_list')::boolean, true) then
    return jsonb_build_object('ok', false, 'error', 'Sign-up is closed.');
  end if;
  if v_email !~ '^[^@\s]+@[^@\s]+\.[^@\s]+$' or length(v_email) > 200 then
    return jsonb_build_object('ok', false, 'error', 'That email doesn''t look right.');
  end if;
  v_ip := split_part(coalesce(current_setting('request.headers', true)::jsonb ->> 'x-forwarded-for', ''), ',', 1);
  v_ip_hash := encode(extensions.digest(v_owner::text || ':' || v_ip, 'sha256'), 'hex');
  if (select count(*) from public.fans where ip_hash = v_ip_hash and created_at > now() - interval '10 minutes') >= 5 then
    return jsonb_build_object('ok', false, 'error', 'Slow down a little - try again in a few minutes.');
  end if;
  insert into public.fans (owner_id, email, name, source, ip_hash)
  values (v_owner, v_email, nullif(left(btrim(coalesce(p_name, '')), 80), ''), 'request page', v_ip_hash)
  on conflict (owner_id, lower(email)) do nothing;
  return jsonb_build_object('ok', true);
end;
$$;

revoke execute on function public.submit_request(text, uuid, text, text, text, text, text, numeric, text, text, date) from public;
revoke execute on function public.join_list(text, text, text) from public;
grant execute on function public.submit_request(text, uuid, text, text, text, text, text, numeric, text, text, date) to anon, authenticated;
grant execute on function public.join_list(text, text, text) to anon, authenticated;
