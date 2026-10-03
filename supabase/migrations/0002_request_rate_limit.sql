-- Patrons on shared venue Wi-Fi (or carrier NAT) share one IP: allow 10 requests per IP per 2 minutes.
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
      where ip_hash = v_ip_hash and created_at > now() - interval '2 minutes') >= 10 then
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
