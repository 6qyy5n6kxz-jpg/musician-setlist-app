-- Audience page needs to know which songs are requestable vs karaoke-only.
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
      select jsonb_agg(jsonb_build_object('id', s.id, 'title', s.title, 'artist', s.artist,
                                          'karaoke', s.karaoke, 'requestable', s.requestable)
                       order by lower(s.title))
      from public.songs s
      where s.owner_id = p.id and s.deleted_at is null and (s.requestable or s.karaoke)
    ), '[]'::jsonb)
  )
  from public.profiles p
  where p.request_token = p_token;
$$;
