-- Acts: the performer plays solo and in a duo. Each act has its own name, message and tip link
-- on the audience page; a setlist belongs to an act, and starting Perform makes it the active act.

alter table public.profiles
  add column acts jsonb not null default '[]'::jsonb,
  add column active_act text;

alter table public.setlists
  add column act_id text;

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
