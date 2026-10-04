-- Ink, highlighter and text notes drawn over a PDF chart (the PDF itself is never modified).
alter table public.song_files add column annotations jsonb not null default '{}'::jsonb;
