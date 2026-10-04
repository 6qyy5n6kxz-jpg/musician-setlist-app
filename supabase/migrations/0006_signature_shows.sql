-- Signature shows: permanent, pinned setlists that get duplicated for individual gigs.
alter table public.setlists add column signature boolean not null default false;
