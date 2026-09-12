-- Schema for the recommendation test stand.
-- Sparse vectors are stored normalized (term -> profile), dense ones live
-- in pgvector. IDF and norms are recomputed by the refresh_stats() function.

create extension if not exists vector;

drop table if exists exposure cascade;
drop table if exists embeddings cascade;
drop table if exists profile_norms cascade;
drop table if exists profile_terms cascade;
drop table if exists terms cascade;
drop table if exists profiles cascade;

create table profiles (
  id        text primary key,
  name      text not null,
  city      text,            -- recall filter from the vision doc; absent in the dataset, filled in optionally
  give_raw  text not null,
  need_raw  text not null
);
create index on profiles (city);

-- A sparse vector element. kind='phrase' -- a whole phrase from the tag list,
-- kind='word' -- a single stemmed token (gives overlap for paraphrased but
-- semantically close tags).
create table terms (
  id    bigserial primary key,
  kind  text not null check (kind in ('phrase','word')),
  term  text not null,
  df    integer not null default 0,
  idf   double precision not null default 0,
  unique (kind, term)
);

create table profile_terms (
  profile_id text not null references profiles(id) on delete cascade,
  field      text not null check (field in ('give','need')),
  term_id    bigint not null references terms(id) on delete cascade,
  tf         double precision not null default 1,
  primary key (profile_id, field, term_id)
);
create index on profile_terms (term_id, field);

-- Vector length in the IDF metric: kept separately per (profile, field, kind).
create table profile_norms (
  profile_id text not null references profiles(id) on delete cascade,
  field      text not null,
  kind       text not null,
  norm       double precision not null,
  primary key (profile_id, field, kind)
);

create table embeddings (
  profile_id text not null references profiles(id) on delete cascade,
  field      text not null check (field in ('give','need')),
  emb        vector(384) not null,
  primary key (profile_id, field)
);

-- Show counter for the post-processing quota.
create table exposure (
  profile_id text primary key references profiles(id) on delete cascade,
  shows      integer not null default 0
);
