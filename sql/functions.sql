-- Recompute IDF and norms. Called after loading profiles.
-- idf = ln(1 + N/df): monotonically decreasing with frequency and never
-- zero, otherwise a mass-market tag would zero out the norm of a profile
-- made up entirely of mass-market tags.
create or replace function refresh_stats() returns void as $fn$
  update terms t
     set df = d.df,
         idf = ln(1 + (select count(*) from profiles)::double precision / d.df)
    from (select term_id, count(distinct profile_id) as df
            from profile_terms group by term_id) d
   where d.term_id = t.id;

  delete from profile_norms;

  insert into profile_norms (profile_id, field, kind, norm)
  select pt.profile_id, pt.field, tr.kind,
         sqrt(sum(power(pt.tf * tr.idf, 2)))
    from profile_terms pt
    join terms tr on tr.id = pt.term_id
   group by 1, 2, 3;
$fn$ language sql;

-- Asymmetric directed similarity, step 1 (recall) + step 2 (ranking) in one query.
-- p_self_field -- the viewer's field, p_other_field -- the candidate's field.
-- ('need','give') = "how useful is the candidate to the viewer",
-- ('give','need') = "how useful is the viewer to the candidate".
-- Cosine in the IDF metric: normalized by both vectors' lengths, so a
-- person with 50 tags doesn't become a match for everyone.
create or replace function directed_scores(
  p_profile     text,
  p_self_field  text,
  p_other_field text,
  p_city        text default null
) returns table (cand text, kind text, s double precision) as $fn$
  select pt_o.profile_id,
         tr.kind,
         sum(pt_s.tf * tr.idf * pt_o.tf * tr.idf) / (ns.norm * nb.norm)
    from profile_terms pt_s
    join terms tr          on tr.id = pt_s.term_id
    join profile_terms pt_o on pt_o.term_id = pt_s.term_id
                           and pt_o.field = p_other_field
                           and pt_o.profile_id <> p_profile
    join profiles pr        on pr.id = pt_o.profile_id
    join profile_norms ns   on ns.profile_id = p_profile
                           and ns.field = p_self_field and ns.kind = tr.kind
    join profile_norms nb   on nb.profile_id = pt_o.profile_id
                           and nb.field = p_other_field and nb.kind = tr.kind
   where pt_s.profile_id = p_profile
     and pt_s.field = p_self_field
     and (p_city is null or pr.city = p_city)
     and ns.norm > 0 and nb.norm > 0
   group by 1, 2, ns.norm, nb.norm;
$fn$ language sql stable;

-- Per-term contribution to the directed score -- used to explain a pair.
create or replace function pair_terms(
  p_viewer text, p_cand text, p_self_field text, p_other_field text
) returns table (kind text, term text, idf double precision, contrib double precision) as $fn$
  select tr.kind, tr.term, tr.idf,
         pt_s.tf * tr.idf * pt_o.tf * tr.idf / (ns.norm * nb.norm)
    from profile_terms pt_s
    join terms tr           on tr.id = pt_s.term_id
    join profile_terms pt_o on pt_o.term_id = pt_s.term_id
                           and pt_o.profile_id = p_cand and pt_o.field = p_other_field
    join profile_norms ns   on ns.profile_id = p_viewer
                           and ns.field = p_self_field and ns.kind = tr.kind
    join profile_norms nb   on nb.profile_id = p_cand
                           and nb.field = p_other_field and nb.kind = tr.kind
   where pt_s.profile_id = p_viewer and pt_s.field = p_self_field
   order by 4 desc;
$fn$ language sql stable;

-- Dense part: cosine similarity between the need and give field embeddings.
create or replace function dense_scores(
  p_profile text, p_self_field text, p_other_field text, p_city text default null
) returns table (cand text, s double precision) as $fn$
  select eo.profile_id, 1 - (es.emb <=> eo.emb)
    from embeddings es
    join embeddings eo on eo.field = p_other_field and eo.profile_id <> p_profile
    join profiles pr   on pr.id = eo.profile_id
   where es.profile_id = p_profile and es.field = p_self_field
     and (p_city is null or pr.city = p_city);
$fn$ language sql stable;
