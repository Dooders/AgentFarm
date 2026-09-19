---
layout: page
title: Experiment writeups
subtitle: One page per experiment, covering the whole design and every variant run.
---

The [devlog](../devlog/index.md) is chronological field notes. This section is
the other way in: **one page per experiment**. Each writeup is meant to be
self-contained — question, shared protocol, every variant run, and the
synthesis — so you can follow a research line without hunting across posts
and design docs.

The [experiments catalog](../experiments-catalog.md) remains the runner / CLI
index for executing work. Use this page to *read* it.

<ul class="posts writeup-list">
{% assign writeups = site.pages | where: "layout", "experiment" %}
{% assign writeups = writeups | sort: "updated" | reverse %}
{% for writeup in writeups %}
  <li>
    <a class="post-card" href="{{ writeup.url | relative_url }}">
      <span class="post-card__meta">
        <span class="post-card__date">{{ writeup.updated | date: "%Y-%m-%d" }}</span>
        {% if writeup.status %}<span class="writeup-pill">{{ writeup.status | replace: "-", " " }}</span>{% endif %}
        {% if writeup.category %}<span class="writeup-pill">{{ writeup.category }}</span>{% endif %}
      </span>
      <h3 class="post-card__title">{{ writeup.title }}</h3>
      <p class="post-card__excerpt">{{ writeup.excerpt | strip }}</p>
      {% if writeup.variants %}
        <p class="post-card__excerpt">{{ writeup.variants | size }} variant run{% if writeup.variants.size != 1 %}s{% endif %} in this writeup.</p>
      {% endif %}
    </a>
  </li>
{% endfor %}
</ul>

## Adding a writeup

Create `docs/research/writeups/<slug>.md` with `layout: experiment`. Treat the
file as the **whole experiment**, not a single run:

1. Front matter: `title`, `subtitle`, `status`, `updated`, `category`,
   `excerpt`, and a `variants` list. Each variant needs an `id` and `title`.
2. Body: shared question and protocol first, then one `##` heading per variant
   run with a matching `{#id}` so the jump list works.
3. Each variant section should say what was varied, what actually ran
   (seeds, cells, CLI), and what it showed. Put the synthesis after the runs,
   not only in a linked post.
4. Link protocol docs, the catalog entry, and any related [devlog](../devlog/index.md)
   notes at the end — those stay supporting material.

Statuses used here: `complete` (ran, results reported), `implemented`
(runnable, limited or no published matrix), `in-progress`, `design`.
