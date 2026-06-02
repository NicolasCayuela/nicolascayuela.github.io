# nicolascayuela.github.io

Personal academic homepage, based on the **Academic Homepage** Jekyll template
(fork of [Shengxiang-Lin/Shengxiang-Lin.github.io](https://github.com/Shengxiang-Lin/Shengxiang-Lin.github.io)).
Single-page layout: profile + education/experience/service/awards/news + selected publications.
Bilingual **EN / FR** (toggle in the navbar).

## Where to edit

| What | File |
|------|------|
| Name, bio, contacts, education, experience, awards, news | `_data/profile.yml` (`*_zh` fields = **French**) |
| UI labels (section titles, buttons) | `_data/i18n/en.yml` (English) and `_data/i18n/zh.yml` (**French**) |
| Navbar links | `_data/navigation.yml` |
| Site URL | `_config.yml` |
| Portrait photo | `assets/images/photos/profile.jpg` |
| Author display (bold name in pub lists) | `_data/authors.yml` |

> The template's second language channel is internally named `zh`, but here it
> holds **French** and the toggle shows **FR**. Leave the `lang-zh` / `_zh`
> names as-is; just write French in them.

## Add a publication

Create `_publications/<year>/<slug>.md`:

```yaml
---
title:    "Paper title"
date:     2025-05-12
selected: true            # true => also shown on the homepage
pub:      "Journal / Conference name"
cover:    assets/images/covers/paper1.png
authors:
  - Nicolas Cayuela       # names in _data/authors.yml render bold/linked
  - Co Author
links:
  Paper: https://arxiv.org/abs/xxxx.xxxxx
  Code:  https://github.com/NicolasCayuela/...
  Cite:  assets/bibtex/yourkey.bib
---
```

## Add a project (optional)

Uncomment the `project_card` block in `index.html`, then create
`_projects/<slug>.md`:

```yaml
---
layout: project
title: "Project name"
date: 2025-06-24
featured: true
image: "/assets/images/projects/yourproject.png"
description: "Short description."
links:
  Code: "https://github.com/NicolasCayuela/..."
---
```

## Google Scholar citation badge (optional)

1. Uncomment `gscholar:` in `_data/profile.yml` and set your Scholar user id.
2. In the repo settings add a variable/secret `GOOGLE_SCHOLAR_ID` with the same id.
   The daily workflow `.github/workflows/google_scholar_crawler.yaml` refreshes
   `google_scholar_crawler/results/`.

## Build & deploy

GitHub Pages builds this via **GitHub Actions** (needed because of the
`jekyll-email-protect` plugin). In **Settings → Pages → Build and deployment**,
set **Source = GitHub Actions**. The workflow `.github/workflows/jekyll.yml`
does the rest on every push to `main`/`master`.

Local preview (requires Ruby):

```bash
bundle install
bundle exec jekyll serve
```
