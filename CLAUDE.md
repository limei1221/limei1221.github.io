# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Personal Jekyll blog hosted on GitHub Pages. Posts are written in Markdown with YAML front matter and deployed automatically by pushing to GitHub.

## Commands

```bash
jekyll serve    # Serve locally with live reload (http://localhost:4000)
jekyll build    # Build static site to _site/ (not committed)
```

Ruby and the Jekyll gem must be installed. There is no Gemfile — install Jekyll directly: `gem install jekyll`.

## Architecture

**Content** lives in `_posts/` as `YYYY-MM-DD-title.markdown` files. Each post begins with YAML front matter:

```yaml
---
layout: post
title: "Post Title"
date: 2024-01-01 12:00:00
comments: true          # enables Disqus
mathjax: true           # enables MathJax (omit if no math)
excerpt: "Short summary shown in post listings"
---
```

**Template hierarchy:**
- `_layouts/default.html` — master HTML shell, includes head/header/footer
- `_layouts/post.html` — extends default; adds Disqus and conditional MathJax
- `_layouts/page.html` — extends default; for static pages like About
- `_includes/` — reusable partials (head, header, footer) used by layouts

**Site config** is `_config.yml`. Markdown processor is kramdown with GFM input and Rouge syntax highlighting. URLs use `permalink: pretty` (e.g., `/2024/01/01/post-title/`).

**Assets** go in `assets/<post-name>/`. The `_site/` and `_drafts/` directories are gitignored.
