# The `dsgelab` site

A static page in two languages, to be published at <https://gustavohm.github.io/dsgelab/>. It uses
no Jekyll, no build and no CDN dependency, only HTML pages, `style.css` and a thirty-line syntax
highlighter. It opens in a browser straight from disk and publishes to GitHub Pages without any extra step.

```
site/
  index.html                 the English page (default)
  replications.html          replications, in English
  pt-br/index.html           the Portuguese page
  pt-br/replications.html    replications, in Portuguese
  style.css                  light and dark theme, following the system preference
  highlight.js               Python and shell highlighting, no dependencies
  .nojekyll                  stops GitHub from processing the folder with Jekyll
```

## The two languages

The two versions of each page have the same sections, the same numbers and the same examples. In
the code blocks, comments, variable names and labels (file and constraint names) are all that
changes, and what the code computes stays the same. Each version has a language
selector at the top (`<p class="lang">`) and `hreflang` links pointing to the other, with English as the
`x-default`. The pages in `pt-br/` load `style.css` and `highlight.js` from `../`. A change to one
version has to be made in the other as well. The citation section differs on purpose, with ABNT in Portuguese and APA in English.

## Publishing

GitHub Pages can serve the site by any of three routes.

**1. A `docs/` folder in the repository.** Copy `site/` to `docs/` at the root and, under
*Settings → Pages*, choose *Deploy from a branch*, branch `main`, folder `/docs`.

```bash
# Settings → Pages → Deploy from a branch, branch main, folder /docs
# after copying site/ to docs/ at the repository root.
cp -r site docs
```

**2. A `gh-pages` branch.**

```bash
git subtree push --prefix site origin gh-pages
```

**3. Two packages, one site.** Publish a `docs/` holding `docs/dsgelab/` and the other package's folder side
by side, with an `index.html` at the root linking to both.

## Documentation links before publishing

The site links to `../docs/*.md` (`../../docs/*.md` in the `pt-br/` pages). Those links work in a cloned repository and **do not** work on a
published page, because GitHub Pages does not serve anything outside the published folder. Either of two changes fixes this.

- **Point to GitHub.** Replace `../docs/` with the URL of the files in the repository.

```bash
sed -i '' 's|href="../docs/|href="https://github.com/GustavoHM/dsgelab/blob/main/docs/|g' \
  site/index.html site/replications.html
sed -i '' 's|href="../../docs/|href="https://github.com/GustavoHM/dsgelab/blob/main/docs/|g' \
  site/pt-br/index.html site/pt-br/replications.html
```

- **Take the documentation along.** Copy `docs/` into the published folder. GitHub Pages
  serves `.md` files as plain text instead of rendering them as HTML. To get rendered pages, convert them first (`pandoc`,
  `mkdocs`) or rewrite the links to GitHub, which takes fewer steps.

## Adapting

There is a single stylesheet, with every colour defined as a CSS variable at the top. Changing `--accent` changes
the accent colour of links, highlighted keywords and the other accented elements across the whole page. The dark theme follows
`prefers-color-scheme` and requires no action from the reader.

## The examples are checked

Every executable code block of each HTML file in `site/`, in both languages (`index.html`,
`replications.html` and the copies in `pt-br/`), is extracted and run by
`tools/check_site_examples.py`. The blocks run in the order in which they appear and share one namespace
per page, because the examples form a sequence in which each block uses what the previous one defined.

```bash
python tools/check_site_examples.py
```

Run it after editing an example. An example that does not run invites the reader to copy broken
code and suggests that the documentation is not checked, which makes it worse than no example at all.

Blocks marked `class="plain"` (shell lines, syntax illustrations and excerpts that depend on
the user's data) are skipped on purpose. The mark is what declares them non-executable.
