# The `dsgelab` site

A static page in two languages, to be published at <https://gustavohm.github.io/dsgelab/>. It uses
no Jekyll, no build and no CDN dependency, only HTML pages, `style.css` and a short syntax
highlighter (`highlight.js`). It opens in a browser straight from disk, and the workflow described
under *Publishing* puts it on GitHub Pages.

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

`.github/workflows/pages.yml` publishes the site to GitHub Pages on every push to `main` that
changes `site/`, and on demand from the Actions tab. It needs *Settings → Pages → Build and
deployment → Source* set to *GitHub Actions*. GitHub Pages is free only for public repositories,
so the job is skipped while the repository is private.

The pages link to `../docs/*.md` (`../../docs/*.md` from `pt-br/`). Those links work in a clone,
but GitHub Pages serves nothing outside the published folder. The workflow therefore publishes a
copy of `site/` in which they point to the same files on GitHub, at the commit being published,
and it fails if any link to `docs/` is left relative.

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

Run it after editing an example. An example that does not run gives the reader broken code to copy
and shows that the documentation went unchecked.

Blocks marked `class="plain"` (shell lines, syntax illustrations and excerpts that depend on
the user's data) are skipped on purpose. The mark is what declares them non-executable.

## The output under each example

Under each block the page shows what it produces, in a `<pre class="output">`: what the block
prints, and the value of every expression that stands alone on a line, as an interactive session
shows it. Without it the reader has to guess what the program does. The output is not typed by
hand:

```bash
python tools/check_site_examples.py --write
```

runs the blocks and writes it into the pages, and the check without `--write` fails when the output
on a page is not what the code prints. Run `--write` after editing an example. Numbers are compared
with a tolerance, because the last digits of a residual at machine precision differ between
platforms.
