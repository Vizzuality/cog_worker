# Contributing guidelines

TODO

# Develop

`git clone https://github.com/vizzuality/cog_worker.git`
`cd cog_worker`
`pip install -e .[test,dev,docs,distributed]`

## Docs

`cd sphinx_docs && make ghpages`

Note: requires `pandoc` to build convert jupyter notebooks.

## Release checklist

 1. Create release candidate branch
 2. Run tests `tox`
 3. Update changelog
 4. Bump to new version `bump2version [major/minor/patch]`
 5. Build docs `cd sphinx_docs && make ghpages`
 6. Commit docs
 7. Merge to `main`
 8. Publish to pypi `tox -e release`

