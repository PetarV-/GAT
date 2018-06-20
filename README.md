# Webpage template for MILA-affiliated papers

[Template live demo](https://mila-udem.github.io/mila-paper-webpage)

# How to use

## Downloading the template

In the repo for which you wish to create a website, create an orphan branch
called `gh-pages`:

``` bash
$ cd [repository-name]
$ git checkout --orphan gh-pages
$ git rm -rf .
```

Download and unzip the template into the repository:

``` bash
$ wget https://github.com/mila-udem/mila-paper-webpage/archive/master.zip
$ unzip master.zip
$ mv mila-paper-webpage-master/* ./
$ rm -r master.zip mila-paper-webpage-master
```

Add all files as the first commit on this branch:

``` bash
$ git commit -a -m "First commit"
```

**Note: once you push to Github, even though your repository may be private, the
website will be publicly available to anyone using the webpage's web address.
You should develop locally until you're ready to go public.**

## Building the website locally

Follow [these instructions](https://help.github.com/articles/setting-up-your-github-pages-site-locally-with-jekyll/#requirements)
to install Ruby and Bundler.

Install the other plugins with

``` bash
$ bundle install
```

Edit `_config.yml` so that `baseurl` is an empty string, and launch the local
server with

``` bash
$ bundle exec jekyll serve -w
```

You can preview the webpage locally at

```
http://localhost:4000
```

## Configuring the template

All necessary configuration tweaks are done through `_config.yml`, which is
self-documented.

## Final step

Once you're ready, edit `_config.yml` so that `baseurl` is `"/repository-name"`,
commit, and push to Github:

``` bash
$ git push origin gh-pages
```

Your paper website should shortly be online at

```
https://[username].github.io/[repository-name]
```
