############
CONTRIBUTING
############

This is the guide for contributing code, documentation and tests, and for
filing issues. Please read it carefully to help make the code review
process go as smoothly as possible and maximize the likelihood of your
contribution being merged.

_Note:_  
If you want to contribute new functionality, you may first consider if this 
functionality belongs to the imagedata core, or is better suited for
a plugin or an application program.

If you're not sure where your contribution belongs,
create an issue where you can discuss this before creating a pull request.


------------
Found a Bug?
------------
If you find a bug in the source code, you can help us by
[submitting an issue](submit-issue_) to our [GitHub Repository][github_].
Even better, you can [submit a Pull Request](submit-pr_) with a fix.

------------------
Missing a Feature?
------------------
You can *request* a new feature by [submitting an issue](submit-issue_) to our GitHub Repository.
If you would like to *implement* a new feature, please submit an issue with a proposal for your work first, to be sure that we can use it. Please consider what kind of change it is:

* For a **Major Feature**, first open an issue and outline your proposal so that it can be
discussed. This will also allow us to better coordinate our efforts, prevent duplication of work,
and help you to craft the change so that it is successfully accepted into the project.
* **Small Features** can be crafted and directly [submitted as a Pull Request](submit-pr_).

-----------------------
Developer Documentation
-----------------------

See `Developer Documentation`_ for details on the structure of the code.

.. _submit:

---------------------
Submission Guidelines
---------------------

.. _submit-issue:

Submitting an Issue
-------------------

Before you submit an issue, please search the issue tracker, maybe an issue
for your problem already exists and the discussion might inform you of
workarounds readily available.

We want to fix all the issues as soon as possible, but before fixing a bug
we need to reproduce and confirm it.
In order to reproduce bugs, we will systematically ask you to provide a
minimal, complete, and verifiable example.
Having a minimal, complete, and verifiable example gives us a wealth of
important information without going back & forth to you with additional questions like:

- version of imagedata, pydicom and pynetdicom used
- and most importantly - a use-case that fails

We will be insisting on a minimal, complete, and verifiable example in order
to save maintainers time and ultimately be able to fix more bugs.
We understand that sometimes it might be hard to extract essentials bits
of code from a larger code-base but we really need to isolate the problem before we can fix it.

Unfortunately, we are not able to investigate / fix bugs without a minimal,
complete, and verifiable example, so if we don't hear back from you we
are going to close an issue that doesn't have enough info to be reproduced.

You can file new issues by filling out our
[new issue form](https://github.com/erling6232/imagedata/issues/new).

.. _submit-pr:

How to contribute
-----------------

The preferred workflow for contributing to imagedata is to fork the
[main repository](https://github.com/erling6232/imagedata) on
GitHub, clone, and develop on a branch. Steps:

1. Fork the [project repository](https://github.com/erling6232/imagedata)
   by clicking on the 'Fork' button near the top right of the page. This creates
   a copy of the code under your GitHub user account. For more details on
   how to fork a repository see [this guide](https://help.github.com/articles/fork-a-repo/).

2. Clone your fork of the imagedata repo from your GitHub account to your local disk:

.. code-block:: bash

   $ git clone git@github.com:YourLogin/imagedata.git
   $ cd imagedata

3. Create a ``feature`` branch to hold your development changes:

.. code-block:: bash

   $ git checkout -b my-feature

   Always use a ``feature`` branch. It's good practice to never work on the ``master`` branch!

4. Develop the feature on your feature branch. Add changed files using ``git add`` and then ``git commit`` files:

.. code-block:: bash

   $ git add modified_files
   $ git commit

5. Add a meaningful commit message. Pull requests are "squash-merged", e.g.
   squashed into one commit with all commit messages combined. The commit
   messages can be edited during the merge, but it helps if they are clearly
   and briefly showing what has been done in the commit. Check out the 
   [seven commonly accepted rules](https://www.theserverside.com/video/Follow-these-git-commit-message-guidelines)
   for commit messages.
   
6. To record your changes in Git, push the changes to your GitHub
   account with:

.. code-block:: bash

   $ git push -u origin my-feature

7. Follow [these instructions](https://help.github.com/articles/creating-a-pull-request-from-a-fork)
   to create a pull request from your fork. This will send an email to the committers.

(If any of the above seems like magic to you, please look up the
[Git documentation](https://git-scm.com/documentation) on the web, or ask a friend or another contributor for help.)

Pull Request Checklist
----------------------

We recommend that your contribution complies with the following rules before you
submit a pull request:

-  Follow the style used in the rest of the code. That mostly means to
   follow [PEP-8 guidelines](https://www.python.org/dev/peps/pep-0008/) for
   the code, and the [Google style](https://github.com/google/styleguide/blob/gh-pages/pyguide.md#38-comments-and-docstrings)
   for documentation.
   
-  If your pull request addresses an issue, please use the pull request title to
   describe the issue and mention the issue number in the pull request
   description. This will make sure a link back to the original issue is
   created. Use "closes #issue-number" or "fixes #issue-number" to let GitHub 
   automatically close the related issue on commit. Use any other keyword 
   (i.e. works on, related) to avoid GitHub to close the referenced issue.

-  All public methods should have informative docstrings with sample
   usage presented as doctests when appropriate.

-  Please prefix the title of your pull request with `[MRG]` (Ready for Merge),
   if the contribution is complete and ready for a detailed review. Some of the
   core developers will review your code, make suggestions for changes, and
   approve it as soon as it is ready for merge. Pull requests are usually merged
   after two approvals by core developers, or other developers asked to review the code. 
   An incomplete contribution -- where you expect to do more work before receiving a full
   review -- should be prefixed with `[WIP]` (to indicate a work in progress) and
   changed to `[MRG]` when it matures. WIPs may be useful to: indicate you are
   working on something to avoid duplicated work, request broad review of
   functionality or API, or seek collaborators. WIPs often benefit from the
   inclusion of a
   [task list](https://github.com/blog/1375-task-lists-in-gfm-issues-pulls-comments)
   in the PR description.

-  Documentation and high-coverage tests are necessary for enhancements to be
   accepted. Bug-fixes shall be provided with 
   [regression tests](https://en.wikipedia.org/wiki/regression_testing) that
   fail before the fix. For new features, the correct behavior shall be
   verified by feature tests. A good practice to write sufficient tests is 
   [test-driven development](https://en.wikipedia.org/wiki/Test-driven_development).

You can also check for common programming errors and style issues with the
following tools:

-  Code with good unittest **coverage** (current coverage or better), check with:

.. code-block:: bash

  $ pip install coverage
  $ coverage run -m unittest discover

-  No flake8 warnings, check with:

.. code-block:: bash

  $ pip install flake8
  $ flake8 .

Filing bugs
-----------
We use GitHub issues to track all bugs and feature requests; feel free to
open an issue if you have found a bug or wish to see a feature implemented.

It is recommended to check that your issue complies with the
following rules before submitting:

-  Verify that your issue is not being currently addressed by other
   [issues](https://github.com/erling6232/imagedata/issues?q=)
   or [pull requests](https://github.com/erling6232/imagedata/pulls?q=).

-  Please ensure all code snippets and error messages are formatted in
   appropriate code blocks.
   See [Creating and highlighting code blocks](https://help.github.com/articles/creating-and-highlighting-code-blocks).

-  Please include your operating system type and version number, as well
   as your Python, pydicom and imagedata versions.

   Please, run the following code snippet:

.. code-block:: python

   import platform, sys, pydicom, pynetdicom, imagedata
   print(platform.platform(),
         "\nPython", sys.version,
         "\npydicom", pydicom.__version__,
         "\npynetdicom", pynetdicom.__version__,
         "\nimagedata", imagedata.__version__)

-  please include a [reproducible](http://stackoverflow.com/help/mcve) code
   snippet or link to a [gist](https://gist.github.com). If an exception is
   raised, please provide the traceback. (use `%xmode` in ipython to use the
   non beautified version of the trackeback)


Documentation
-------------

We are glad to accept any sort of documentation: function docstrings,
reStructuredText documents, tutorials, etc.
reStructuredText documents live in the source code repository under the
``docs`` directory.

You can edit the documentation using any text editor and then generate
the HTML output by typing ``make html`` from the ``docs/`` directory.
Alternatively, ``make`` can be used to quickly generate the
documentation without the example gallery. The resulting HTML files will
be placed in ``_build/html/`` and are viewable in a web browser. See the
``README`` file in the ``docs/`` directory for more information.

For building the documentation, you will need
[sphinx](https://www.sphinx-doc.org/),
[numpy](http://numpy.org/),
[matplotlib](http://matplotlib.org/), and
[pillow](http://pillow.readthedocs.io/en/latest/).

When you are writing documentation that references DICOM, it is often
helpful to reference the related part of the
[DICOM standard](https://www.dicomstandard.org/current/). Try to make the
explanations intuitive and understandable also for users not fluent in DICOM.

.. _github: https://github.com/erling6232/imagedata
.. _Developer Documentation: https://imagedata.readthedocs.io/en/latest/DeveloperDocumentation.html