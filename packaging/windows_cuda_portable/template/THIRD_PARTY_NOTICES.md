# Third-party runtime notices

This portable test bundle contains third-party runtime components:

* CPython. Its license is included as `runtime/python/LICENSE.txt`.
* NumPy, SciPy, Narwhals, and LightGBM. Their package metadata and license
  files are included under `runtime/python/Lib/site-packages`.
* Microsoft Visual C++ and OpenMP redistributable DLLs. They are copied from
  the licensed Visual Studio redistributable directory used to build the
  bundle. Redistribution remains subject to the applicable Microsoft Visual
  Studio license terms. See
  <https://learn.microsoft.com/cpp/windows/redistributing-visual-cpp-files>.

Do not replace runtime DLLs with files downloaded from third-party DLL sites.
