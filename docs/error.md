root@localhost:/home/w00949828/ir/cann_pypto# pytest python/tests/ut/block/ --ignore=python/tests/ut/block/frontend/a5/
==================================================================================================================== test session starts =====================================================================================================================
platform linux -- Python 3.11.10, pytest-8.4.2, pluggy-1.6.0
rootdir: /npu/w00949828/ir/cann_pypto
configfile: pytest.ini
plugins: xdist-3.8.0, forked-1.6.0, anyio-4.11.0
collected 1963 items

python/tests/ut/block/backend/test_backend.py .........                                                                                                                                                                                                [  0%]
python/tests/ut/block/backend/test_backend_910b.py sssssss                                                                                                                                                                                             [  0%]
python/tests/ut/block/codegen/test_cce_codegen.py .................................                                                                                                                                                                    [  2%]
python/tests/ut/block/codegen/test_compact_tile.py .................                                                                                                                                                                                   [  3%]
python/tests/ut/block/codegen/test_dynamic_shape.py ..                                                                                                                                                                                                 [  3%]
python/tests/ut/block/codegen/test_orchestration_codegen.py ...............                                                                                                                                                                            [  4%]
python/tests/ut/block/codegen/test_pto_codegen.py ......................................................................................                                                                                                               [  8%]
python/tests/ut/block/codegen/test_pto_codegen_ops.py .                                                                                                                                                                                                [  8%]
python/tests/ut/block/codegen/test_pto_codegen_paged_attn.py .                                                                                                                                                                                         [  8%]
python/tests/ut/block/codegen/test_type_converter.py ...................                                                                                                                                                                               [  9%]
python/tests/ut/block/core/test_dtype.py ........................                                                                                                                                                                                      [ 10%]
python/tests/ut/block/core/test_error.py ..........................                                                                                                                                                                                    [ 12%]
python/tests/ut/block/core/test_logging.py ...........................................                                                                                                                                                                 [ 14%]
python/tests/ut/block/frontend/a3/debug/test_assert.py Fatal Python error: Aborted

Current thread 0x0000ffffa0cca140 (most recent call first):
  File "/usr/local/python3.11.10/lib/python3.11/site-packages/torch_npu/npu/utils.py", line 91 in set_device
  File "/npu/w00949828/ir/cann_pypto/python/tests/ut/block/frontend/a3/debug/test_assert.py", line 68 in test_assert
  File "/npu/w00949828/ir/cann_pypto/python/pypto_block/frontend/jit.py", line 1069 in wrapper
  File "/usr/local/python3.11.10/lib/python3.11/site-packages/_pytest/python.py", line 157 in pytest_pyfunc_call
  File "/usr/local/python3.11.10/lib/python3.11/site-packages/pluggy/_callers.py", line 121 in _multicall
  File "/usr/local/python3.11.10/lib/python3.11/site-packages/pluggy/_manager.py", line 120 in _hookexec
  File "/usr/local/python3.11.10/lib/python3.11/site-packages/pluggy/_hooks.py", line 512 in __call__
  File "/usr/local/python3.11.10/lib/python3.11/site-packages/_pytest/python.py", line 1671 in runtest
  File "/usr/local/python3.11.10/lib/python3.11/site-packages/_pytest/runner.py", line 178 in pytest_runtest_call
  File "/usr/local/python3.11.10/lib/python3.11/site-packages/pluggy/_callers.py", line 121 in _multicall
  File "/usr/local/python3.11.10/lib/python3.11/site-packages/pluggy/_manager.py", line 120 in _hookexec
  File "/usr/local/python3.11.10/lib/python3.11/site-packages/pluggy/_hooks.py", line 512 in __call__
  File "/usr/local/python3.11.10/lib/python3.11/site-packages/_pytest/runner.py", line 246 in <lambda>
  File "/usr/local/python3.11.10/lib/python3.11/site-packages/_pytest/runner.py", line 344 in from_call
  File "/usr/local/python3.11.10/lib/python3.11/site-packages/_pytest/runner.py", line 245 in call_and_report
  File "/usr/local/python3.11.10/lib/python3.11/site-packages/_pytest/runner.py", line 136 in runtestprotocol
  File "/usr/local/python3.11.10/lib/python3.11/site-packages/_pytest/runner.py", line 117 in pytest_runtest_protocol
  File "/usr/local/python3.11.10/lib/python3.11/site-packages/pluggy/_callers.py", line 121 in _multicall
  File "/usr/local/python3.11.10/lib/python3.11/site-packages/pluggy/_manager.py", line 120 in _hookexec
  File "/usr/local/python3.11.10/lib/python3.11/site-packages/pluggy/_hooks.py", line 512 in __call__
  File "/usr/local/python3.11.10/lib/python3.11/site-packages/_pytest/main.py", line 367 in pytest_runtestloop
  File "/usr/local/python3.11.10/lib/python3.11/site-packages/pluggy/_callers.py", line 121 in _multicall
  File "/usr/local/python3.11.10/lib/python3.11/site-packages/pluggy/_manager.py", line 120 in _hookexec
  File "/usr/local/python3.11.10/lib/python3.11/site-packages/pluggy/_hooks.py", line 512 in __call__
  File "/usr/local/python3.11.10/lib/python3.11/site-packages/_pytest/main.py", line 343 in _main
  File "/usr/local/python3.11.10/lib/python3.11/site-packages/_pytest/main.py", line 289 in wrap_session
  File "/usr/local/python3.11.10/lib/python3.11/site-packages/_pytest/main.py", line 336 in pytest_cmdline_main
  File "/usr/local/python3.11.10/lib/python3.11/site-packages/pluggy/_callers.py", line 121 in _multicall
  File "/usr/local/python3.11.10/lib/python3.11/site-packages/pluggy/_manager.py", line 120 in _hookexec
  File "/usr/local/python3.11.10/lib/python3.11/site-packages/pluggy/_hooks.py", line 512 in __call__
  File "/usr/local/python3.11.10/lib/python3.11/site-packages/_pytest/config/__init__.py", line 175 in main
  File "/usr/local/python3.11.10/lib/python3.11/site-packages/_pytest/config/__init__.py", line 201 in console_main
  File "/usr/local/python3.11.10/bin/pytest", line 8 in <module>

Extension modules: numpy.core._multiarray_umath, numpy.core._multiarray_tests, numpy.linalg._umath_linalg, numpy.fft._pocketfft_internal, numpy.random._common, numpy.random.bit_generator, numpy.random._bounded_integers, numpy.random._mt19937, numpy.random.mtrand, numpy.random._philox, numpy.random._pcg64, numpy.random._sfc64, numpy.random._generator, torch._C, torch._C._dynamo.autograd_compiler, torch._C._dynamo.eval_frame, torch._C._dynamo.guards, torch._C._dynamo.utils, torch._C._fft, torch._C._linalg, torch._C._nested, torch._C._nn, torch._C._sparse, torch._C._special, torch_npu._C, markupsafe._speedups, yaml._yaml (total: 27)
Aborted (core dumped)
