"""Smoke test of app.py against a stub `streamlit` (the real package is not installable in the
build sandbox). Verifies the script executes top to bottom, calls the service, and produces downloads."""
import runpy, sys, types, unittest
from tests.helpers import ROOT


class _Any:
    def __getattr__(self, k): return _Any()
    def __call__(self, *a, **k): return _Any()
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def __iter__(self): return iter([_Any(), _Any(), _Any(), _Any(), _Any()])


def make_stub(click_run: bool):
    st = types.ModuleType("streamlit")
    calls = {"download": 0, "dataframe": 0}
    st.session_state = {}
    st.secrets = {}
    st.set_page_config = lambda **k: None
    for n in ("title", "caption", "markdown", "subheader", "header", "info", "warning", "error", "write", "code", "bar_chart", "stop"):
        setattr(st, n, lambda *a, **k: None)
    st.stop = lambda: (_ for _ in ()).throw(SystemExit)
    st.sidebar = _Any()
    st.expander = lambda *a, **k: _Any()
    st.status = lambda *a, **k: _Any()
    st.progress = lambda *a, **k: _Any()
    class Col(_Any):
        def download_button(self, *a, **k): calls["download"] += 1
    st.columns = lambda spec, **k: [Col() for _ in range(spec if isinstance(spec, int) else len(spec))]
    st.selectbox = lambda label, options, index=0, format_func=None, **k: options[index] if options else None
    st.multiselect = lambda label, options, default=None, **k: default or []
    st.slider = lambda label, lo, hi, val=None, step=None, **k: val
    st.number_input = lambda label, lo, hi, val, step=None, **k: val
    st.text_input = lambda *a, **k: k.get("value", "")
    st.checkbox = lambda label, value=False, **k: value
    st.file_uploader = lambda *a, **k: []
    st.button = lambda *a, **k: click_run
    st.cache_data = lambda **k: (lambda f: f)
    def df(*a, **k): calls["dataframe"] += 1
    st.dataframe = df
    def dl(*a, **k): calls["download"] += 1
    st.download_button = dl
    st.metric = lambda *a, **k: None
    return st, calls


class AppSmoke(unittest.TestCase):
    def test_app_runs_with_sample(self):
        st, calls = make_stub(True)
        sys.modules["streamlit"] = st
        try:
            runpy.run_path(str(ROOT / "app.py"))
        finally:
            sys.modules.pop("streamlit", None)
        self.assertIn("result", st.session_state)
        self.assertGreater(len(st.session_state["result"].promos), 0)
        self.assertEqual(calls["download"], 2)      # CSV + JSON


if __name__ == "__main__":
    unittest.main()
