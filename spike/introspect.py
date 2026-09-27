"""Print the public API surface of qdrant_edge 0.8.0 so our adapter matches reality, not docs."""
import inspect
import qdrant_edge as qe

names = sorted(n for n in dir(qe) if not n.startswith("_"))
print("MODULE:", qe.__file__)
print("NAMES:", names)
for n in names:
    obj = getattr(qe, n)
    doc = (inspect.getdoc(obj) or "").split("\n")[0][:140]
    try:
        sig = str(inspect.signature(obj))
    except (TypeError, ValueError):
        sig = getattr(obj, "__text_signature__", None) or ""
    print(f"\n== {n} {sig}\n   {doc}")
    if inspect.isclass(obj):
        for m in sorted(x for x in dir(obj) if not x.startswith("_")):
            a = getattr(obj, m)
            s = getattr(a, "__text_signature__", None) or ""
            print(f"     .{m}{s}")
