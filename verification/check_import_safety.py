# Import-safety gate for the verification suite.
#
# WHY THIS EXISTS. On 2026-07-27 a commit whose message was "harden the suite" appended a
# bare module-level `sys.exit(0)` to criterion.py. Python runs a module's whole body on
# import, so every script that did `import criterion` -- nine of them, directly or through
# evend_frame_probe -- terminated DURING THE IMPORT, ran none of its own checks, printed
# criterion.py's "criterion PASS" line, and exited 0.
#
# Nine scripts were dead. All nine looked green. run_all.sh could not see it, because the
# verdict token it grepped for was genuinely present in the output -- it just belonged to a
# different script. The mathematics was fine; the harness was lying.
#
# THE RULE THIS ENFORCES. A module that is imported anywhere in the suite must not, at
# import time, call sys.exit / exit / quit / os._exit, nor raise SystemExit. Falling off
# the end of a script already exits 0; a bare exit buys nothing and costs this.
#
# HISTORY OF THIS FILE. Its first version (same day) tested for a __main__ guard with
# `"__main__" in src` -- a bare substring test that SKIPPED THE SCAN ENTIRELY for any file
# merely containing that string, whether or not the exit was inside the guard. Seven of the
# thirteen imported modules were exempted by accident, including arf_global, which sits one
# link further up the same import chain as criterion. A gate with a hole in exactly the
# place it was written to cover is worse than no gate, because it is trusted. The check
# below therefore resolves guards STRUCTURALLY, follows module-level calls into
# module-level helper functions, and handles import aliasing.
#
# Expect: "check_import_safety PASS" and exit 0. Any violation exits 1 and names the file.

import ast
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BARE_EXITERS = {"exit", "quit"}
DOTTED_EXITERS = {("sys", "exit"), ("os", "_exit")}


def is_main_guard(node):
    """True for `if __name__ == '__main__':` -- the one place an exit is legitimate."""
    if not isinstance(node, ast.If):
        return False
    t = node.test
    if not isinstance(t, ast.Compare) or len(t.ops) != 1 or not isinstance(t.ops[0], ast.Eq):
        return False
    left, right = t.left, t.comparators[0]
    def is_name_dunder(n):
        return isinstance(n, ast.Name) and n.id == "__name__"
    def is_main_str(n):
        return isinstance(n, ast.Constant) and n.value == "__main__"
    return (is_name_dunder(left) and is_main_str(right)) or \
           (is_name_dunder(right) and is_main_str(left))


class Analyzer:
    """Find exit-like effects reachable at IMPORT time.

    Import-time flow = the module body, minus anything inside a __main__ guard, minus
    function and class bodies -- EXCEPT that a module-level call to a module-level
    function does reach that function's body, so those are followed (one level of
    recursion per function, cycles cut).
    """

    def __init__(self, tree):
        self.tree = tree
        self.exit_aliases = set(BARE_EXITERS)   # local names bound to an exit callable
        self.mod_aliases = {}                   # local module name -> real module name
        self.funcs = {}                         # module-level def name -> node
        self._collect_bindings()

    def _collect_bindings(self):
        for node in self.tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self.funcs[node.name] = node
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    self.mod_aliases[a.asname or a.name] = a.name.split(".")[0]
            elif isinstance(node, ast.ImportFrom) and node.module:
                base = node.module.split(".")[0]
                for a in node.names:
                    if (base, a.name) in DOTTED_EXITERS:
                        self.exit_aliases.add(a.asname or a.name)

    def _exit_kind(self, call):
        f = call.func
        if isinstance(f, ast.Name) and f.id in self.exit_aliases:
            return f.id
        if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name):
            real = self.mod_aliases.get(f.value.id, f.value.id)
            if (real, f.attr) in DOTTED_EXITERS:
                return f"{f.value.id}.{f.attr}" + ("" if real == f.value.id else f" (= {real}.{f.attr})")
        return None

    def _scan_stmts(self, body, seen, via):
        """Yield (lineno, description) for exit effects reachable from these statements."""
        hits = []
        for node in body:
            if is_main_guard(node):
                continue                                   # legitimately guarded
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue                                   # only reached if called
            hits.extend(self._scan_node(node, seen, via))
        return hits

    def _scan_node(self, node, seen, via):
        hits = []
        # descend, but do not walk into nested defs/classes or nested __main__ guards
        stack = [node]
        while stack:
            cur = stack.pop()
            for child in ast.iter_child_nodes(cur):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    continue
                if is_main_guard(child):
                    continue
                stack.append(child)
            if isinstance(cur, ast.Call):
                kind = self._exit_kind(cur)
                if kind:
                    hits.append((cur.lineno, f"`{kind}()`" + via))
                elif isinstance(cur.func, ast.Name) and cur.func.id in self.funcs:
                    name = cur.func.id
                    if name not in seen:
                        inner = self._scan_stmts(
                            self.funcs[name].body, seen | {name},
                            via + f", via the module-level call to {name}() on line {cur.lineno}",
                        )
                        hits.extend(inner)
            if isinstance(cur, ast.Raise):
                exc = cur.exc
                nm = None
                if isinstance(exc, ast.Name):
                    nm = exc.id
                elif isinstance(exc, ast.Call) and isinstance(exc.func, ast.Name):
                    nm = exc.func.id
                if nm == "SystemExit":
                    hits.append((cur.lineno, "`raise SystemExit`" + via))
        return hits

    def import_time_exits(self):
        return sorted(set(self._scan_stmts(self.tree.body, frozenset(), "")))


def main():
    files = sorted(f for f in os.listdir(HERE) if f.endswith(".py"))
    trees = {}
    for f in files:
        with open(os.path.join(HERE, f), encoding="utf-8") as fh:
            src = fh.read()
        try:
            trees[f] = ast.parse(src, filename=f)
        except SyntaxError as e:
            print(f"  FAILED CHECK: syntax error in {f}: {e}")
            print("check_import_safety FAIL")
            return 1

    # Which modules are imported by some other module here. Imports nested inside
    # functions count: they execute the module body just the same, when called.
    mods = {f[:-3] for f in files}
    imported_by = {m: set() for m in mods}
    for f, tree in trees.items():
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                names = [node.module.split(".")[0]]
            for n in names:
                if n in mods and n != f[:-3]:
                    imported_by[n].add(f)

    violations = []
    scanned = 0
    for f, tree in trees.items():
        importers = imported_by[f[:-3]]
        if not importers:
            continue
        scanned += 1
        for lineno, what in Analyzer(tree).import_time_exits():
            violations.append(
                f"{f}:{lineno}: {what} runs at IMPORT time, and {f[:-3]} is imported by "
                f"{', '.join(sorted(importers))} -- this kills the importer silently at status 0"
            )

    print(f"scanned {len(files)} scripts; {scanned} are imported by another script "
          f"and were checked for import-time exits")
    for m in sorted(m for m in mods if imported_by[m]):
        print(f"  {m}.py  <- imported by {', '.join(sorted(imported_by[m]))}")

    print("\n--- verdict ---")
    for v in violations:
        print(f"  FAILED CHECK: {v}")
    if violations:
        print("check_import_safety FAIL")
        return 1
    print(f"check_import_safety: {scanned}/{scanned} imported modules are import-safe")
    print("check_import_safety PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
