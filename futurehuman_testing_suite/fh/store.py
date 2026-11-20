"""Reference store + deletion audit.  The REAL product must plug its own stores into DeletionAudit.
Tests the manual's requirement: 'deletion propagation across all storage systems' and per-user isolation."""
import sqlite3


class Store:
    def __init__(self, leaky=None):
        self.leaky = set(leaky or [])  # simulate a bug: which store forgets to delete ('vectors','cache','log')
        self.db = sqlite3.connect(":memory:")
        self.db.executescript("create table captures(id integer primary key, user text, ts real, text text);"
                              "create table memories(id integer primary key, user text, ts real, text text, capture_id integer);")
        self.vectors, self.cache, self.log = [], {}, []

    def add_capture(self, user, text, ts):
        cid = self.db.execute("insert into captures(user,ts,text) values(?,?,?)", (user, ts, text)).lastrowid
        mid = self.db.execute("insert into memories(user,ts,text,capture_id) values(?,?,?,?)", (user, ts, "memory: " + text, cid)).lastrowid
        self.vectors.append(dict(user=user, ts=ts, text=text, mem_id=mid))
        self.cache[f"{user}:{cid}"] = dict(user=user, ts=ts, text=text)
        self.log.append(f"add user={user} capture={cid}")  # logs never contain content
        return cid

    def search(self, user, needle):
        """Every read is scoped to one user (per-user isolation)."""
        return [r[0] for r in self.db.execute("select text from captures where user=? and text like ?", (user, f"%{needle}%"))]

    def delete_range(self, user, t0, t1):
        self.db.execute("delete from captures where user=? and ts between ? and ?", (user, t0, t1))
        self.db.execute("delete from memories where user=? and ts between ? and ?", (user, t0, t1))
        if "vectors" not in self.leaky:
            self.vectors = [v for v in self.vectors if not (v["user"] == user and t0 <= v["ts"] <= t1)]
        if "cache" not in self.leaky:
            self.cache = {k: v for k, v in self.cache.items() if not (v["user"] == user and t0 <= v["ts"] <= t1)}
        self.log.append(f"delete user={user} range=({t0},{t1})")

    def delete_account(self, user):
        self.delete_range(user, float("-inf"), float("inf"))

    def residue(self, needle):
        """Where does this text still exist? (must be empty after a deletion)"""
        hits = []
        for t in ("captures", "memories"):
            if self.db.execute(f"select 1 from {t} where text like ?", (f"%{needle}%",)).fetchone():
                hits.append(t)
        if any(needle in v["text"] for v in self.vectors):
            hits.append("vectors")
        if any(needle in v["text"] for v in self.cache.values()):
            hits.append("cache")
        if any(needle in l for l in self.log):
            hits.append("log")
        return hits
