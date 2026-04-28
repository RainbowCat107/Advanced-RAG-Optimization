"""
Windows compatibility shim for libraries that unconditionally import pwd.

The stdlib pwd module exists only on Unix. Some optional LangChain community
loaders import it during module discovery even when the project does not use
those loaders. Returning an "unknown" owner is enough for that optional path.
"""

from collections import namedtuple


struct_passwd = namedtuple(
    "struct_passwd",
    ["pw_name", "pw_passwd", "pw_uid", "pw_gid", "pw_gecos", "pw_dir", "pw_shell"],
)


def getpwuid(uid):
    return struct_passwd("unknown", "", uid, 0, "", "", "")

