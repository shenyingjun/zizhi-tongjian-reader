"""One-time vendoring of the canonical 百家姓 surname set into surnames_baijiaxing.json.
Source: kongnet/meeko fakeResource.json `firstName` field (the standard 百家姓
sequence: ~440 single 姓 followed by the closed 复姓 block). Run once; the output
JSON is committed so the build never needs network access.
"""
import json, urllib.request
from pathlib import Path

URL = ("https://raw.githubusercontent.com/kongnet/meeko/"
       "4a00c85bc9eefd7b2890e87f501478ab2b3a273e/lib/fake/fakeResource.json")
raw = urllib.request.urlopen(URL, timeout=30).read().decode("utf-8")
data = json.loads(raw)
seq = data["firstName"]

# The canonical 复姓 block (closed, well-known). Everything in `seq` that is not
# part of a 复姓 is a single 姓. We split by locating these known 2-char 姓.
COMPOUND = [
    "万俟","司马","上官","欧阳","夏侯","诸葛","闻人","东方","赫连","皇甫",
    "尉迟","公羊","澹台","公冶","宗政","濮阳","淳于","单于","太叔","申屠",
    "公孙","仲孙","轩辕","令狐","钟离","宇文","长孙","慕容","鲜于","闾丘",
    "司徒","司空","亓官","司寇","仉督","子车","颛孙","端木","巫马","公西",
    "漆雕","乐正","壤驷","公良","拓拔","夹谷","宰父","谷粱","段干","百里",
    "东郭","南门","呼延","归海","羊舌","微生","梁丘","左丘","东门","西门",
    "第五",
]
# Remove the 复姓 substrings, then the remaining chars are the single 姓.
s = seq
for c in COMPOUND:
    s = s.replace(c, "")
singles = sorted(set(s))

assert "柴" in singles, "柴 missing — source parse failed"
assert "慕容" in COMPOUND and "拓拔" in COMPOUND

out = {
    "_source": "百家姓 (canonical), via kongnet/meeko fakeResource.json firstName",
    "_note": "单姓 split by removing the closed 复姓 block; 复姓 listed explicitly.",
    "single": singles,
    "compound": sorted(set(COMPOUND)),
}
p = Path(__file__).resolve().parent / "surnames_baijiaxing.json"
p.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"singles={len(singles)} compounds={len(COMPOUND)} -> {p.name}")
