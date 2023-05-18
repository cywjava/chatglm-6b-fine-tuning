import re

item = "2. 有你才会更美好"
prefix = item[0:3]
print(prefix)

prefix = re.sub("\\d", "1", prefix)
item = "[SEP]" + prefix + item[3:]
print(item)
